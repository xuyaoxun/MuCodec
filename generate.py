import json
import torch
from tqdm import tqdm
import sys
from model import PromptCondAudioDiffusion
from diffusers import DDIMScheduler, DDPMScheduler
import torchaudio
import librosa
import os
import math
import numpy as np
from tools.get_melvaehifigan48k import build_pretrained_models
import tools.torch_tools as torch_tools
from safetensors.torch import load_file

class MuCodec:
    def __init__(self, \
        model_path, \
        layer_num, \
        load_main_model=True, \
        device="cuda:0"):
        
        self.layer_num = layer_num - 1
        self.sample_rate = 48000
        self.device = device

        self.MAX_DURATION = 360
        if load_main_model:
            audio_ldm_path = os.path.dirname(os.path.abspath(__file__)) + "/tools/audioldm_48k.pth"
            self.vae, self.stft = build_pretrained_models(audio_ldm_path)
            self.vae, self.stft = self.vae.eval().to(device), self.stft.eval().to(device)
            main_config = {
                "num_channels":32,
                "unet_model_name":None,
                "unet_model_config_path":os.path.dirname(os.path.abspath(__file__)) + "/configs/models/transformer2D.json",
                "snr_gamma":None,
            }
            self.model = PromptCondAudioDiffusion(**main_config)
            if model_path.endswith('.safetensors'):
                main_weights = load_file(model_path)
            else:
                main_weights = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(main_weights, strict=False)
            self.model = self.model.to(device)
            print ("Successfully loaded checkpoint from:", model_path)
        else:
            main_config = {
                "num_channels":32,
                "unet_model_name":None,
                "unet_model_config_path":None,
                "snr_gamma":None,
            }
            self.model = PromptCondAudioDiffusion(**main_config).to(device)
            main_weights = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(main_weights, strict=False)
            self.model = self.model.to(device)
            print ("Successfully loaded checkpoint from:", model_path)
        
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        print("scaling factor: ", self.model.normfeat.std)

    def file2code(self, fname):
        orig_samples, fs = torchaudio.load(fname)
        if(fs!=self.sample_rate):
            orig_samples = torchaudio.functional.resample(orig_samples, fs, self.sample_rate)
            fs = self.sample_rate
        if orig_samples.shape[0] == 1:
            orig_samples = torch.cat([orig_samples, orig_samples], 0)
        return self.sound2code(orig_samples)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code(self, orig_samples, batch_size=3):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        audios = self.preprocess_audio(audios)
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40.96 * self.sample_rate)
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples + 480):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        # print("int_max_len: ", int_max_len)
        audios = torch.cat([audios, audios], -1)
        # print("audios:",audios.shape)
        audios=audios[:,:int(int_max_len*(min_samples+480))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples+480).permute(1, 0, 2).reshape(-1, 2, min_samples+480)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes, _, spk_embeds = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num)
            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(1, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    def code2sound(self, codes, prompt=None, duration=40.96, guidance_scale=1.5, num_steps=20, disable_progress=False):
        codes = codes.to(self.device)
        first_latent = torch.randn(codes.shape[0], 32, 512, 32).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0
        if(isinstance(prompt, torch.Tensor)):
            prompt = prompt.to(self.device)
            if(prompt.ndim == 3):
                assert prompt.shape[0] == 1, prompt.shape
                prompt = prompt[0]
            elif(prompt.ndim == 1):
                prompt = prompt.unsqueeze(0).repeat(2,1)
            elif(prompt.ndim == 2):
                if(prompt.shape[0] == 1):
                    prompt = prompt.repeat(2,1)

            if(prompt.shape[-1] < int(30.76 * self.sample_rate)):
                prompt = prompt[:,:int(10.24*self.sample_rate)] # limit max length to 10.24
            else:
                prompt = prompt[:,int(20.48*self.sample_rate):int(30.72*self.sample_rate)] # limit max length to 10.24
            
            true_mel , _, _ = torch_tools.wav_to_fbank2(prompt, -1, fn_STFT=self.stft) # maximum 10.24s
            true_mel = true_mel.unsqueeze(1).to(self.device)
            true_latent = torch.cat([self.vae.get_first_stage_encoding(self.vae.encode_first_stage(true_mel[[m]])) for m in range(true_mel.shape[0])],0)
            true_latent = true_latent.reshape(true_latent.shape[0]//2, -1, true_latent.shape[2], true_latent.shape[3]).detach()
            
            first_latent[:,:,0:true_latent.shape[2],:] = true_latent
            first_latent_length = true_latent.shape[2]
            first_latent_codes = self.sound2code(prompt)[:,:,0:first_latent_length*2] # B 4 T
            first_latent_codes_length = first_latent_codes.shape[-1]
            codes = torch.cat([first_latent_codes, codes], -1)

        min_samples = 1024
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples // 2
        ovlp_frames = ovlp_samples // 2

        codes_len= codes.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * self.sample_rate)

        if(codes_len < min_samples):
            while(codes.shape[-1] < min_samples):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:min_samples]
        codes_len = codes.shape[-1]
        if((codes_len - ovlp_frames) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(codes.shape[-1] < len_codes):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:len_codes]
        latent_length = 512
        latent_list = []
        spk_embeds = torch.zeros([1, 32, 1, 32], device=codes.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for sinx in range(0, codes.shape[-1]-hop_samples, hop_samples):
                codes_input=[]
                codes_input.append(codes[:,:,sinx:sinx+min_samples])
                if(sinx == 0):
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    true_latent = latent_list[-1][:,:,-ovlp_frames:,:]
                    len_add_to_512 = 512 - true_latent.shape[-2]
                    incontext_length = true_latent.shape[-2]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0], true_latent.shape[1], len_add_to_512, true_latent.shape[-1]).to(self.device)], -2)
                    latents = self.model.inference_codes(codes_input, spk_embeds, true_latent, latent_length, incontext_length,  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,:,first_latent_length:,:]
        min_samples =  int(duration * self.sample_rate)
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                bsz , ch, t, f = latent.shape
                latent = latent.reshape(bsz*2, ch//2, t, f)
                mel = self.vae.decode_first_stage(latent)
                cur_output = self.vae.decode_to_waveform(mel)
                cur_output = torch.from_numpy(cur_output)[:, 0:min_samples]

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    @torch.no_grad()
    def preprocess_audio(self, input_audios, threshold=0.8):
        assert len(input_audios.shape) == 3, input_audios.shape
        nchan = input_audios.shape[1]
        input_audios = input_audios.reshape(input_audios.shape[0], -1)
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios.reshape(input_audios.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)
    
    @torch.no_grad()
    def sound2sound(self, sound, prompt=None, min_duration=40.96, steps=50, disable_progress=False):
        codes = self.sound2code(sound)
        wave = self.code2sound(codes, prompt, duration=min_duration, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        return wave

if __name__=="__main__":
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/mucodec.pt")
    mucodec = MuCodec(model_path=ckpt_path,layer_num=7,load_main_model=True)

    filelist = []

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_wav")
    for f in [os.path.join(root_dir, f) for f in os.listdir(root_dir) if '.flac' in f or '.wav' in f or '.mp3' in f]:
        a, fs = torchaudio.load(f)
        if(fs!=48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if(a.shape[0]==1):
            a = torch.cat([a,a],0)
        ori_len = a.shape[-1]
        filelist.append([a, '', [0, a.shape[-1]/48000.], f,ori_len])
    
    reconstructed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstructed")

    os.makedirs(reconstructed_dir, exist_ok=True)

    for sample_idx, (orig_samples, lyric, st_et, fname,ori_len) in enumerate(filelist):
        print(fname, lyric)
        wave = mucodec.sound2sound(orig_samples,None)
        wave = wave[:,0:ori_len]
        torchaudio.save(os.path.join(reconstructed_dir, os.path.basename(fname)),wave.detach().cpu(), 48000)
    