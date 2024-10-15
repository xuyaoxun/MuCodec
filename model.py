import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
import typing as tp
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from einops import repeat
from tools.torch_tools import wav_to_fbank
import os
import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from models.transformer_2d_flow import Transformer2DModel
from libs.rvq.descript_quantize3 import ResidualVectorQuantize
from torch.cuda.amp import autocast
from muq_dev.test import load_model




class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor):
        """Project back from diffusion space to the actual sample space."""
        return z

class Feature2DProcessor(SampleProcessor):
    def __init__(self, dim: int = 8, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1., \
                 num_samples: int = 100_000):
        super().__init__()
        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim, 32))
        self.register_buffer('sum_x2', torch.zeros(dim, 32))
        self.register_buffer('sum_target_x2', torch.zeros(dim, 32))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 4
        if self.counts.item() < self.num_samples:
            self.counts += len(x)
            self.sum_x += x.mean(dim=(2,)).sum(dim=0)
            self.sum_x2 += x.pow(2).mean(dim=(2,)).sum(dim=0)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, 1, 32).contiguous()) * rescale.view(1, -1, 1, 32).contiguous()
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 4
        rescale = (self.std / self.target_std) ** self.power_std
        x = x * rescale.view(1, -1, 1, 32).contiguous() + self.mean.view(1, -1, 1, 32).contiguous()
        return x

def pad_or_tunc_tolen(prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds, len_size=77):
    if(prior_text_encoder_hidden_states.shape[1]<len_size):
        prior_text_encoder_hidden_states = torch.cat([prior_text_encoder_hidden_states, \
            torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], \
            prior_text_encoder_hidden_states.shape[2], device=prior_text_mask.device, \
            dtype=prior_text_encoder_hidden_states.dtype)],1)
        prior_text_mask = torch.cat([prior_text_mask, torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], device=prior_text_mask.device, dtype=prior_text_mask.dtype)],1)
    else:
        prior_text_encoder_hidden_states = prior_text_encoder_hidden_states[:,0:len_size]
        prior_text_mask = prior_text_mask[:,0:len_size]
    prior_text_encoder_hidden_states = prior_text_encoder_hidden_states.permute(0,2,1).contiguous()
    return prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        estimator,
    ):
        super().__init__()
        self.sigma_min = 1e-4

        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span)

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, added_cond_kwargs, guidance_scale):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in tqdm(range(1, len(t_span))):
            x[:,:,0:incontext_length,:] = (1 - (1 - self.sigma_min) * t) * noise[:,:,0:incontext_length,:] + t * incontext_x[:,:,0:incontext_length,:]
            if(guidance_scale > 1.0):
                dphi_dt = self.estimator( \
                    torch.cat([ \
                        torch.cat([x, x], 0), \
                        torch.cat([incontext_x, incontext_x], 0), \
                        torch.cat([torch.zeros_like(mu), mu], 0), \
                        ], 1), \
                timestep = t.unsqueeze(-1).repeat(2), \
                added_cond_kwargs={k:torch.cat([v,v],0) for k,v in added_cond_kwargs.items()}).sample
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2,0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dhpi_dt_cond - dphi_dt_uncond)
            else:
                dphi_dt = self.estimator(torch.cat([x, incontext_x, mu], 1), \
                timestep = t.unsqueeze(-1),
                added_cond_kwargs=added_cond_kwargs).sample

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mu, added_cond_kwargs, latent_masks, validation_mode=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        b = mu[0].shape[0]

        # random timestep
        if(validation_mode):
            t = torch.ones([b, 1, 1, 1], device=mu[0].device, dtype=mu[0].dtype) * 0.5
        else:
            t = torch.rand([b, 1, 1, 1], device=mu[0].device, dtype=mu[0].dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        out = self.estimator(
            torch.cat([y, *mu],1), 
            timestep = t.squeeze(-1).squeeze(-1).squeeze(-1),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        weight = (latent_masks > 1.5).unsqueeze(1).unsqueeze(-1).repeat(1, out.shape[1], 1, out.shape[-1]).float() + (latent_masks < 0.5).unsqueeze(1).unsqueeze(-1).repeat(1, out.shape[1], 1, out.shape[-1]).float() * 0.01

        loss = F.mse_loss(out * weight, u * weight, reduction="sum") / weight.sum()
        return loss

class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self,
        num_channels,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        uncondition=True,
        out_paint=False,
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.num_channels = num_channels

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.normfeat = Feature2DProcessor(dim=num_channels)

        self.sample_rate = 48000
        self.rsp48toclap = torchaudio.transforms.Resample(48000, 24000)
        self.rsq48towav2vec = torchaudio.transforms.Resample(48000, 16000)
        muencoder_dir = "muq_dev/muq_fairseq"
        muencoder_ckpt = "muq_dev/muq.pt"

        self.muencoder = load_model(
            model_dir=os.path.abspath(muencoder_dir),
            checkpoint_dir=os.path.abspath(muencoder_ckpt),
        )
        self.rsq48tomuencoder = torchaudio.transforms.Resample(48000, 24000)
        for v in self.muencoder.parameters():v.requires_grad = False
        self.rvq_muencoder_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        self.cond_muencoder_emb = nn.Linear(1024, 16*32)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(32*32,))

        unet = Transformer2DModel.from_config(
            unet_model_config_path,
        )
        self.set_from = "random"
        self.cfm_wrapper = BASECFM(unet)
        print("Transformer initialized from pretrain.")


    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def preprocess_audio(self, input_audios, threshold=0.9):
        assert len(input_audios.shape) == 2, input_audios.shape
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios/norm_value.unsqueeze(-1)



    
    def extract_muencoder_embeds(self, input_audio_0,input_audio_1,layer):
        input_wav_mean = (input_audio_0 + input_audio_1) / 2.0
        input_wav_mean = self.muencoder(self.rsq48tomuencoder(input_wav_mean), features_only = True)
        layer_results = input_wav_mean['layer_results']
        muencoder_emb = layer_results[layer]
        muencoder_emb = muencoder_emb.permute(0,2,1).contiguous()
        return muencoder_emb




    def forward(self, input_audios, lyric, latents, latent_masks, validation_mode=False, \
        additional_feats = ['spk', 'lyric'], \
        train_rvq=True, train_ssl=False,layer=5):
        if not hasattr(self,"device"):
            self.device = input_audios.device
        if not hasattr(self,"dtype"):
            self.dtype = input_audios.dtype
        device = self.device
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        with torch.no_grad():
            with autocast(enabled=False):
                muencoder_emb = self.extract_muencoder_embeds(input_audio_0,input_audio_1,layer)
            muencoder_emb = muencoder_emb.detach()

        text_encoder_hidden_states, text_mask = None, None


        if(train_rvq):
            quantized_muencoder_emb, _, _, commitment_loss_muencoder_emb, codebook_loss_muencoder_emb,_ = self.rvq_muencoder_emb(muencoder_emb) # b,d,t
        else:
            muencoder_emb = muencoder_emb.float()
            self.rvq_muencoder_emb.eval()
            quantized_muencoder_emb, _, _, commitment_loss_muencoder_emb, codebook_loss_muencoder_emb,_ = self.rvq_muencoder_emb(muencoder_emb) # b,d,t
            commitment_loss_muencoder_emb = commitment_loss_muencoder_emb.detach()
            codebook_loss_muencoder_emb = codebook_loss_muencoder_emb.detach()
            quantized_muencoder_emb = quantized_muencoder_emb.detach()

        commitment_loss = commitment_loss_muencoder_emb
        codebook_loss = codebook_loss_muencoder_emb

        alpha=1
        quantized_muencoder_emb = quantized_muencoder_emb * alpha + muencoder_emb * (1-alpha)

        quantized_muencoder_emb = self.cond_muencoder_emb(quantized_muencoder_emb.permute(0,2,1)) # b t 16*32
        quantized_muencoder_emb = quantized_muencoder_emb.reshape(quantized_muencoder_emb.shape[0], quantized_muencoder_emb.shape[1]//2, 2, 16, 32).reshape(quantized_muencoder_emb.shape[0], quantized_muencoder_emb.shape[1]//2, 2*16, 32).permute(0,2,1,3).contiguous() # b 32 t f

        scenario = np.random.choice(['start_seg', 'other_seg'])
        if(scenario == 'other_seg'):
            for binx in range(input_audios.shape[0]):
                latent_masks[binx,0:random.randint(64,128)] = 1
        
        quantized_muencoder_emb = (latent_masks > 0.5).unsqueeze(1).unsqueeze(-1) * quantized_muencoder_emb \
            + (latent_masks < 0.5).unsqueeze(1).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,32,1,32)


        bsz, _, height, width = latents.shape
        resolution = torch.tensor([height, width]).repeat(bsz, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=muencoder_emb.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=muencoder_emb.dtype, device=device)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        if self.uncondition:
            mask_indices = [k for k in range(quantized_muencoder_emb.shape[0]) if random.random() < 0.1]
            if len(mask_indices) > 0:
                quantized_muencoder_emb[mask_indices] = 0

        latents = self.normfeat.project_sample(latents)
        incontext_latents = latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(1).unsqueeze(-1).float()
        loss = self.cfm_wrapper.compute_loss(latents, [incontext_latents, quantized_muencoder_emb], added_cond_kwargs, latent_masks, validation_mode=validation_mode)
        return loss, commitment_loss.mean(), codebook_loss.mean()

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes(self, input_audios, additional_feats,layer):
        input_audio_0 = input_audios[[0],:]
        input_audio_1 = input_audios[[1],:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.muencoder.eval()


        muencoder_emb = self.extract_muencoder_embeds(input_audio_0,input_audio_1,layer)
        muencoder_emb = muencoder_emb.detach()

        self.rvq_muencoder_emb.eval()
        quantized_muencoder_emb, codes_muencoder_emb, *_ = self.rvq_muencoder_emb(muencoder_emb)


        spk_embeds = None


        return [codes_muencoder_emb], [muencoder_emb], spk_embeds
    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, additional_feats,layer):
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.muencoder.eval()


        muencoder_emb = self.extract_muencoder_embeds(input_audio_0,input_audio_1,layer)
        muencoder_emb = muencoder_emb.detach()

        self.rvq_muencoder_emb.eval()
        quantized_muencoder_emb, codes_muencoder_emb, *_ = self.rvq_muencoder_emb(muencoder_emb) # b,d,t

        spk_embeds = None

        return [codes_muencoder_emb], [muencoder_emb], spk_embeds
    @torch.no_grad()
    def inference_codes(self, codes, spk_embeds, true_latents, latent_length,incontext_length, additional_feats, 
                  guidance_scale=2, num_steps=20,
                  disable_progress=True, scenario='start_seg'):
        classifier_free_guidance = guidance_scale > 1.0
        device = self.device
        dtype = self.dtype
        codes_muencoder_emb = codes[0]


        batch_size = codes_muencoder_emb.shape[0]


        quantized_muencoder_emb,_,_=self.rvq_muencoder_emb.from_codes(codes_muencoder_emb)

        quantized_muencoder_emb = self.cond_muencoder_emb(quantized_muencoder_emb.permute(0,2,1)) # b t 16*32
        quantized_muencoder_emb = quantized_muencoder_emb.reshape(quantized_muencoder_emb.shape[0], quantized_muencoder_emb.shape[1]//2, 2, 16, 32).reshape(quantized_muencoder_emb.shape[0], quantized_muencoder_emb.shape[1]//2, 2*16, 32).permute(0,2,1,3).contiguous() # b 32 t f


        num_frames = quantized_muencoder_emb.shape[-2]

        num_channels_latents = self.num_channels
        latents = self.prepare_latents(batch_size, num_frames, num_channels_latents, dtype, device)

        bsz, _, height, width = latents.shape
        resolution = torch.tensor([height, width]).repeat(bsz, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=quantized_muencoder_emb.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=quantized_muencoder_emb.dtype, device=device)
        if classifier_free_guidance:
            resolution = torch.cat([resolution, resolution], 0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], 0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        latent_masks = torch.zeros(latents.shape[0], latents.shape[2], dtype=torch.int64, device=latents.device)
        latent_masks[:,0:latent_length] = 2
        if(scenario=='other_seg'):
            latent_masks[:,0:incontext_length] = 1

        

        quantized_muencoder_emb = (latent_masks > 0.5).unsqueeze(1).unsqueeze(-1) * quantized_muencoder_emb \
            + (latent_masks < 0.5).unsqueeze(1).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,32,1,32)
        true_latents = self.normfeat.project_sample(true_latents)
        incontext_latents = true_latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(1).unsqueeze(-1).float()
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        additional_model_input = torch.cat([quantized_muencoder_emb],1)

        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=quantized_muencoder_emb.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, incontext_latents, incontext_length, t_span, additional_model_input, added_cond_kwargs, guidance_scale)

        latents[:,:,0:incontext_length,:] = incontext_latents[:,:,0:incontext_length,:]
        latents = self.normfeat.return_sample(latents)
        return latents

    @torch.no_grad()
    def inference(self, input_audios, lyric, true_latents, latent_length, additional_feats, guidance_scale=2, num_steps=20,
                  disable_progress=True,layer=5,scenario='start_seg'):
        codes, embeds, spk_embeds = self.fetch_codes(input_audios, additional_feats,layer)

        latents = self.inference_codes(codes, spk_embeds, true_latents, latent_length, additional_feats, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress,scenario=scenario)
        return latents

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, dtype, device):
        divisor = 4
        shape = (batch_size, num_channels_latents, num_frames, 32)
        if(num_frames%divisor>0):
            num_frames = round(num_frames/float(divisor))*divisor
            shape = (batch_size, num_channels_latents, num_frames, 32)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents


