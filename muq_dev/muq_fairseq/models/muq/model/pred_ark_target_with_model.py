import sys
import torch.nn as nn
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rvq_musicfm import PreprocessorWithModel, ResidualVectorQuantize

class RVQ(nn.Module):
    def __init__(self,
                model_config,
                rvq_ckpt_path,
                preprocess,
                ):
        super().__init__()
        self.rvq = ResidualVectorQuantize(**model_config)
        if rvq_ckpt_path is not None:
            self.rvq.load_state_dict(torch.load(rvq_ckpt_path, map_location='cpu'))
        self.preprocess = preprocess
    
    def get_targets(self, x):
        self.rvq.eval()
        x = self.preprocess(x)
        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = self.rvq(x)
        return codes.permute(1,0,2)

    @torch.no_grad()
    def encode_wavs(self, wavs):
        wavs = wavs[..., :int((wavs.shape[-1]//320)*320)]
        return self.get_targets(wavs)

def This_Music_ModelTarget_Config():
    config = dict(
        model = dict(
            input_dim = 1024, 
            n_codebooks = 8, 
            codebook_size = 1024, 
            codebook_dim = 16, 
            quantizer_dropout = 0.0,
        ),
        train = dict(
            batch_size = 32,
            num_workers = 6,
            valid_interval = 10,
            save_interval = 100,
            max_updates = 500000,
            lr = 1e-4,
            # device = 'cuda:1',
            loss = 'commitment_loss * 0.25 + codebook_loss * 1.0 + (x - quantized_prompt_embeds).abs().mean()',
            preprocess = PreprocessorWithModel(
                model_dir= 'path/to/muq_fairseq',
                checkpoint_dir='path/to/muq_m4a_75K.pt',
                use_layer_idx=9,
            )
        ),
        pred = dict(
            rvq_ckpt_path='path/to/runs/Aug07_18-09-24_ts-828fa13e58384d0bba4144fda78ecc92-launcher/ckpt/RVQ_8100.pth',
            sr=24000,
            data_jsonl_path='path/to/data/music4all/train.json', 
            save_target_dir= 'path/to/data/music4all_ark/reiter_musicssl_m4a',
        ),
    )
    return config


CLEN = 30
N_GPU_PER = 8
N_NODE = 4

def parse_lr(wave_length, sr):
    n_step = int( wave_length // (sr*CLEN) )
    if n_step == 0:
        n_step = 1
        print('wave_length: ', wave_length, 'sr: ', sr, 'n_step: ', n_step)
    starts = torch.arange(n_step) * CLEN * sr
    left_rights = torch.stack((starts, starts+CLEN*sr)).T
    return left_rights[:10, ...]

@torch.no_grad()
def main(index, rank):
    device = f'cuda:{rank}'
    config = This_Music_ModelTarget_Config()
    preprocess = config['train']['preprocess']
    model = RVQ(
        model_config = config['model'],
        rvq_ckpt_path = config['pred']['rvq_ckpt_path'],
        preprocess = preprocess
    ).to(device)
    model.eval()
    sr = config['pred']['sr']

    fname_nobase = os.path.basename(config['pred']['data_jsonl_path']).split('.')[0]
    scp_dir = os.path.join(config['pred']['save_target_dir'], 'scp')
    ark_dir = os.path.join(config['pred']['save_target_dir'], 'ark')
    os.makedirs(scp_dir, exist_ok=True)
    os.makedirs(ark_dir, exist_ok=True)

    scp_path = os.path.join(scp_dir, f'{fname_nobase}.{index}_{rank}.scp')
    ark_path = os.path.join(ark_dir, f'{fname_nobase}.{index}_{rank}.ark')

    from kaldiio import WriteHelper

    with open(config['pred']['data_jsonl_path']) as f:
        lines = f.readlines()
    
    print("Total:", len(lines))

    from tqdm import tqdm
    import json
    import librosa
    import time
    from einops import rearrange
    import numpy as np

    # lines = lines[(index*N_GPU_PER+rank)::(N_GPU_PER*N_NODE)]

    with WriteHelper(f'ark,scp:{ark_path},{scp_path}') as writer:
        for idx, line in tqdm(enumerate(lines)):
            try:
                if idx % (N_GPU_PER*N_NODE) != (index*N_GPU_PER+rank):
                    continue
                item = json.loads(line)
                path = item['path']
                wave, _ = librosa.load(path, sr=sr)
                wave = torch.from_numpy(wave)
                wave_length = wave.shape[-1]
                if wave_length < sr*CLEN:
                    continue
                left_rights = parse_lr(wave_length, sr)
                lr = left_rights.tolist()
                wavs = torch.stack(
                    [wave[l:r] for l,r in lr]
                ).to(device)
                targets = model.encode_wavs(wavs) # [Codebook=8, N_Steps, Feature]

                final_target = rearrange(targets, "c n f -> n (c f)").cpu().numpy().astype(np.int32)
                for j in range(final_target.shape[0]):
                    writer(f'{idx}:{j}', final_target[j])
            except Exception as e:
                print(e)


if __name__ == '__main__':
    import sys
    index = int(sys.argv[1])
    import multiprocessing
    pool = multiprocessing.Pool(processes=N_GPU_PER)
    for rank in range(8):
        pool.apply_async(main, (index, rank))
    pool.close()
    pool.join()
    print("Done.")