
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stale_tolerance: int = 1000, mfcc_clustering=False, n_layer=1):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.mfcc_clustering = mfcc_clustering

        ProjClass = nn.Identity if mfcc_clustering else WNConv1d
        if n_layer==1:
            self.in_proj = ProjClass(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = ProjClass(codebook_dim, input_dim, kernel_size=1)
        elif n_layer >= 2:
            ndim_hidden = 128
            self.in_proj = nn.Sequential(
                ProjClass(input_dim, ndim_hidden, kernel_size=1),
                *[nn.Sequential(nn.ReLU(), ProjClass(ndim_hidden, ndim_hidden, kernel_size=1),) for _ in range(n_layer-2)],
                nn.ReLU(),
                ProjClass(ndim_hidden, codebook_dim, kernel_size=1)
            )
            self.out_proj = nn.Sequential(
                ProjClass(codebook_dim, ndim_hidden, kernel_size=1),
                nn.ReLU(),
                *[nn.Sequential(ProjClass(ndim_hidden, ndim_hidden, kernel_size=1), nn.ReLU()) for _ in range(n_layer-2)],
                ProjClass(ndim_hidden, input_dim, kernel_size=1),
            )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.register_buffer("stale_counter", torch.zeros(self.codebook_size,))
        self.stale_tolerance = stale_tolerance

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space

        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        if(self.training):
            onehots = torch.nn.functional.one_hot(indices, self.codebook_size).float()  # B, T, codebook_size
            stale_codes = (onehots.sum(0).sum(0) == 0).float()
            self.stale_counter = self.stale_counter * stale_codes + stale_codes

            # random replace codes that haven't been used for a while
            replace_code = (self.stale_counter == self.stale_tolerance).float() # codebook_size
            if replace_code.sum(-1) > 0:
                print("Replace {} codes".format(replace_code.sum(-1)))
                random_input_idx = torch.randperm(encodings.shape[0])
                random_input = encodings[random_input_idx].view(encodings.shape)
                if random_input.shape[0] < self.codebook_size:
                    random_input = torch.cat([random_input]*(self.codebook_size // random_input.shape[0] + 1), 0)
                random_input = random_input[:self.codebook_size,:].contiguous()  # codebook_size, dim

                self.codebook.weight.data = self.codebook.weight.data * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.stale_counter = self.stale_counter * (1 - replace_code)

        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        stale_tolerance: int = 100,
        use_multi_layer_num:int = 1,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i], stale_tolerance=stale_tolerance, n_layer=use_multi_layer_num)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)
        else:
            n_quantizers = torch.ones((z.shape[0],)) * n_quantizers + 1
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            # if self.training is False and i >= n_quantizers:
            #     break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            ) 
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        encodings = F.one_hot(codes, self.codebook_size).float() # B N T 1024

        return z_q, codes, latents, commitment_loss, codebook_loss, n_quantizers.clamp(max=self.n_codebooks).long() - 1

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

from torch.utils.data import Dataset, DataLoader
import json, traceback
import torchaudio
import math

from typing import List, Tuple, Dict, Any

CLIPSECS = 5
def load_audio_by_json(json_path, max_keep, min_keep, tgt_sample_rate):
    # read json file
    print(json_path)
    datas = []
    inds = []
    sizes = []
    with open(json_path) as fp:
        for ind,line in  enumerate(fp):
            data = json.loads(line)
            datas.append(data)
            inds.append(ind)
            # sz = int(data['duration'] * data['sample_rate'])
            sz = int(tgt_sample_rate * CLIPSECS)
            sizes.append(sz)
    tot = ind + 1 
    return datas,inds,tot,sizes

class Read_and_PadCrop_Normalized_T(torch.nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize


    def __call__(self, filename: str, duration: float, cur_sample_rate: int) -> Tuple[torch.Tensor, float, float, int, int]:
        if(duration<(float(self.n_samples)/self.sample_rate+1)):
            # print(duration,(float(self.n_samples)/self.sample_rate+1))
            chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
            t_start = 0.
            t_end = min(1.0, float(self.n_samples) / float(self.sample_rate) / duration)
            offset = 0
            # print('c1:',chunk.shape)
        else:
            offset = np.random.randint(0,int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
            t_start = offset / float(cur_sample_rate) / duration
            t_end = t_start + float(self.n_samples) / float(self.sample_rate) / duration
            chunk, _ = torchaudio.load(filename, frame_offset=offset, num_frames=int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
            # print('offset:',offset)
            # print('c0:',chunk.shape)
        # Pad with silence if necessary.
        if(chunk.shape[0]>1):
            chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
        else:
            chunk = chunk[[0],:].float()
        if(cur_sample_rate!=self.sample_rate):
            # print('a:',cur_sample_rate,chunk.shape)
            chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sample_rate)
            # print('b:',self.sample_rate,chunk.shape)
        if chunk.shape[-1] < self.n_samples:
            chunk = torch.cat([chunk, torch.zeros((1, self.n_samples - chunk.shape[-1],))],-1)
        else:
            chunk = chunk[:,0:self.n_samples]
        seconds_start = math.floor(offset / cur_sample_rate)
        seconds_total = math.floor(duration)

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total
        )

class RVQDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.datas,inds,tot,self.sizes = load_audio_by_json(manifest_path, None, None, self.sample_rate)
        self.dataset_len = len(self.datas)

        self.reader = Read_and_PadCrop_Normalized_T(n_samples=CLIPSECS*sample_rate,sample_rate = self.sample_rate)
        self.normalize = normalize
    

    def __getitem__(self, i):
        # WORLD_SIZE = int(torch.distributed.get_world_size())
        # WORLD_RANK = int(torch.distributed.get_rank())
        # np.random.seed(1337 + self.epoch * WORLD_SIZE + WORLD_RANK + i)
        # index = random.randint(0,len(self.sizes) - 1)
        index = i
        item = None
        while item is None:
            try:
                wav = self.get_audio_by_slice(index)
                # labels = self.get_labels(index)
                # labels = None
                # item = {"id": index, "source": wav, "label_list": labels}
                item = {"id": index, "source": wav}
            except Exception as e:
                # print(e)
                traceback.print_exc()
                print(f'skip damaged data {index}')
                index = np.random.randint(0,len(self.sizes)-1)
        return item

    def __len__(self):
        return self.dataset_len
    
    def get_audio_by_slice(self,index):
        
        wav_path = self.datas[index]['path']
        # print(wav_path)
        audio_info =  torchaudio.info(wav_path)
        origin_sample_rate = audio_info.sample_rate
        origin_duration = audio_info.num_frames / origin_sample_rate

        wav, *ignored = self.reader(wav_path, origin_duration,origin_sample_rate)
        wav = wav.float()
        
        # _path, slice_ptr = parse_path(wav_path)
        # original way
        # if len(slice_ptr) == 0:
        #     wav, cur_sample_rate = sf.read(_path)
        # else:
        #     assert _path.endswith(".zip")
        #     data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #     f = io.BytesIO(data)
        #     wav, cur_sample_rate = sf.read(f)
        # wav = torch.from_numpy(wav).float()
        # print(wav.shape)
        wav = wav.permute(1,0)
        wav = self.postprocess(wav, self.sample_rate)
        # print(wav.shape)

        # wav = wav.squeeze(0)
        return wav
    
    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

