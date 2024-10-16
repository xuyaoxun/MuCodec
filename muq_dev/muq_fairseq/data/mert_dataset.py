# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)

import math
import io
import torchaudio
# this is in the user_dir
from nnAudio import features as nnAudioFeatures

# from tqdm import tqdm
import tqdm
import json
import random
import traceback
from einops import rearrange
# from scripts.prepare_codecs_from_manifest import *

logger = logging.getLogger(__name__)

class model_cqt_pred(torch.nn.Module):
    def __init__(self, n_bins=84, sr=16000, freq=50):
        super().__init__()
        self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = nnAudioFeatures.cqt.CQT(sr=sr, hop_length=sr//freq, fmin=32.7, 
                                           fmax=None, n_bins=n_bins, bins_per_octave=n_bins//7, 
                                           filter_scale=1, norm=1, window='hann', center=True, 
                                           pad_mode='constant', trainable=False, 
                                           output_format='Magnitude', verbose=True)

        # self.fc = nn.Linear(input_dim, n_bins)

        # self.criterion = nn.MSELoss()
        self.forward_dict = {
            # 'masked_transformer_output': self.plain_forward
            'compute_cqt': self.compute_cqt
        }
    def compute_cqt(self, x):
        '''
        convert waveform to CQT -> [batch, bins, len] -> transpose
        '''
        # align with the padding of HuBERT model, 
        # the truncation is calculated by bruteforce search since the nnAudio padding strategy and fairseq models are different
        # x = x[..., :-560] 
        return torch.transpose(self.spec_layer(x), -1, -2)

    def forward(self, x, forward_type='masked_transformer_output'):
        '''
        take input from transformer hidden states: [batch, len_seq, channel]
        output: [batch, len_seq, n_bins]
        '''
    
        return self.forward_dict[forward_type](x)

def load_audio_by_json(json_path, max_keep, min_keep, tgt_sample_rate, clip_secs=5):
    # read json file
    print(json_path)
    datas = []
    inds = []
    sizes = []
    with open(json_path) as fp:
        for ind,line in  enumerate(fp):
            data = json.loads(line)
            if 'duration' in data and min_keep is not None and tgt_sample_rate*data['duration'] < min_keep:
                continue
            datas.append(data)
            inds.append(ind)
            # sz = int(data['duration'] * data['sample_rate'])
            if clip_secs > 0:
                sz = int(tgt_sample_rate * clip_secs)
            else:
                sz = int(tgt_sample_rate * data['duration'])
            sizes.append(sz)
    tot = ind + 1 
    return datas,inds,tot,sizes
def load_audio(manifest_path, max_keep, min_keep):
    pass


def load_label(label_path, inds, tot):
    pass

def load_numpy_label(label_path, inds, tot):
    labels = np.load(label_path, mmap_mode='r')
    assert (labels.shape[0] == tot), f"number of labels does not match ({labels.shape[0]} != {tot})"
    return labels

def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    pass

class Read_and_PadCrop_Normalized_T(torch.nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize


    def __call__(self, filename: str, duration: float, cur_sample_rate: int, fixed_offset_duration=None) -> Tuple[torch.Tensor, float, float, int, int]:
        pass


class MERTDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_scp_path: Optional[str] = None,
        label_scp_clip_duration: float = -1,
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        npmemmap: bool = False,
        random_crop: bool = False,
        single_target: bool = False,
        augmentation_effects: List[str] = [],
        augmentation_probs: List[float] = [],
        inbatch_noise_augment_len_range: List[int] = [8000, 24000],
        inbatch_noise_augment_number_range: List[int] = [1, 3],
        inbatch_noise_augment_volume: float = 1.0,
        cqt_prediction_bin: int = -1,
        dataset_len:int = 128*3000,
        clip_secs = 5,
    ):
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.datas,inds,tot,self.sizes = load_audio_by_json(manifest_path,max_keep_sample_size,min_keep_sample_size, self.sample_rate, clip_secs)
        self.inds = inds

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        self.npmemmap = npmemmap
        self.label_scp_path = label_scp_path
        self.label_scp_clip_duration = label_scp_clip_duration


        if self.label_scp_path is not None:
            from kaldiio import load_scp
            self.label_scp = load_scp(self.label_scp_path)

        # self.dataset_len = dataset_len
        self.dataset_len = len(self.datas)
        logger.info('preparing labels')
        logger.info('========dataset len: {}=========='.format(self.dataset_len))
        if store_labels:
            if self.npmemmap:
                self.label_list = [load_numpy_label(p+'.npy', inds, tot) for p in label_paths] 
            else:
                self.label_list = [load_label(p, inds, tot) for p in label_paths]        
        else:
            self.label_paths = label_paths
            # self.label_offsets_list = [
            #     load_label_offset(p, inds, tot) for p in label_paths
            # ]
        assert label_processors is None or len(label_processors) == self.num_labels


        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        self.augmentation_effects = augmentation_effects
        self.augmentation_probs = augmentation_probs


        self.inbatch_noise_augment_len_range = inbatch_noise_augment_len_range
        self.inbatch_noise_augment_number_range = inbatch_noise_augment_number_range
        self.inbatch_noise_augment_volume = inbatch_noise_augment_volume
        

        self.cqt_prediction_bin = cqt_prediction_bin
        if self.cqt_prediction_bin > 0:
            self.encoder_cqt_model = model_cqt_pred(n_bins=self.cqt_prediction_bin)
            logger.info('preparing cqt loss objective in dataloader with cpu')

        self.epoch = -1

        self.reader = Read_and_PadCrop_Normalized_T(n_samples=clip_secs*sample_rate if clip_secs>0 else None, sample_rate = self.sample_rate)


            
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        pass
    def set_epoch(self, epoch):
        pass
        
    def inbatch_noise_augment(self, 
        target_audio: torch.Tensor, target_audio_idx: int , 
        batch_audios: torch.Tensor, # [bsz, audio_lengths]
        noise_len_min: int, noise_len_max: int, 
        n_noise_min: int, n_noise_max: int,
        noise_vol: float = 1.0):
        pass
    
    def get_audio_by_slice(self,index):
        pass
    def get_audio(self, index):
        pass

    def get_label(self, index, label_idx):
        pass

    def get_labels(self, index):
        pass

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.dataset_len

    def crop_to_max_size(self, wav, target_size):
        pass

    def collater(self, samples):
        pass

    def collater_audio(self, audios, audio_size):
        pass

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        pass

    def collater_seq_label(self, targets, pad):
        pass

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        pass

    def num_tokens(self, index):
        pass

    def size(self, index):
        pass

    def ordered_indices(self):
        pass

    def postprocess(self, wav, cur_sample_rate):
        pass
