import logging
import torch
import torch.nn.functional as F
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from typing import Tuple
try:
    import kaldiio
except:
    kaldiio = None
import warnings

logger = logging.getLogger(__name__)


class ArkDataset(RawAudioDataset):
    def __init__(
        self,
        wav_scp, 
        dur_scp,
        sr = 24000,
        max_dur = 20,
        num_buckets=0,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sr,
            max_sample_size=max_dur*sr,
            min_sample_size=1200,
            shuffle=True,
            pad=True,
            normalize=normalize,
            compute_mask=False,
        )
        self.sr = sr 
        self.max_dur = max_dur
        self.normalize = normalize

        logger.info("Loading Kaldi scp files from {}".format(wav_scp))

        self.wav_data = kaldiio.load_scp(wav_scp)
        self.keys = list(self.wav_data.keys())
        dur_data = {}
        keys_set = set(self.keys)
        
        with open(dur_scp, 'r') as f:
            for line in f:
                line = line.strip().split()
                if line[0] in keys_set:
                    dur_data[line[0]] = float(line[-1])
        self.sizes = [int(dur_data[k]*self.sr/100) for k in self.keys]

        logger.info("Loading Kaldi scp files done")

        self.dataset_len = len(self.keys)
        self.set_bucket_info(num_buckets)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        pass

    def size(self, idx):
        pass
    
    def postprocess(self, wav):
        pass
    
    def collater(self, samples):
        pass

