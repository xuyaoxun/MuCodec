# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

from ..data.mert_dataset import MERTDataset
from ..data.ark_dataset import ArkDataset

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        # encode_line return a torch.IntTensor, should be all 1 for vanila HuBERT
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )
class PaddedNumpyLabelEncoder(object):
    def __init__(self):
        # self.dictionary = dictionary
        pass

    def __call__(self, label):
        t = torch.IntTensor(np.asarray(label))
        t = t[t>=0] # remove padded -1 values at the end
        return t

@dataclass
class MuQPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sharding_data: int = field(
        default=-1,
        metadata={
            "help": "set this para >1 to use sharding dataset to prevent OOM"
            "prepare data tsv and label files by adding postfix for sharding 64 like:"
            "train_28_64.tsv and train_28_64.encodec_6"
        },
    )
    load_random_data_shard: bool = field(
        default=True,
        metadata={
            "help": "whether to laod shards randomly or in order when use sharding_data"
        },
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_scp_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'if set, load label from scp file'
        }
    )
    label_scp_clip_duration: float = field(
        default=-1,
        metadata={
            'help': 'clip duration for loading scp label. if set to -1, this will not make effect.'
        }
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    store_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load all of the label into memory"}, 
    )

    numpy_memmap_label: Optional[bool] = field(
        default=False,
        metadata={"help": "whether the label file is saved as a numpy file, each line is ended with padding -1"}, 
    )

    augmentation_effects: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "a list of effects that might apply to the audios"
                "example: \"['random_mute', 'random_Gaussian', 'reverse_polarity']\" "
                "supported: random_mute,"
                "todo: "
            )
        },
    )
    augmentation_probs: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "the corresponding probabilities for the data augmentation effects"
                "example: \"[0.1, 0.5, 0.8]\" "
                "the sum is not necessarily need to be 1.0, and multiple effects can be applied to the same audio"
            )
        },
    )

    # inbatch_noise_augment_len_range: Optional[List[int]] = field(
        # default_factory=lambda: [8000, 24000],
        # default = [8000, 24000],
    inbatch_noise_augment_len_range: Optional[str] = field(
        default = "[8000, 24000]",
        metadata={
            "help": (
                "the range of length of the mix-up noise augmentation, unit in smaples"
            )
        },
    )
    # inbatch_noise_augment_number_range: Optional[List[int]] = field(
    #     default_factory=lambda: [1, 3],
        # default = [1, 3],
    inbatch_noise_augment_number_range: Optional[str] = field(
        default = "[1, 3]",
        metadata={
            "help": (
                "the range of numbers of the mix-up noise augmentation"
            )
        },
    )
    inbatch_noise_augment_volume: float = field(
        default = 1.0,
        metadata={
            "help": (
                "the coefficient used to modify the volume of the noise audios wavs"
            )
        },
    )
    dynamic_crops: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "used to set the maximum audio length setting, for training"
                "example: \"[1, 2, 3, 4, 5, 10]\" "
            )
        },
    )
    dynamic_crops_epoches: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "used to set training epoches of changing the maximum audio length"
                "example: \"[1, 10, 20, 40, 80, 160,]\" "
                "then len need to be equal to len(dynamic_crops)"
            )
        },
    )

    cqt_loss_bin_dataloader: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "use this parameter to prepare cqt prediction objective in dataloader"
            )
        },
    )

    clip_secs: int = field(
        default=5,
        metadata={
            "help": "clip secs for each audio"
        }
    )

    dataset_shuffle: bool = field(
        default=True,
        metadata={
            "help": (
                "dataset shuffle when sample a batch"
            )
        },
    )


@register_task("muq_pretraining", dataclass=MuQPretrainingConfig)
class MuQPretrainingTask(FairseqTask):

    cfg: MuQPretrainingConfig

    def __init__(
        self,
        cfg: MuQPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MuQPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"
        
        # use eval() to pass list parameters, skirt the fairseq/torch error:  Can't pickle <enum 'Choices'>: attribute lookup Choices on fairseq.dataclass.constants failed
        self.augmentation_effects = eval(self.cfg.augmentation_effects)
        self.augmentation_probs = eval(self.cfg.augmentation_probs)
        if len(self.augmentation_effects) > 0:
            assert len(self.augmentation_effects) == len(self.augmentation_probs)
            logger.info(f"Applying audio augmentation {self.augmentation_effects}, probabilities: {self.augmentation_probs}")
        
        self.inbatch_noise_augment_number_range = eval(self.cfg.inbatch_noise_augment_number_range)
        self.inbatch_noise_augment_len_range = eval(self.cfg.inbatch_noise_augment_len_range)

        self.max_sample_size = self.cfg.max_sample_size

        self.dynamic_crops = eval(self.cfg.dynamic_crops)
        self.dynamic_crops_epoches = eval(self.cfg.dynamic_crops_epoches)
        assert len(self.dynamic_crops) == len(self.dynamic_crops_epoches)
        if len(self.dynamic_crops) > 0:
            assert self.dynamic_crops_epoches[0] == 1
            
        self.cqt_loss_bin_dataloader = self.cfg.cqt_loss_bin_dataloader

        self.numpy_memmap_label = self.cfg.numpy_memmap_label
        self.store_labels = self.cfg.store_labels
        if self.numpy_memmap_label:
            assert self.store_labels

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: MuQPretrainingConfig, **kwargs
    ) -> "MuQPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if (self.cfg.label_dir is None or self.cfg.label_dir == '') else self.cfg.label_dir
        print(label_dir)
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None or self.cfg.label_dir=='':
            return self.cfg.data
        return self.cfg.label_dir


    def is_force_load_dataset(self, epoch, training_restore=False):
        # find the threshold that holds epoch \in [threshold, next_threshold)
        return (epoch in self.dynamic_crops_epoches) or training_restore or (self.cfg.sharding_data > 1)


    def set_dynamic_crop_max_sample(self, epoch):
        pass

    def load_dataset(self, split: str, **kwargs) -> None:
        pass

    def load_dataset_ark(self, split, **kwargs):
        pass

    def load_dataset_mert(self, split: str, **kwargs) -> None:
        pass

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
