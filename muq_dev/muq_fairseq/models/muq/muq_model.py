try:
    from .model.muq import MuQ
except:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.muq import MuQ
try:
    from fairseq.fairseq.dataclass import FairseqDataclass
    from fairseq.fairseq.models import BaseFairseqModel, register_model
    from fairseq.fairseq.tasks.fairseq_task import FairseqTask
except:
    from fairseq.dataclass import FairseqDataclass
    from fairseq.models import BaseFairseqModel, register_model
    from fairseq.tasks.fairseq_task import FairseqTask
    
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

from logging import getLogger

logger = getLogger(__name__)

@dataclass
class MuQConfig(FairseqDataclass):
    label_rate:int = field(default=25)
    num_codebooks:int = field(default=1)
    codebook_dim:int = field(default=16)
    codebook_size:int = field(default=4096)
    features:List[str] = field(default_factory=lambda:["melspec_2048"])
    hop_length:int = field(default=240)
    n_mels:int = field(default=128)
    conv_dim:int = field(default=512)
    encoder_dim:int = field(default=1024)
    encoder_depth:int = field(default=12)
    mask_hop:float = field(default=0.4)
    mask_prob:float = field(default=0.6)
    is_flash:bool = field(default=False)
    stat_path:Optional[str] = field(default=None)
    model_path:Optional[str] = field(default=None)
    w2v2_config_path:Optional[str] = field(default=None)
    use_rvq_target:bool = field(default=False)
    use_vq_target:bool = field(default=False)
    rvq_ckpt_path: Optional[str] = field(default=None)
    recon_loss_ratio: Optional[float] = field(default=None)
    resume_checkpoint: Optional[str] = None
    use_hubert_masking_strategy:bool = field(default=False)
    use_hubert_featurizer:bool = field(default=False)
    hubert_conv_feature_layers:str = field(default_factory=lambda:"[(512,10,5)] + [(512,3,2)] * 3  + [(512,3,3)] + [(512,2,2)] * 2")
    rvq_n_codebooks:int = field(default=8)
    rvq_multi_layer_num:int = field(default=1)
    use_encodec_target:bool = field(default=False)

SAMPLE_RATE = 24_000

@register_model("muq", dataclass=MuQConfig)
class MuQModel(BaseFairseqModel):
    def __init__(self, cfg: MuQConfig, task_cfg: FairseqTask):
        super().__init__()
        self.cfg = cfg
        self.model = MuQ(
            num_codebooks=cfg.num_codebooks,
            codebook_dim=cfg.codebook_dim,
            codebook_size=cfg.codebook_size,
            features=cfg.features,
            n_mels=cfg.n_mels,
            conv_dim=cfg.conv_dim,
            encoder_dim=cfg.encoder_dim,
            encoder_depth=cfg.encoder_depth,
            mask_hop=cfg.mask_hop,
            mask_prob=cfg.mask_prob,
            is_flash=cfg.is_flash,
            stat_path=cfg.stat_path,
            model_path=cfg.model_path,
            w2v2_config_path=cfg.w2v2_config_path,
            use_rvq_target=cfg.use_rvq_target,
            use_vq_target=cfg.use_vq_target,
            rvq_ckpt_path=cfg.rvq_ckpt_path,
            recon_loss_ratio=cfg.recon_loss_ratio,
            label_rate=cfg.label_rate,
            use_hubert_masking_strategy=cfg.use_hubert_masking_strategy,
            use_hubert_featurizer=cfg.use_hubert_featurizer,
            hubert_conv_feature_layers=cfg.hubert_conv_feature_layers,
            rvq_n_codebooks=cfg.rvq_n_codebooks,
            rvq_multi_layer_num=cfg.rvq_multi_layer_num,
            use_encodec_target=cfg.use_encodec_target,
        )

    def forward(
        self,
        source: torch.Tensor, # B,L
        features_only: bool = False,
        label = None, # pre-extracted labeks, dim is [Batch, N_Codebook, SeqLen]
        **kwargs,
    ):
        source = source[..., :int((source.shape[-1]//(SAMPLE_RATE//self.cfg.label_rate))*(SAMPLE_RATE//self.cfg.label_rate)) ]
        if features_only:
            if 'attention_mask' in kwargs:
                attention_mask = kwargs['attention_mask']
            elif 'padding_mask' in kwargs:
                attention_mask = ~kwargs['padding_mask'].bool()
            else: 
                attention_mask = None
            _, hidden_states = self.model.get_predictions(source, attention_mask=attention_mask, is_features_only=True)
            result = {
                "layer_results": hidden_states
            }
            return result
        else:
            result = {}
            logits, hidden_emb, losses, accuracies = self.model(source, label=label)
            result["losses"] = losses
            result["accuracies"] = accuracies
            result["logits"] = logits
            result["hidden_emb"] = hidden_emb
            for k, v in losses.items():
                result[k] = v
            return result

    @classmethod
    def build_model(cls, cfg: MuQConfig, task: FairseqTask):
        """Build a new model instance."""

        model = MuQModel(cfg, task.cfg)
        import numpy as np
        s = 0
        for param in model.parameters():
            s += np.product(param.size())
        # print('# of parameters: '+str(s/1024.0/1024.0))
        
        if cfg.get("resume_checkpoint", None):
            print("Loading checkpoint from {}".format(cfg.resume_checkpoint))
            model.load_state_dict(torch.load(cfg.resume_checkpoint)['model'], strict=False)

        return model

    def get_losses(self, result, batch):
        return result['losses']
    