import torch
from dataclasses import dataclass
import fairseq
import os.path as op

root = op.dirname(op.abspath(__file__))


@dataclass
class UserDirModule:
    user_dir: str

def load_model(model_dir, checkpoint_dir):
    '''Load Fairseq SSL model'''

    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir], strict=False)
    model = model[0]

    return model
