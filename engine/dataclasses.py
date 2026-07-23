
from typing import Dict
import torch 
from dataclasses import dataclass,asdict


def format_extra_output(raw_dict):
    if raw_dict == None:
        return ''
    
    extra_output = []
    for k,v in raw_dict.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
                extra_output.append(f'{k}:{v:.4g}')
            else:
                v = v.detach().cpu().numpy()
                extra_output.append(f'{k}:{v:.4g}')
        elif isinstance(v, float):
            extra_output.append(f'{k}:{v:.4g}')
        else:
            extra_output.append(f'{k}:{v}')
    extra_output = " | ".join(extra_output)
    return extra_output



@dataclass
class TrainingLogOutput:
    loss:float
    grad_scale:float
    lr:float
    epoch:int
    batch:int 
    global_step:int 
    total_samples:int
    extra_output: Dict
    
    def __str__(self):
        extra_output = format_extra_output(self.extra_output)
        return f'step: {self.global_step} (Epoch {self.epoch} Iter {self.batch+1}) | Loss: {self.loss:.4g} | LR: {self.lr:.4g} | grad_scale: {self.grad_scale:.4g} | ' + extra_output

@dataclass
class ValidLogOutput:
    valid_loss: float
    epoch: int
    num_samples: int  = None
    extra_output: Dict  = None
    logits:torch.Tensor = None
    label:torch.Tensor = None

    def __str__(self):
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Valid Loss: {self.valid_loss:.4g} | num samples: {self.num_samples} | "
            + extra_output
        )
        
@dataclass
class TrainerState:
    args: None
    global_step =0
    epoch = 0
    batch =0
    sample =0
    