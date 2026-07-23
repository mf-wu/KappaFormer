from dataclasses import dataclass
import torch 
from typing import Optional,Dict
@dataclass
class ModelOutput:
    loss:torch.Tensor
    total_loss:torch.Tensor
    num_samples: Optional[int] = None
    log_output: Optional[Dict] = None
    pred: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None