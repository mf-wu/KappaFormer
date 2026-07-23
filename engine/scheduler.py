from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CustomStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]
        
        

import math
import torch

class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            lr = self.warmup_start_lr + (self.base_lrs[0] - self.warmup_start_lr) * \
                 (self.last_epoch / self.warmup_epochs)
            return [lr for _ in self.optimizer.param_groups]
        else:
            # Cosine 衰减阶段
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = self.base_lrs[0] * 0.5 * (1.0 + math.cos(math.pi * progress))
            return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                   (self.last_epoch / self.warmup_epochs) 
                   for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)) 
                   for base_lr in self.base_lrs]