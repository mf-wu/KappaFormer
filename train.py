import os
from engine.trainer import Trainer
import argparse
import torch.optim as optim
import os
from datasets.lmdb_dataset import get_dataset
from engine.logging import  logger
import  torch
from torch_ema import ExponentialMovingAverage
# import wandb

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="kappaformer from Mengfan Wu, Junfu Tan")

parser.add_argument('--strategy', type=str, default='standalone', help="standalone or ddp")
parser.add_argument('--dist_backend', type=str, default='nccl', help="standalone or ddp")
parser.add_argument('--device', type=str, default='cuda', help="XXX")
parser.add_argument('--weight_decay', type=float, default=4e-4, help="XXX")
parser.add_argument('--lr', type=float, default=2e-4, help="XXX")
parser.add_argument('--fp16', type=bool, default=False, help="XXX")
parser.add_argument('--compile', type=bool, default=False, help="XXX")
parser.add_argument('--train_batch_size', type=int, default=16, help="XXX") # bsz 1 1e-4
parser.add_argument('--valid_batch_size', type=int, default=4, help="XXX")
parser.add_argument('--accu_steps', type=int, default=4, help="XXX")
parser.add_argument('--profiling', type=bool, default=False, help="XXX")
parser.add_argument('--begin_save_epoch', type=int, default=1, help="XXX")
parser.add_argument('--save_dir', type=str, default='.', help="XXX")
parser.add_argument('--ifresume', type=bool, default=False, help="XXX")
parser.add_argument('--load_ckpt', type=bool, default=False, help="XXX")
parser.add_argument('--ckpt_path', type=str, default='', help="XXX")
parser.add_argument('--max_epochs', type=int, default=100, help="XXX")
parser.add_argument('--max_steps', type=int, default=100000000, help="XXX")
parser.add_argument('--gradient_clipping', type=float, default=10, help="XXX")
parser.add_argument('--warmup_ratio', type=float, default=0.1, help="XXX")
parser.add_argument('--batch_validate_interval', type=int, default=1000, help="XXX")
parser.add_argument('--val_batch_log_interval', type=int, default=200, help="XXX")
parser.add_argument('--log_interval', type=int, default=100, help="XXX")
parser.add_argument('--save_batch_interval', type=int, default=500000000, help="XXX")
parser.add_argument('--save_epoch_interval', type=int, default=10, help="XXX")
parser.add_argument('--epoch_validate_interval', type=int, default=1, help="XXX")
parser.add_argument('--find_unused_parameters', type=bool, default=True, help="XXX")
parser.add_argument('--enable_kappa', type=bool, default=False, help="XXX")
parser.add_argument('--use_checkpoints', type=str, default='', help="XXX")
parser.add_argument('--warmup_epochs', type=int, default=5, help="XXX")

# Parse parameters 
args = parser.parse_args()
# wandb_api_key = os.getenv('WANDB_API_KEY')
# if wandb_api_key:
#     wandb.init(
#         project=os.getenv('WANDB_PROJECT'),
#         name=os.getenv('WANDB_RUN_NAME'),
#                )
################################################### 
############## DATASET ############################
###################################################
kappa_keys = ['kappa_log', 'density', 'V1', 'n_atoms', 'Ma'] # density-rho, n_atoms-natoms, volume-vol, molecular_weight-mol_weight
bg_keys = ['B_VRH','G_VRH']

if args.enable_kappa:
    label_keys = kappa_keys + bg_keys 
    train_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/kappadata_lmdb/structures_with_exp_kappa_basic',split='train',batch_size=args.train_batch_size  ,label_keys=label_keys)
    valid_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/kappadata_lmdb/structures_with_exp_kappa_basic',split='valid',batch_size=args.valid_batch_size,label_keys=label_keys)
    test_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/kappadata_lmdb/structures_with_exp_kappa_basic',split='test',batch_size=args.valid_batch_size,label_keys=label_keys)
else:
    label_keys = bg_keys
    train_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/bgdata_lmdb/MP_structures_B_G_20250619',split='train',batch_size=args.train_batch_size  ,label_keys=label_keys)
    valid_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/bgdata_lmdb/MP_structures_B_G_20250619',split='valid',batch_size=args.valid_batch_size,label_keys=label_keys)
    test_dataset = get_dataset('/share/home/u15502/mfwu/kappaformer/data/bgdata_lmdb/MP_structures_B_G_20250619',split='test',batch_size=args.valid_batch_size,label_keys=label_keys)

total_num_steps = len(train_dataset) * args.max_epochs / args.train_batch_size
warmup_num_steps = total_num_steps * args.warmup_epochs /args.max_epochs
###################################################
############## MODEL ##############################
###################################################
from models.kappaformer.kappaformer  import Kappaformer

model = Kappaformer(enable_kappa=args.enable_kappa, 
                    label_keys=label_keys)
if args.use_checkpoints != '':
    logger.info(f'Import checkpoints from: {args.use_checkpoints}')
    state_dict= torch.load(args.use_checkpoints) # ['state_dict']
    # new_state_dict = {}
    # for k,v in state_dict.items():
    #     new_k = k.replace('module.module.','')
    #     if new_k in ['sphere_embedding.weight'] or 'source_embedding.weight' in new_k or 'target_embedding.weight' in new_k:
    #         continue
    #     new_state_dict[new_k] = v
    model.load_state_dict(state_dict,strict=False)

    for p in model.parameters():
        p.requires_grad_(True)

    # Freeze main block
    for p in model.block.parameters():
        p.requires_grad_(False)

    for p in model.fusion_harm.parameters():
        p.requires_grad_(False)
    
    # for p in model.inv_block_harm.parameters():
    #     p.requires_grad_(False)

    # for p in model.fusion_anharm.parameters():
    #     p.requires_grad_(False)
    
    # for p in model.inv_block_anharm.parameters():
    #     p.requires_grad_(False)

    # for p in model.moe_block.parameters():
    #     p.requires_grad_(False)
    
    # for p in model.pool_B1.parameters():
    #     p.requires_grad_(False)
    # for p in model.pool_G1.parameters():
    #     p.requires_grad_(False)
    # for p in model.head_B1.parameters():
    #     p.requires_grad_(False)
    # for p in model.head_G1.parameters():
    #     p.requires_grad_(False)

    # for p in model.pool_B2.parameters():
    #     p.requires_grad_(False)
    # for p in model.pool_G2.parameters():
    #     p.requires_grad_(False)
    # for p in model.head_B2.parameters():
    #     p.requires_grad_(False)
    # for p in model.head_G2.parameters():
    #     p.requires_grad_(False)

    # for p in list(model.blocks[-2:].parameters()):
    #     p.requires_grad_(True)

    # for p in model.gate_blocks.parameters():
    #     p.requires_grad_(True)
    # for p in model.gate_blocks_kappa.parameters():
    #     p.requires_grad_(True)

    # for p in model.inv_blocks.parameters():
    #     p.requires_grad_(True)
    # for p in model.inv_blocks_kappa.parameters():
    #     p.requires_grad_(True)

    # for p in model.attn_pool.parameters():
    #     p.requires_grad_(True)
    # for p in model.attn_pool_unharm.parameters():
    #     p.requires_grad_(True)
    
    # for backbone_block in model.blocks:
    #     for p in backbone_block.ffn.parameters():
    #         p.requires_grad_(True)    
        

###################################################
############## Scheduler ##########################
###################################################
from engine.scheduler import WarmupCosineLR
optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
# scheduler = WarmupCosineLR(
#     optimizer,
#     max_epochs=args.max_epochs,
#     warmup_epochs=args.max_epochs*args.warmup_ratio,
#     warmup_start_lr=args.lr*0.01
# )

from torch.optim.lr_scheduler import _LRScheduler

class WarmupLinearDecayScheduler(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr,
                 max_lr,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # 支持为每个 param_group 指定不同的 min/max lr
        if isinstance(min_lr, (float, int)):
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            assert len(min_lr) == len(optimizer.param_groups)
            self.min_lrs = list(min_lr)

        if isinstance(max_lr, (float, int)):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            assert len(max_lr) == len(optimizer.param_groups)
            self.max_lrs = list(max_lr)

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for min_lr, max_lr in zip(self.min_lrs, self.max_lrs):
            if step < self.warmup_steps:
                # 
                lr = min_lr + (max_lr - min_lr) * step / max(1, self.warmup_steps)
            elif step < self.total_steps:
                #
                decay_steps = self.total_steps - self.warmup_steps
                lr = max_lr - (max_lr - min_lr) * (step - self.warmup_steps) / max(1, decay_steps)
            else:
                # 
                lr = min_lr
            lrs.append(lr)
        return lrs

scheduler = WarmupLinearDecayScheduler(
    optimizer,
    warmup_steps=warmup_num_steps,
    total_steps=total_num_steps,
    min_lr=1e-6,
    max_lr=args.lr,
)


# ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

mTrainer = Trainer(args,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss_log_dict=None,
        ema=None)


mTrainer.train()
