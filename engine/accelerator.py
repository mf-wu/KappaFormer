from abc import ABC, abstractmethod
import torch
import logging
import torch.distributed
from torch.utils.data import RandomSampler,DataLoader,DistributedSampler,IterableDataset
from engine.logging import logger
from engine.dataclasses import ValidLogOutput
import os 
import multiprocessing
from contextlib import nullcontext
import datetime
from engine.dynamic_loader import DynamicDistributedSampler
from typing import Optional



def torch_compile(model: torch.nn.Module, state: bool) -> torch.compile:
    if not torch.cuda.is_available():
        logging.warning('cuda is not available, can\'t compile torch model.')
        return model
    else:
        # you can do more things hereS
        return torch.compile(
            model.cuda(),
            fullgraph=False,
            dynamic=True,
            backend='inductor',
            mode='default',
            disable=(not state)
        )

def move_to_device(x, device, nonblocking =False):
    
    if isinstance(device,str):
        device = torch.device(device=device)
    
    if isinstance(x, dict):
        for n in x.keys():
            x[n] = move_to_device(x[n],device=device)
    elif isinstance(x,torch.Tensor) and x.device != device:
        x = x.to(device,non_blocking=nonblocking)
        
    elif isinstance(x,(list,tuple) ):
        x = [move_to_device(xi,device=device) for xi in x ]
        
    return x 

class FP16Scaler(object):
    def __init__(
        self,
        init_scale: int,
        scale_factor: float = 2.0,
        scale_interval: int = 1000,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.since_last_scale_up = 0
        self.scale_interval = scale_interval

    def check_grad_overflow(self, params) -> bool:
        for p in params:
            if p.grad is None:
                continue

            grad_norm = p.grad.data.norm()
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                return True

        return False

    def backward(self, loss):
        if self.enabled:
            scaled_loss = loss * self.scale
        else:
            scaled_loss = loss
        scaled_loss.backward()

    def unscale_and_clip_grad(self, params, clip_grad_norm: float):
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data.float()
                p.grad.data /= self.scale

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(p, clip_grad_norm)

    def step(self, model, optimizer, clip_grad_norm: float = 1.0):
        params = model.parameters()
        if self.enabled:
            if self.check_grad_overflow(params):
                self.scale /= self.scale_factor
                logger.info(
                    f"Gradient overflow detected, reducing scale to {self.scale}"
                )
                self.since_last_scale_up = 0
                # Skip optimizer step
            else:
                self.unscale_and_clip_grad(params, clip_grad_norm)
                optimizer.step()

                self.since_last_scale_up += 1
                if (
                    self.since_last_scale_up >= self.scale_interval
                    and self.scale < 2**15
                ):
                    self.scale *= self.scale_factor
                    self.since_last_scale_up = 0
        else:
            self.unscale_and_clip_grad(params, clip_grad_norm)
            optimizer.step()

def safe_div(x,y):
    if y ==0:
        return 0
    else:
        return x/y
    
    
class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass
    
    @abstractmethod
    def train_step(self,grouped_data_batch):
        pass
    
    @abstractmethod
    def valid_step(self,data_batch, epoch):
        pass
    
    @abstractmethod
    def save_ckpt(self,ckpt_id):
        pass
    
    @abstractmethod
    def load_ckpt(self,ckpt_paths):
        pass
    
    @abstractmethod
    def build_dataloader(self,train_data,valid_data,test_data):
        pass
    
    @abstractmethod
    def barrier(self,):
        pass
    
    @abstractmethod
    def sync_valid_loss(self,total_loss,num_samples):
        pass
    
    @property
    @abstractmethod
    def grad_scale(self,):
        pass
    
    
    def before_epoch(self,epoch):
        pass
    
    def skip_first_batch(self,start_iter):
        return False
    
    def _accumulate_log_output(self, total_log_output, log_output, current_sample_count, num_new_samples):
        for k in log_output:
            v  = log_output[k]
            if k not in total_log_output:
                if isinstance(v,torch.Tensor):
                    total_log_output[k] = v.item()
                # elif isinstance(v,tuple):
                #     total_log_output[k] = [vv.item() for vv in v]
                else:
                    total_log_output[k] = v 
            else:
                if isinstance(v, torch.Tensor):
                    total_log_output[k] = (total_log_output[k] * current_sample_count + v.item() * num_new_samples) / (current_sample_count + num_new_samples)
                else:
                    total_log_output[k] = (total_log_output[k] * current_sample_count + v * num_new_samples) / (current_sample_count + num_new_samples)
    

class StandaloneAccelerator(Accelerator):
    def __init__(self,args, model, optimizer, lr_scheduler, device, ema=None):
        super().__init__()       
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.ema = ema
        self.world_size = 1
        
        if not torch.cuda.is_available():
            logging.warning('cuda is not available. use cpu instead.')
            self.device = 'cpu'
        self.scaler = FP16Scaler(init_scale=1, enabled=self.args.fp16)
        
        if args.fp16:
            self.model = self.model.half()
            
        self.model.to(self.device)
        
        if self.ema is not None:
            self.ema.to(self.device)
        
        self.model = torch_compile(self.model, self.args.compile)
        
        
    
    @property
    def grad_scale(self,):
        return self.scaler.scale
    
    def build_dataloader(self, train_data, valid_data, test_data=None):
        train_batch_size_per_gpu = self.args.train_batch_size // (self.world_size * self.args.accu_steps)
        
        assert (train_batch_size_per_gpu > 0 ), 'train_batch_size_per_gpu must greater than 0'
        
        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            batch_size=train_batch_size_per_gpu,
            sampler=self.train_sampler,
            collate_fn=train_data.collate,
            drop_last=True
        )
        
        if valid_data:
            valid_batch_size_per_gpu = self.args.valid_batch_size // (self.world_size)
        
            assert (valid_batch_size_per_gpu>0), 'valid_batch_size_per_gpu must greater than 0'
            
            self.valid_data_loader = DataLoader(
                valid_data,
                batch_size=valid_batch_size_per_gpu,
                sampler=None,
                collate_fn=valid_data.collate,
                drop_last=False
            )
        else:
            self.valid_data_loader = None
        
        if test_data:
            test_batch_size_per_gpu = self.args.valid_batch_size 
            self.test_data_loader = DataLoader(
                test_data,
                batch_size=valid_batch_size_per_gpu,
                sampler=None,
                collate_fn=test_data.collate,
                drop_last=False
            )
        else:
            self.test_data_loader = None   
            
            
        return 
    
    
    def train_step(self, grouped_data_batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        
        for batch_data in grouped_data_batch:
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)
            
            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred,batch_data)
            loss = model_output.loss / len(grouped_data_batch)
            
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logging.warning('loss is nan or inf.')
                loss = loss.new_tensor(0.0,requires_grad=True)
            else:
                success_batch_count+=1
                
            self.scaler.backward(loss)
            
            if model_output.num_samples is not None:
                self._accumulate_log_output(total_log_output,model_output.log_output,sample_count,model_output.num_samples)
                sample_count += model_output.num_samples
                total_loss += model_output.total_loss
            
            self.model.after_batch()
        
        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)
        
        self.lr_scheduler.step()
        
        # return for up call
        model_output.num_samples = sample_count
        model_output.loss = total_loss / sample_count if sample_count != 0 else 0
        model_output.log_output = total_log_output
                
        
        return model_output
        
        
    def valid_step(self, batch_data, epoch):
        self.model.eval()
        
        batch_data = move_to_device(batch_data, self.device)
        if self.ema is not None:
            with self.ema.average_parameters():
                pred = self.model(batch_data)
        else:
            pred = self.model(batch_data)
            
        model_output = self.model.compute_loss(pred,batch_data)
        

        return ValidLogOutput(
            valid_loss = model_output.loss.item(),
           epoch = epoch,
           num_samples = model_output.num_samples,
           extra_output = model_output.log_output
        )
        
    def sync_valid_loss(self, total_loss, num_samples):
        return total_loss, num_samples
    
    def set_up(self):
        pass
    
    def save_ckpt(self, ckpt_id,state):
        return super().save_ckpt(ckpt_id)
    
    def load_ckpt(self, ckpt_paths):
        return super().load_ckpt(ckpt_paths)

    def barrier(self):
        return super().barrier()
    
    @staticmethod
    def _allreducelog(log,log_num_dict):
        return None
    
    
class DDPAccelerator(StandaloneAccelerator):
    def __init__(self, args, model, optimizer, lr_scheduler, device, ema=None):
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda", ema=ema)
        
    def set_up(self):
        super().set_up()
        
        assert "WORLD_SIZE" in os.environ, "no WORLD_SIZE in ENV"
        assert "RANK" in os.environ, "no RANK in ENV"
        assert "LOCAL_RANK" in os.environ, "no LOCAL_RANK in ENV"
        
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        
        self.master_addr = os.environ.get('MASTER_ADDR','')
        self.master_port = os.environ.get('MASTER_PORT','')
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda',self.local_rank)
        
        multiprocessing.set_start_method('spawn',force=True)
        
        ddp_timeout = os.environ.get('DDP_TIMEOUT_MINUTES',None)
        logger.critical(f'init ddp, world_size:{self.world_size}| rank: {self.rank}|\n local_rank: {self.local_rank}| master_addr: {self.master_addr}| master_port: {self.master_port}')
        
        torch.distributed.init_process_group(backend=self.args.dist_backend,init_method='env://',world_size=self.world_size,rank=self.rank,timeout=datetime.timedelta(days=2))
        
        torch.distributed.barrier()
        
        logger.success('DDP initialized!')        
        self.model.to(self.device)
        
        self.ddp_model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=self.args.find_unused_parameters)
        
        if self.model.ckpt_loaded:
            logger.info('reload ckpt after DDP')
            self.ddp_model.module.reload_checkpoint()
        
        self.ddp_model = torch_compile(self.ddp_model, self.args.compile)
        return 
    
    def barrier(self):
        torch.distributed.barrier()
        
    def train_step(self, grouped_data_batch):
        self.ddp_model.train()
        self.optimizer.zero_grad()
        
        success_batch_count = 0
        sample_count = 0
        total_loss =0.0
        total_log_output = {}
        
        for idx, batch_data in enumerate(grouped_data_batch):
            self.model.before_batch()
            batch_data = move_to_device(batch_data,self.device)
            maybe_no_sync = self.ddp_model.no_sync() if idx != len(grouped_data_batch) -1 else nullcontext()
            
            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_data_batch)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info('loss is nan or inf, skip')
                    success_batch_count += 1
                    mask = torch.isnan(loss) | torch.isinf(loss) 
                    loss[mask] = 0.0
                    self.scaler.backward(loss)
                else:
                    success_batch_count += 1
                    self.scaler.backward(loss)

            self._accumulate_log_output(
                total_log_output,
                model_output.log_output,
                sample_count,
                model_output.num_samples
            )
            sample_count += model_output.num_samples
            total_loss += model_output.loss * model_output.num_samples
            self.model.after_batch()
        

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)
            if self.ema is not None:
                self.ema.update()
        
        self.lr_scheduler.step()  
        # return for up call
        model_output.num_samples = sample_count
        model_output.loss = total_loss / sample_count if sample_count != 0 else 0
        model_output.log_output = total_log_output

        # if self.ema is not None:
        #     self.ema.update()   
        
        return model_output
    
    def build_data_loader(
        self,
        train_data,
        val_data,
        test_data = None,
    ):
        
       
        train_batch_size_per_gpu = self.args.train_batch_size // (
            self.world_size * self.args.accu_steps
        )
        assert (train_batch_size_per_gpu > 0), "train_batch_size_per_gpu should be greater than 0"

        if not isinstance(train_data, IterableDataset):
                self.train_sampler = DistributedSampler(
                    train_data, num_replicas=self.world_size, rank=self.rank
                )
                self.train_data_loader = DataLoader(
                    train_data,
                    sampler=self.train_sampler,
                    batch_size=train_batch_size_per_gpu,
                    collate_fn=train_data.collate,
                    drop_last=True,
                )
        else:
            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=train_batch_size_per_gpu,
                collate_fn=train_data.collate,
                drop_last=True,
                num_workers=1,
            )

        if val_data:
            if self.args.use_unified_batch_sampler:
                
                valid_batch_size_per_gpu = self.args.val_batch_size // self.world_size
                assert (
                    valid_batch_size_per_gpu > 0
                ), "valid_batch_size_per_gpu should be greater than 0"
                validsampler = torch.utils.data.distributed.DistributedSampler(
                    val_data, num_replicas=self.world_size, shuffle=False
                )
                self.valid_data_loader = DataLoader(
                    val_data,
                    sampler=validsampler,
                    batch_size=valid_batch_size_per_gpu,
                    collate_fn=val_data.collate,
                    drop_last=False,
                )
        else:
            self.valid_data_loader = None

        if test_data:
            valid_batch_size_per_gpu = self.args.val_batch_size
            testsampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=self.world_size, shuffle=False
            )
            self.test_data_loader = DataLoader(
                test_data,
                sampler=testsampler,
                batch_size=valid_batch_size_per_gpu,
                collate_fn=val_data.collate,
                drop_last=False,
            )

    def before_epoch(self, epoch: int):
        if self.args.strategy == 'ddp':
            return
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def save_ckpt(self, ckpt_id, extra_state):
        if self.rank == 0:
            if self.ema is not None:
                with self.ema.average_parameters():
                    super().save_ckpt(ckpt_id, extra_state)
            else:
                super().save_ckpt(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_samples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_samples = torch.Tensor([num_samples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_samples)
        total_loss = total_loss.item()
        num_samples = num_samples.item()

        return total_loss, num_samples

    def sync_valid_metric(self, label_list, logits_list):
        if not label_list or not logits_list:
            return None, None

        label = torch.cat(label_list, dim=0).to(self.device)
        logits = torch.cat(logits_list, dim=0).to(self.device)
        num_samples = torch.zeros(
            self.world_size + 1, device=self.device, dtype=torch.long
        )
        num_samples[self.rank + 1] = label.shape[0]
        torch.distributed.all_reduce(num_samples)
        total_samples = int(torch.sum(num_samples).item())
        for i in range(1, self.world_size + 1):
            num_samples[i] += num_samples[i - 1]
        total_label = torch.zeros(
            total_samples, *label.shape[1:], device=self.device, dtype=label.dtype
        )
        total_logits = torch.zeros(
            total_samples, *logits.shape[1:], device=self.device, dtype=logits.dtype
        )

        total_label[num_samples[self.rank] : num_samples[self.rank + 1]] = label
        total_logits[num_samples[self.rank] : num_samples[self.rank + 1]] = logits
        torch.distributed.all_reduce(total_label)
        torch.distributed.all_reduce(total_logits)
        return total_label, total_logits

    def calculate_metric(self, label, logits):
        return self.model.calculate_metric(label, logits)

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: safe_div(v, log_num_dict[k]) for k, v in log_dict.items()}

    def skip_first_batches(self, start_iteration):
            return False