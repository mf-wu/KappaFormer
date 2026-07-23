from engine.accelerator import StandaloneAccelerator,DDPAccelerator
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Dict
import torch 
from engine.logging import logger, metric_logger
import time
import numpy as np
import os
from engine.dataclasses import TrainerState,TrainingLogOutput,ValidLogOutput
from typing import Any, Dict, List, Optional, Tuple

class LossAccumulator(object):
    def __init__(self):
        self.sum = 0
        self.num_examples = 0

    def add(self, loss, num_examples):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            try:
                loss = loss.item()
            except:
                logger.error(f"Loss is not a valid tensor: {loss}")
                loss = 0.0

        if type(num_examples) == torch.Tensor:
            num_examples = num_examples.item()

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples


class MetricAccumulator(object):
    def __init__(self, world_size=1):
        self.sum = 0
        self.num_examples = 0
        self.start_time = time.time()
        self.world_size = world_size
        self.label_list = []
        self.logits_list = []

    def add(self, logits, label):
        if logits is None or label is None:
            return

        self.label_list.append(label)
        self.logits_list.append(logits)

    def reset(self):
        self.label_list = []
        self.logits_list = []


class LogAccumulator(object):
    def __init__(self, world_size=1, allreduce_fn=None):
        self.sum = 0
        self.num_examples = 0
        self.extra_log = {}
        self.extra_log_num = {}
        self.start_time = time.time()
        self.allreduce_fn = allreduce_fn
        self.world_size = world_size
        self.extra_log["total_acc_sample"] = 0

    def add(self, loss, num_examples, extra_log=None):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            loss = float(loss.clone().item())

        if type(num_examples) == torch.Tensor:
            num_examples = int(num_examples.clone().item())

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

        if extra_log is not None:
            for k, v in extra_log.items():
                if k not in self.extra_log and isinstance(
                    v, (torch.Tensor, float, tuple)
                ):
                    if k == "total_acc_sample":
                        continue
                    elif isinstance(v, torch.Tensor):
                        self.extra_log[k] = float(v.item()) * num_examples
                        self.extra_log_num[k] = 1 * num_examples
                    elif isinstance(v, tuple):
                        self.extra_log[k] = float(v[0]) * float(v[1])
                        self.extra_log_num[k] = float(v[1])
                    else:
                        self.extra_log[k] = float(v) * num_examples
                        self.extra_log_num[k] = 1 * num_examples
                elif k in self.extra_log and isinstance(
                    v, (torch.Tensor, float, tuple)
                ):
                    if k == "total_acc_sample":
                        continue
                    elif isinstance(v, torch.Tensor):
                        self.extra_log[k] += float(v.item()) * num_examples
                        self.extra_log_num[k] += 1 * num_examples
                    elif isinstance(v, tuple):
                        self.extra_log[k] += float(v[0]) * float(v[1])
                        self.extra_log_num[k] += float(v[1])
                    else:
                        self.extra_log[k] += float(v) * num_examples
                        self.extra_log_num[k] += 1 * num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0
        self.start_time = time.time()
        for k, v in self.extra_log.items():
            if k == "total_acc_sample":
                continue
            self.extra_log[k] = 0.0
            self.extra_log_num[k] = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        if self.allreduce_fn is not None:
            log_dict = {"loss": self.sum}
            log_num_dict = {"loss": self.num_examples}
            reduced_loss_dict = self.allreduce_fn(log_dict, log_num_dict)
            return reduced_loss_dict["loss"]
        else:
            return self.sum / self.num_examples

    def _allreducelog(self, log_dict: dict = {}, log_num_dict: dict = {}):
        return self.allreduce_fn(log_dict, log_num_dict)

    @property
    def averge_log(self):
        self.extra_log["SamplePerSec"] = self.num_examples / (
            time.time() - self.start_time
        )
        self.extra_log_num["SamplePerSec"] = 1.0 / self.world_size

        self.extra_log["total_acc_sample"] /= self.world_size
        self.extra_log["total_acc_sample"] += self.num_examples
        self.extra_log_num["total_acc_sample"] = 1.0 / self.world_size

        if self.world_size == 1 or self.allreduce_fn is None:
            return {
                k: (v / self.extra_log_num[k]) if self.extra_log[k] else 0
                for k, v in self.extra_log.items()
            }
        else:
            reduced_log = self._allreducelog(self.extra_log, self.extra_log_num)
            self.extra_log["total_acc_sample"] = reduced_log["total_acc_sample"]
            return reduced_log
        

    
class GroupedBatchIter(object):
    def __init__(self, it , gsz, drop_last=False):
        self.it = it 
        self.group_size = gsz
        self.drop_last = drop_last
    
    def __iter__(self):
        chunk = []
        for item in self.it:
            chunk.append(item)
            if len(chunk) == self.group_size:
                yield chunk
                chunk = []
        if not self.drop_last and chunk:
            yield chunk
    
    def __len__(self):
        if self.drop_last:
            return len(self.it) // self.group_size
        else:
            return (len(self.it) + self.group_size - 1)// self.group_size
    
class Trainer(object):
    def __init__(self,
                 training_args,
                 model,
                 train_data,
                 valid_data,
                 test_data,
                 optimizer,
                 lr_scheduler,
                 loss_log_dict,
                 ema):
        super().__init__()
        
        self.args = training_args
        
        logger.info(f'Training Args{training_args}')
        
        self.model = model
        self.train_data=train_data
        self.valid_data=valid_data
        self.test_data=test_data
        self.ema=ema
        self.device = training_args.device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.accelerator = self.build_accelerator(loss_log_dict=loss_log_dict)
        self.accelerator.set_up()
        
        self.model = self.accelerator.model 
        
        self.accelerator.build_dataloader(train_data, valid_data, test_data)
        
        self.state = TrainerState(args=self.args)
        
        self.begin_save_epoch = self.args.begin_save_epoch
        self.save_dir = Path(self.args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok = True)
        
        self.world_size = self.accelerator.world_size
        self.start_iter = 0

        if self.args.profiling:
            assert torch.cuda.is_available(), "Profiling only works on GPU trainings"
            self.prof_dir = Path(self.args.prof_dir)
            self.prof_dir.mkdir(exist_ok=True)
            self.prof = self.profiler_init()
        
    # def save_ckpt(self, name, state):
    #     # pass
    #     # if isinstance(state, TrainerState):
    #     #     self.accelerator.save_ckpt(name, asdict(state))
    #     # else:
    #     #     self.accelerator.save_ckpt(name, state)
    #     # self._save_rng_and_iter_state(self.save_dir)
    #     ckpt = self.model.state_dict()
    #     save_path = os.path.join(self.save_dir, str(name))
    #     torch.save(ckpt, save_path)
    
    def save_ckpt(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_path = os.path.join(self.save_dir, str(ckpt_id))
        # if self.rank == 0:
        #     if self.ema is not None:
        #         with self.ema.average_parameters():
        #             super().save_checkpoint(save_path, extra_state)
        #     else:
        #         super().save_checkpoint(save_path, extra_state)
        if self.ema is not None:
            with self.ema.average_parameters():
                ckpt = self.model.state_dict()
        else:
            ckpt = self.model.state_dict()
        torch.save(ckpt, save_path)


    def load_ckpt(self, path: Path, model_states_only: bool = False):
        checkpoint_list_path = path / "checkpoint_list.txt"
        latest_path = path / "latest"  # latest path for DeepSpeed
        

        checkpoint_last = None
        if model_states_only and self.args.finetune_from_checkpoint_id is not None:
            checkpoint_last = self.args.finetune_from_checkpoint_id
        elif checkpoint_list_path.exists():
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]
        elif latest_path.exists():
            with open(latest_path, "r") as f:
                latest_list = f.read().splitlines()
            if len(latest_list) > 0:
                checkpoint_last = latest_list[-1]
        # elif ENABLE_NNSCALER and nnscaler_latest_path.exists():
        #     checkpoint_last = f"latest.pt.{DeviceGroup().rank}"

        if checkpoint_last is not None:
            checkpoint_path = path / checkpoint_last
            if checkpoint_path.exists():
                if not model_states_only:
                    logger.info(f"Resume from checkpoint: {checkpoint_path}")
                else:
                    logger.info(f"Finetune from checkpoint: {checkpoint_path}")
                self.state = self.accelerator.load_checkpoint(
                    path,
                    checkpoint_last,
                    self.state,
                    model_states_only=model_states_only,
                )
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} does not exist.")
        else:
            logger.warning(
                f"Non-empty checkpoint_list.txt or latest file is not present in {path}, "
                f"or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
            )

    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint(self.save_dir)
        self.start_iteration = self._load_rng_and_iter_state(self.save_dir)
        logger.info(f"self.start_iteration = {self.start_iteration}.")

    def finetune_from_checkpoint(self):
        if self.finetune_from_checkpoint_dir is not None:
            self._load_checkpoint(
                self.finetune_from_checkpoint_dir, model_states_only=True
            )
        else:
            logger.warning("No finetune_from_checkpoint_dir is provided.")
        
    def train(self):
        logger.info('Start training...')
        logger.info(self.model)
        pnums = self.count_parameters()
        logger.info('total parameters:{:,}, trainable parameters: {:,}, ratio: {}%',pnums[0],pnums[1],pnums[1]*100/pnums[0])

        assert self.group_train_dataloader is not None 
        
        if hasattr(self.model, 'before_training'):
            self.model.before_training()
        if self.args.ifresume:
            self.resume()
        elif self.args.load_ckpt:
            self.load_ckpt(self.args.ckpt_path,model_states_only=True)
            
        while not self.should_stop():
            self.accelerator.before_epoch(self.state.epoch)
            
            logger.info(f'State epoch : {self.state.epoch}')
            
            loss_accumulator = LossAccumulator()
            interval_loss_accumulator = LogAccumulator(self.accelerator.world_size, None)# bug self.accelerator._allreducelog
            
            data_iterator = iter(self.group_train_dataloader)
            
            try:
                for grouped_data in data_iterator:
                    model_output = self.accelerator.train_step(grouped_data)
                    loss_accumulator.add(model_output.loss, model_output.num_samples)
                    interval_loss_accumulator.add(model_output.loss,model_output.num_samples,model_output.log_output)
                    
                    self.state.batch += 1
                    self.state.global_step += 1
                    self.state.sample += model_output.num_samples
                    
                    if self.should_batch_validate() and not self.args.profiling:
                        self.validate()
                    
                    if self.should_log():
                        log_output = self.build_log_output(interval_loss_accumulator.averge_loss,interval_loss_accumulator.averge_log)
                        interval_loss_accumulator.reset()
                        metric_logger.log(log_output, 'train_inner', self.state.global_step)
                    
                    if self.should_save_batch_ckpt():
                        ckpt_name = f'ckpt_E{self.state.epoch}_B{self.state.batch}.pt'
                        self.save_ckpt(ckpt_name,self.state)
                    
                    # if self.args.profiling:
                    #     prof.step()
            except StopIteration:
                logger.info('StopIteration')
            
            log_output = self.build_log_output(loss_accumulator.averge_loss)
            metric_logger.log(log_output,'train',self.state.global_step)
            
            self.state.batch = 0
            
            self.accelerator.barrier()
            if self.should_save_epoch_ckpt():
                ckpt_name = f'ckpt_E{self.state.epoch}.pt'
                self.save_ckpt(ckpt_name,self.state)
            
            if self.should_epoch_validate():
                valid_log = self.validate()
                self.validate(valid_type='test')
                
            if self.should_stop():
                break 
            self.state.epoch+=1
            
        if hasattr(self.model,'after_training'):
            self.model.after_training()
            
        if self.args.profiling:
            self.profiler_end(self.prof)

        logger.info('Finished training')
        
    
    def validate(self,valid_type='valid'):
        if self.valid_data_dataloader is None:
            logger.warning('no valid data')
            return
        
        if valid_type == 'test':
            if self.test_data is None:
                return
            logger.info(f'Start test: epoch:{self.state.epoch}, global step: {self.state.global_step}' )
        else:
            logger.info(f'Start valid: epoch:{self.state.epoch}, global step: {self.state.global_step}' )
        
        loss_accumulator = LossAccumulator()
        interval_loss_accumulator = LogAccumulator(self.accelerator.world_size, self.accelerator._allreducelog)
        
        # metric_accumulator = MetricAccumulator(self.world_size)
        
        loader = self.test_data_dataloader if valid_type == 'test' else self.valid_data_dataloader
        
        for idx, batch_data in enumerate(loader):
            output = self.accelerator.valid_step(batch_data, epoch=self.state.epoch)
            
            loss_accumulator.add(output.valid_loss, output.num_samples)
            
            interval_loss_accumulator.add(output.valid_loss,output.num_samples,output.extra_output)
            
            # metric_accumulator.add(output.pred,output.label)
            if (idx+1) % self.args.val_batch_log_interval ==  0:
                logger.info(f'{valid_type} batch: {idx+1}/ {len(self.valid_data_dataloader)}, loss:{output.valid_loss}')
        
        total_loss, num_samples = self.accelerator.sync_valid_loss(loss_accumulator.sum,loss_accumulator.num_examples)
        
        if num_samples >0:
            valid_loss = total_loss / num_samples
        else:
            valid_loss = 0
        
        valid_log = ValidLogOutput(
            valid_loss = valid_loss,
            num_samples=num_samples,
            epoch=self.state.epoch,
            extra_output = {**interval_loss_accumulator.averge_log}
        )
        
        metric_logger.log(valid_log,valid_type,self.state.global_step)
        return valid_log
                    
        
        
        
    def build_log_output(self,loss, extra_output=None):
        try:
            lr = self.accelerator.lr_scheduler.get_last_lr()[0]
        except:
            lr = 0.0
        
        return TrainingLogOutput(
            loss = loss,
            grad_scale=self.accelerator.grad_scale,
            lr=lr,
            epoch=self.state.epoch,
            batch=self.state.batch,
            total_samples=self.state.sample,
            global_step=self.state.global_step,
            extra_output=extra_output
        )
    
    def should_stop(self):
        assert (self.args.max_epochs>0), 'max_epochs must > 0'
    
        if self.state.epoch > self.args.max_epochs:
            return True 
        if self.state.global_step > self.args.max_steps:
            return True 
        return  False
    
    def should_save_batch_ckpt(self):
        assert (self.args.save_batch_interval>0), 'save_batch_interval must > 0'
        return  ((self.state.global_step % self.args.save_batch_interval) ==  0)
        
    
    def should_save_epoch_ckpt(self):
        assert (self.args.save_epoch_interval>0), 'save_epoch_interval must > 0'
        return  (self.state.epoch > self.begin_save_epoch) and (self.state.epoch % self.args.save_epoch_interval) ==  0
    
    # def should_log(self):
    #     assert (self.args.log_interval>0), 'log_interval must > 0'
    #     return  (self.state.global_step % self.args.log_interval) ==  0
    def should_log(self):
        assert (self.args.log_interval > 0)
        is_interval = (self.state.global_step % self.args.log_interval) == 0
        steps_per_epoch = len(self.group_train_dataloader)
        is_epoch_end = (self.state.batch == steps_per_epoch)
        return is_interval or is_epoch_end

    def should_batch_validate(self):
        assert (self.args.batch_validate_interval>0), 'batch_validate_interval must > 0'
        return  (self.state.global_step % self.args.batch_validate_interval) ==  0
    
    def should_epoch_validate(self):
        assert (self.args.epoch_validate_interval>0), 'epoch_validate_interval must > 0'
        return  ((self.state.epoch+1) % self.args.epoch_validate_interval) ==  0
    
    @property
    def group_train_dataloader(self,):
        return GroupedBatchIter(self.accelerator.train_data_loader, self.args.accu_steps,drop_last=True)
    
    @property
    def valid_data_dataloader(self,):
        return self.accelerator.valid_data_loader
    
    @property
    def test_data_dataloader(self,):
        return self.accelerator.test_data_loader
    
    def count_parameters(self):
        total_num = sum(p.numel()  for p in self.model.parameters())
        trainable_num = sum(p.numel()  for p in self.model.parameters() if p.requires_grad )
        return total_num,trainable_num 
    
    
    
    def build_accelerator(self,loss_log_dict):
        if self.args.strategy == 'standalone':
            return StandaloneAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.device
            )
        elif self.args.strategy == 'ddp':
            return DDPAccelerator(self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.device)
            # raise NotImplementedError('still working on ddp mode.')
        else:
            raise NotImplementedError('only support ddp and standalone mode.')
    
