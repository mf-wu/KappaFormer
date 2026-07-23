class DdpAccelerator(SingleNodeAccelerator):
    def __init__(
        self,
        args,
        model,
        optimizer,
        lr_scheduler,
        ema=None,
    ) -> None:
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda", ema=ema)

    def set_up(self):
        super().set_up()
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        ddp_timeout = os.environ.get("DDP_TIMEOUT_MINUTES", None)
        logger.critical(
            f"Initializing DDP by env://. word size: {self.world_size}, rank: {self.rank}, "
            f"local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}, "
            f"DDP_TIMEOUT_MINUTES: {ddp_timeout}"
        )
        torch.distributed.init_process_group(
            backend=self.args.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(days=2),
        )

        torch.distributed.barrier()

        logger.success("DDP initialized.")

        self.model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
        )
        if self.model.checkpoint_loaded:
            logger.info("Reloading checkpoint after DDP to ensure correctness.")
            self.ddp_model.module.reload_checkpoint()

        self.ddp_model = torch_compile(self.ddp_model, self.args.compile)

    def barrier(self):
        torch.distributed.barrier()

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.ddp_model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}
        for idx, batch_data in enumerate(grouped_batch_data):
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            # No sync for gradient accumulation
            maybe_no_sync = (
                self.ddp_model.no_sync()
                if idx != len(grouped_batch_data) - 1
                else nullcontext()
            )

            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info("loss is nan or inf. skip this batch")
                    # continue
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
                model_output.num_examples,
            )
            sample_count += model_output.num_examples
            total_loss += model_output.loss * model_output.num_examples
            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)
            if self.ema is not None:
                self.ema.update()

        self.lr_scheduler.step()
        model_output.num_examples = sample_count
        model_output.loss = safe_div(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def build_data_loader(
        self,
        train_data: FoundationModelDataset,
        val_data: FoundationModelDataset,
        test_data: FoundationModelDataset = None,
    ):
        if self.args.dynamic_loader:
            self.train_sampler = DynamicDistributedSampler(
                dataset=train_data,
                batch_by_size_fn=batch_by_size,
                max_tokens=self.args.max_tokens,
                max_length=self.args.max_length,
                num_tokens_fn=train_data.num_tokens,
                shuffle=True,
                drop_last=False,
                num_replicas=self.world_size,
                rank=self.rank,
            )
            self.train_data_loader = DataLoader(
                dataset=train_data,
                collate_fn=train_data.collate,
                batch_sampler=self.train_sampler,
            )
        elif self.args.ifstack:
            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            self.train_sampler = None
            self.train_data_loader = DataLoader(
                train_data,
                batch_size=train_batch_size_per_gpu,
                collate_fn=train_data.collate,
                drop_last=True,
                num_workers=0,
            )
        else:
            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            if not isinstance(train_data, IterableDataset):
                if self.args.use_unified_batch_sampler:
                    self.train_sampler = UnifiedDataSampler(
                        train_data,
                        self.args.dataset_split_raito,
                        self.args.dataset_micro_batch_size,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        seed=self.args.seed,
                    )
                    self.train_data_loader = DataLoader(
                        train_data,
                        batch_sampler=self.train_sampler,
                        collate_fn=train_data.collate,
                        num_workers=self.args.unified_data_num_workers,
                        pin_memory=True,
                        persistent_workers=True
                        if self.args.unified_data_num_workers > 0
                        else False,
                        prefetch_factor=4
                        if self.args.unified_data_num_workers > 0
                        else None,
                    )
                else:
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
            elif self.args.use_dali_pipeline:
                self.train_sampler = None
                self.train_data_loader = DataLoader(
                    train_data,
                    batch_size=None,
                    collate_fn=train_data.collate,
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
                valid_sampler = UnifiedDataSampler(
                    val_data,
                    self.args.dataset_split_raito,
                    self.args.dataset_micro_batch_size,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=self.args.seed,
                )
                self.valid_data_loader = DataLoader(
                    val_data,
                    batch_sampler=valid_sampler,
                    collate_fn=val_data.collate,
                )
            elif self.args.use_dali_pipeline:
                self.valid_data_loader = DataLoader(
                    val_data,
                    batch_size=None,
                    collate_fn=val_data.collate,
                )
            else:
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
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            if self.ema is not None:
                with self.ema.average_parameters():
                    super().save_checkpoint(ckpt_id, extra_state)
            else:
                super().save_checkpoint(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_examples = torch.Tensor([num_examples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

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
        if self.args.use_unified_batch_sampler:
            self.train_data_loader.batch_sampler.set_skip_batches(
                start_iteration * self.args.gradient_accumulation_steps, 0
            )
            return True
        else:
            return False