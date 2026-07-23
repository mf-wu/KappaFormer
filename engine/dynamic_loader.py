# -*- coding: utf-8 -*-

import math
from typing import Optional

# import deepspeed
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Sampler

from engine.logging import logger


class DynamicDistributedSampler(Sampler):
    def __init__(
        self, dataset, batch_by_size_fn, num_tokens_fn, collate_fn=None, *args, **kwargs
    ):
        self.dataset = dataset
        self.batch_by_size_fn = batch_by_size_fn
        self.num_tokens_fn = num_tokens_fn
        self.collate_fn = collate_fn
        self.drop_last = kwargs.pop("drop_last", False)
        self.max_samples = kwargs.pop("max_sample", None)
        self.max_tokens = kwargs.pop("max_tokens", None)
        self.max_length = kwargs.pop("max_length", 1024)
        self.num_replicas = kwargs.pop("num_replicas", 1)
        self.rank = kwargs.pop("rank", 0)
        self.shuffle = kwargs.pop("shuffle", False)
        self.seed = kwargs.pop("seed", 0)
        self.epoch = kwargs.pop("epoch", 0)

        # define micro batches, only sequence of micro batches will be shuffled
        self.__set_micro_batch_indices()
        super().__init__(dataset, *args, **kwargs)

    def sort_dataset(self, indices):
        sort_indices = sorted(indices, key=lambda x: self.num_tokens_fn(x))
        return sort_indices

    def __set_micro_batch_indices(self):
        indices = list(range(len(self.dataset)))
        # shuffle the indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # indices = self.sort_dataset(indices)
        self.batches = self.batch_by_size_fn(
            indices=indices,
            max_length=self.max_length,
            num_tokens_fn=self.num_tokens_fn,
            max_tokens=self.max_tokens,
            max_samples=self.max_samples,
            required_batch_size_multiple=1,
        )

        length = len(self.batches)

        if self.drop_last and length % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (length - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(length / self.num_replicas)  # type: ignore[arg-type]

    def __set_dist_indices(self, length):
        if self.drop_last and length % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (length - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(length / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(length))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(local_indices) == self.num_samples
        self.epoch += 1
        return local_indices

    def set_epoch(self, epoch: Optional[int] = None) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        if epoch is None:
            self.epoch += 1
        else:
            self.epoch = epoch

    def __iter__(self):
        self.__set_micro_batch_indices()
        local_batch_id = self.__set_dist_indices(len(self.batches))
        for batch_indices in local_batch_id:
            yield self.batches[batch_indices]

        self.set_epoch()

    def __len__(self):
        return self.num_samples


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        sampler,
        num_tokens_fn,
        num_buckets=128,
        min_size=0,
        max_size=1000,
        max_tokens=None,
        max_sentences=None,
        drop_last=False,
    ):
        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_tokens_fn = num_tokens_fn
        self.num_buckets = num_buckets

        self.min_size = min_size
        self.max_size = max_size

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert (
            max_tokens is not None or max_sentences is not None
        ), "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens if max_tokens is not None else float("Inf")
        self.max_sentences = (
            max_sentences if max_sentences is not None else float("Inf")
        )
        self.drop_last = drop_last

    def is_batch_full(self, num_tokens, batch):
        if len(batch) == 0:
            return False
        if len(batch) == self.max_sentences:
            return True
        if num_tokens > self.max_tokens:
            return True
        return False

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)]
        sample_len = [0] * self.num_buckets

        for idx in self.sampler:
            idx_length = self.num_tokens_fn(idx)
            if not (self.min_size <= idx_length <= self.max_size):
                # Ignore the sentence that not in the range of min_size and max_size
                continue

            index_buckets = math.floor(
                (idx_length - self.min_size)
                / (self.max_size - self.min_size + 1)
                * self.num_buckets
            )
            sample_len[index_buckets] = max(sample_len[index_buckets], idx_length)

            num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
            if self.is_batch_full(num_tokens, buckets[index_buckets]):
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                sample_len[index_buckets] = 0

            buckets[index_buckets].append(idx)

        leftover_batch = []
        leftover_sample_len = 0
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            idx_length = self.num_tokens_fn(idx)
            leftover_sample_len = max(leftover_sample_len, idx_length)
            num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
            if self.is_batch_full(num_tokens, leftover_batch):
                yield leftover_batch
                leftover_batch = []
                leftover_sample_len = 0
            leftover_batch.append(idx)

        if len(leftover_batch) > 0 and not self.drop_last:
            yield leftover_batch

    def __len__(self):
        raise NotImplementedError
        # we do not know the exactly batch size, so do not call len(dataloader)