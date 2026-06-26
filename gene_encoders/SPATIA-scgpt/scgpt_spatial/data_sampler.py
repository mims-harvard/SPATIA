from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SubsetsBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        subsets: List[Sequence[int]],
        batch_size: int,
        intra_subset_shuffle: bool = True,
        inter_subset_shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.intra_subset_shuffle = intra_subset_shuffle
        self.inter_subset_shuffle = inter_subset_shuffle
        self.drop_last = drop_last

        if intra_subset_shuffle:
            self.subset_samplers = [SubsetRandomSampler(subset) for subset in subsets]
        else:
            self.subset_samplers = [
                SubsetSequentialSampler(subset) for subset in subsets
            ]

        self.batch_samplers = [
            BatchSampler(sampler, batch_size, drop_last)
            for sampler in self.subset_samplers
        ]

        if inter_subset_shuffle:
            _id_to_batch_sampler = []
            for i, batch_sampler in enumerate(self.batch_samplers):
                _id_to_batch_sampler.extend([i] * len(batch_sampler))
            self._id_to_batch_sampler = np.array(_id_to_batch_sampler)

            assert len(self._id_to_batch_sampler) == len(self)

            self.batch_sampler_iterrators = [
                batch_sampler.__iter__() for batch_sampler in self.batch_samplers
            ]

    def __iter__(self) -> Iterable[List[int]]:
        if self.inter_subset_shuffle:
            random_idx = torch.randperm(len(self._id_to_batch_sampler))
            batch_sampler_ids = self._id_to_batch_sampler[random_idx]
            for batch_sampler_id in batch_sampler_ids:
                batch_sampler_iter = self.batch_sampler_iterrators[batch_sampler_id]
                yield next(batch_sampler_iter)
        else:
            for batch_sampler in self.batch_samplers:
                yield from batch_sampler

    def __len__(self) -> int:
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)
