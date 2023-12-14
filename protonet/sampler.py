from typing import Iterator, List
import random

from torch.utils.data import Sampler

from .dataset import ProtoDataset


class EposideSampler(Sampler):
    def __init__(
        self,
        ds: ProtoDataset,
        num_episodes: int,
        nc: int, # number of classes per episode
        ns: int, # number of support examples per class
        nq: int, # number of query examples per class
    ):
        assert nc <= ds.num_classes
        for label_idx in ds.label_ids:
            assert ds.num_samples_for_cls(label_idx) >= (ns + nq), f"{ds.num_samples_for_cls(label_idx)} ?? {ns} {nq}"

        self.ds = ds
        self.num_episodes = num_episodes
        self.nc = nc
        self.ns = ns
        self.nq = nq
        self.alpha = 0

    @property
    def batch_size(self) -> int:
        return self.nc * (self.ns + self.nq)

    @property
    def num_query_per_episode(self) -> int:
        return self.nc * self.nq

    @property
    def num_support_per_episode(self) -> int:
        return self.nc * self.ns

    def generate_episode(self) -> List[int]:
        selected_classes = random.sample(
            self.ds.label_ids,
            k=self.nc,
        )
        query_sample_ids = []
        support_sample_ids = []
        for label_idx in selected_classes:
            sample_ids = random.sample(
                self.ds.get_samples_for_cls(label_idx),
                k=self.ns + self.nq,
            )
            query_sample_ids += sample_ids[:self.nq]
            support_sample_ids += sample_ids[self.nq:]

        return query_sample_ids + support_sample_ids

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            yield self.generate_episode()