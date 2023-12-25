from typing import Iterator, List
from collections import defaultdict
import random
import math

from torch.utils.data import Sampler

from .dataset import ProtoDataset


class EpisodeSampler(Sampler):
    def __init__(
        self,
        ds: ProtoDataset,
        num_episodes: int,
        nc: int,  # number of classes per episode
        ns: int,  # number of support examples per class
        nq: int,  # number of query examples per class
    ):
        assert nc <= ds.num_classes
        for label_idx in ds.label_ids:
            assert ds.num_samples_for_cls(label_idx) >= (
                ns + nq
            ), f"{ds.num_samples_for_cls(label_idx)} ?? {ns} {nq}"

        self.ds = ds
        self.num_episodes = num_episodes
        self.nc = nc
        self.ns = ns
        self.nq = nq

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
            query_sample_ids += sample_ids[: self.nq]
            support_sample_ids += sample_ids[self.nq :]

        return query_sample_ids + support_sample_ids

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            yield self.generate_episode()


class DynamicEpisodeSampler(EpisodeSampler):
    def __init__(
        self,
        ds: ProtoDataset,
        num_episodes: int,
        nc: int,  # number of classes per episode
        ns: int,  # number of support examples per class
        nq: int,  # number of query examples per class
        scaling: int,
        scaling_formula: str,
        percent_formula: str,
    ):
        parents = defaultdict(set)
        for label in ds._labels:
            parents[label.parent].add(label.name)

        for parent_cls, child_labels in parents.items():
            # assert len(child_labels) >= nc, f"{len(child_labels)} == {nc}"

            for label in child_labels:
                num_samples = ds.num_samples_for_cls(f"{parent_cls}:{label}")
                assert num_samples >= (ns + nq), f"{num_samples} ?? {ns} {nq}"

        super().__init__(
            ds=ds,
            num_episodes=num_episodes,
            nc=nc,
            ns=ns,
            nq=nq,
        )
        self.parents = parents
        self.scaling = scaling
        self.scaling_formula = scaling_formula
        self.percent_formula = percent_formula

    def generate_episode(self) -> List[int]:
        selected_parent = random.choice(list(self.parents.keys()))
        eval_params = {
            "math": math,
            "self": self,
            "nc": self.nc,
            "scaling": self.scaling,
            "parent": len(list(self.parents[selected_parent])),
        }
        self.nc = eval(self.percent_formula, eval_params)

        selected_labels = random.sample(
            list(self.parents[selected_parent]),
            k=self.nc,
        )
        print(
            selected_parent,
            len(selected_labels),
            len(list(self.parents[selected_parent])),
        )
        query_sample_ids = []
        support_sample_ids = []
        for label in selected_labels:
            sample_ids = random.sample(
                self.ds.get_samples_for_cls(f"{selected_parent}:{label}"),
                k=self.ns + self.nq,
            )
            query_sample_ids += sample_ids[: self.nq]
            support_sample_ids += sample_ids[self.nq :]

        return query_sample_ids + support_sample_ids

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            yield self.generate_episode()
            eval_params = {"math": math, "self": self, "scaling": self.scaling}
            self.scaling = eval(self.scaling_formula, eval_params)
