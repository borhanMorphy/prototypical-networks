from typing import List, Optional, Union, Callable
from dataclasses import dataclass
from collections import defaultdict

from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Label:
    name: str
    parent: Optional[str] = None


class ProtoDataset(Dataset):
    def __init__(
        self, 
        file_paths: List[str], 
        labels: List[Label], 
        transforms: Callable = None,
    ):
        assert len(file_paths) == len(labels)
        self._file_paths = file_paths
        self._labels = labels

        self._unq_labels = list(
            set(map(lambda label: label.name, labels))
        )

        self._label_to_idx = {
            label: i
            for i, label in enumerate(self._unq_labels)
        }
        self._idx_to_label = {
            i: label
            for i, label in enumerate(self._unq_labels)
        }
        self._label_idx_grouped_samples = defaultdict(list)
        for sample_idx, label in enumerate(self._labels):
            label_idx = self._label_to_idx[label.name]
            self._label_idx_grouped_samples[label_idx].append(
                sample_idx
            )

        self._transforms = transforms

    @property
    def label_ids(self) -> List[int]:
        return sorted(self._idx_to_label.keys())

    @property
    def labels(self) -> List[str]:
        return self._unq_labels

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    def get_samples_for_cls(self, label: Union[str, int]) -> List[int]:
        if isinstance(label, str):
            label = self._label_to_idx[label]
        return self._label_idx_grouped_samples[label]

    def num_samples_for_cls(self, label: Union[str, int]) -> int:
        return len(self.get_samples_for_cls(label))

    def load_file(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, index: int) -> Tensor:
        data = self.load_file(
            self._file_paths[index]
        )

        if self._transforms:
            return self._transforms(data)

        return data
