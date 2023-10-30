from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T # TODO


@dataclass
class Metadata:
    sample_class: str
    sample_super_class: Optional[str] = None


# OmniGlotDataset
class ProtoDataset(Dataset):
    def __init__(self, file_paths: List[str], meta: List[Metadata]):
        assert len(file_paths) == len(meta)
        self._file_paths = file_paths
        self._meta = meta

        self._labels = list(
            set(
                map(lambda m: m.sample_class, meta)
            )
        )
        self._label_to_idx = {
            label: i
            for i, label in enumerate(self._labels)
        }
        self._label_idx_grouped_samples = defaultdict(list)
        for sample_idx, m in enumerate(meta):
            label_idx = self._label_to_idx[m.sample_class]
            self._label_idx_grouped_samples[label_idx].append(
                sample_idx
            )

        self._transforms = T.Compose([
            T.Resize((28, 28)),
            T.ToTensor(),
        ])

    @property
    def label_ids(self) -> List[int]:
        return sorted(self._label_to_idx.values())

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    def get_samples_for_cls(self, label: Union[str, int]) -> List[int]:
        if isinstance(label, str):
            label = self._label_to_idx[label]
        return self._label_idx_grouped_samples[label]

    def num_samples_for_cls(self, label: Union[str, int]) -> int:
        return len(self.get_samples_for_cls(label))

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, index: int):
        img = Image.open(self._file_paths[index]).convert('L')

        return self._transforms(img)

    @classmethod
    def from_path(cls, folder_path: str):
        # omniglot/images_background/
        #   /<super_class>
        #       /<class>
        #           /<img_file.png>
        file_paths = list()
        meta = list()
        for img_file_path in Path(folder_path).rglob("*.png"):
            file_paths.append(img_file_path)
            *_, alphabet, character_index, _ = img_file_path.parts

            meta.append(
                Metadata(
                    sample_super_class=alphabet,
                    sample_class=alphabet + "-" + character_index,
                )
            )

        return cls(file_paths, meta)
