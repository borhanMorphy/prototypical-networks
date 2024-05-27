from typing import List, Callable
from pathlib import Path
import random
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
from PIL import Image

import protonet as pn


class Encoder(nn.Module):
    def __init__(self, features: List[int] = [1, 64, 64, 64, 64]):
        super().__init__()
        self.nn = nn.Sequential(
            *[
                self.make_conv_block(din, dout)
                for din, dout in zip(features[:-1], features[1:])
            ],
            nn.Flatten(start_dim=1),
        )

    @staticmethod
    def make_conv_block(din: int, dout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x)


class OmniGlotDataset(pn.ProtoDataset):
    def load_file(self, file_path: str):
        return Image.open(file_path).convert("L")

    @classmethod
    def from_path(cls, folder_path: str, transforms: Callable = None):
        # <folder_path>
        #   /<super_class>
        #       /<class>
        #           /<img_file.png>
        file_paths = list()
        labels = list()
        for img_file_path in Path(folder_path).rglob("*.png"):
            file_paths.append(img_file_path)
            *_, alphabet, character_index, _ = img_file_path.parts

            labels.append(
                pn.dataset.Label(
                    name=alphabet + "-" + character_index,
                    parent=alphabet,
                )
            )

        return cls(file_paths, labels, transforms=transforms)


def main():
    encoder = Encoder()
    model = pn.ProtoNet(encoder)
    transforms = T.Compose(
        [
            T.Resize((28, 28)),
            T.ToTensor(),
        ]
    )

    config = pn.config.EpisodeConfig(
        num_episodes=2000,
        nc=60,  # 60 classes
        ns=1,  # 1-shot
        nq=5,  # 5 query per class
    )

    train_ds = OmniGlotDataset.from_path(
        "data/omniglot/images_background",
        transforms=transforms,
    )

    val_ds = OmniGlotDataset.from_path(
        "data/omniglot/images_evaluation",
        transforms=transforms,
    )

    ids_to_pick = []
    for label_idx in val_ds.label_ids:
        ids = val_ds.get_samples_for_cls(label_idx)
        ids_to_pick += random.sample(ids, k=config.ns)

    support_ds = OmniGlotDataset(
        file_paths=[val_ds._file_paths[idx] for idx in ids_to_pick],
        labels=[deepcopy(val_ds._labels[idx]) for idx in ids_to_pick],
        transforms=transforms,
    )

    train_sampler = pn.sampler.EpisodeSampler(
        train_ds,
        config.num_episodes,
        config.nc,
        config.ns,
        config.nq,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )

    trainer = pn.ProtoTrainer(
        model,
        train_sampler,
        criterion,
        optimizer,
        val_dataset=val_ds,
        support_dataset=support_ds,
        num_epochs=1,
        scheduler=scheduler,
        device="cpu",
        num_workers="max",
    )

    trainer.run()


if __name__ == "__main__":
    main()
