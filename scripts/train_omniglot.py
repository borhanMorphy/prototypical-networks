from typing import List, Callable
from pathlib import Path
import random
from copy import deepcopy
import argparse

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
from PIL import Image
import lightning as L

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


def main(args):

    config = pn.config.EpisodeConfig(
        num_episodes=args.num_episodes,
        nc=args.num_classes,  # m classes
        ns=args.num_support,  # k-shot
        nq=args.num_query,  # n query per class
    )

    if args.model == "tapnet":
        encoder = Encoder(features=[1, 64, 64, 64, 64 + config.nc])
        model = pn.TapNet(
            encoder,
            embed_dim=64 + config.nc,
            num_classes=config.nc,
        )
    elif args.model == "protonet":
        encoder = Encoder(features=[1, 64, 64, 64, 64])
        model = pn.ProtoNet(encoder)
    elif args.model == "sen-protonet":
        encoder = Encoder(features=[1, 64, 64, 64, 64])
        dist_layer = pn.metric.SEN(
            config.nc,
            config.nq,
            pos_epsilon=1.0,
            neg_epsilon=-1e-7,
        )
        model = pn.SENProtoNet(encoder, dist_layer=dist_layer)

    transforms = T.Compose(
        [
            T.Resize((28, 28)),
            T.ToTensor(),
        ]
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
        lr=args.learning_rate,
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
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        device=args.device,
        root_dir="./logs/",
        num_workers="max",
    )

    trainer.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["tapnet", "protonet", "sen-protonet"],
        default="protonet",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-classes", "-nc", type=int, default=60)
    ap.add_argument("--num-support", "-ns", type=int, default=1)
    ap.add_argument("--num-query", "-nq", type=int, default=5)

    ap.add_argument("--num-episodes", "-ep", type=int, default=2000)
    ap.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    ap.add_argument("--num-epochs", "-ne", type=int, default=10)

    args = ap.parse_args()

    L.seed_everything(args.seed)

    main(args)
