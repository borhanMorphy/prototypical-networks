from typing import List

import torch
import torch.nn as nn
from torch import Tensor

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


def main():
    encoder = Encoder()
    model = pn.ProtoNet(encoder)
    train_config = pn.config.EpisodeConfig(
        num_episodes=2000,
        nc=60,              # 60 classes
        ns=1,               # 1-shot
        nq=5,               # 5 query per class
    )
    test_config = pn.config.EpisodeConfig(
        num_episodes=1000,
        nc=5,               # 5-way
        ns=1,               # 1-shot
        nq=1,               # 1 query per class
    )

    train_ds = pn.dataset.ProtoDataset.from_path("omniglot/images_background")
    test_ds = pn.dataset.ProtoDataset.from_path("omniglot/images_evaluation")

    samplers = {}

    samplers["train"] = pn.sampler.EposideSampler(
        train_ds,
        train_config.num_episodes,
        train_config.nc,
        train_config.ns,
        train_config.nq,
    )

    samplers["val"] = pn.sampler.EposideSampler(
        test_ds,
        test_config.num_episodes,
        test_config.nc,
        test_config.ns,
        test_config.nq,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = pn.ProtoTrainer(
        model,
        samplers,
        criterion,
        optimizer,
        num_epochs=1,
    )

    trainer.run()


if __name__ == '__main__':
    main()