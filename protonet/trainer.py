from typing import Literal, Union, Optional
from multiprocessing import cpu_count
import os
import math
from collections import defaultdict
import random

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import lightning as L

from .sampler import EpisodeSampler
from .model import ProtoNet
from .dataset import ProtoDataset


def kl_divergence_between_gaussian(
    mu_1: float,
    std_1: float,
    mu_2: float,
    std_2: float,
) -> float:
    # ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    var_1 = std_1**2
    var_2 = std_2**2

    return (
        math.log(std_2 / std_1) + ((var_1 + (mu_1 - mu_2) ** 2) / (2 * var_2)) - 1 / 2
    )


class ProtoTrainer:
    def __init__(
        self,
        model: ProtoNet,
        train_sampler: EpisodeSampler,
        criterion,
        optimizer,
        val_dataset: ProtoDataset = None,
        test_dataset: ProtoDataset = None,
        support_dataset: ProtoDataset = None,
        num_epochs: int = 1,
        fabric: L.Fabric = None,
        root_dir: str = ".",
        device: Literal["cpu", "cuda"] = "cuda",
        precision: Optional[Literal["16-mixed", "32"]] = None,
        scheduler=None,
        num_workers: Union[int, Literal["max"]] = 0,
        checkpoint_save_path: Optional[str] = None,
    ):
        if (val_dataset is not None) or (test_dataset is not None):
            assert support_dataset is not None, "support dataset must be given"

        self.train_sampler = train_sampler
        self.criterion = criterion
        self.fabric = fabric or L.Fabric(
            accelerator=device,
            precision=precision,
            loggers=L.fabric.loggers.TensorBoardLogger(
                root_dir=root_dir,
                name="runs",
            ),
        )

        self.model, self.optimizer = self.fabric.setup(
            model,
            optimizer,
        )

        self.scheduler = scheduler

        self.dataloaders = {}

        if num_workers == "max":
            num_workers = cpu_count()

        self.dataloaders["train"] = self.fabric.setup_dataloaders(
            DataLoader(
                train_sampler.ds,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        )
        if val_dataset is not None:
            self.dataloaders["val"] = self.fabric.setup_dataloaders(
                DataLoader(
                    val_dataset,
                    batch_size=32,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
        if test_dataset is not None:
            self.dataloaders["test"] = self.fabric.setup_dataloaders(
                DataLoader(
                    test_dataset,
                    batch_size=32,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
        if support_dataset is not None:
            self.dataloaders["support"] = self.fabric.setup_dataloaders(
                DataLoader(
                    support_dataset,
                    batch_size=32,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )

        self.num_epochs = num_epochs
        self.checkpoint_save_path = checkpoint_save_path or os.getcwd()

    def train(self):
        sampler = self.train_sampler
        dataloader = self.dataloaders["train"]
        best_acc = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            if self.scheduler:
                self.fabric.log(
                    "learning_rate",
                    self.scheduler.get_last_lr()[0],
                    step=epoch,
                )

            loop = tqdm(
                enumerate(dataloader),
                total=sampler.num_episodes,
                leave=True,
            )
            offset = sampler.num_episodes * epoch
            for i, (batch, _) in loop:
                self.optimizer.zero_grad()

                targets = self.get_targets(sampler)

                logits = self.model.forward_train(
                    batch[: sampler.num_query_per_episode, :],
                    batch[-sampler.num_support_per_episode :, :].unflatten(
                        dim=0, sizes=(sampler.nc, sampler.ns)
                    ),
                )
                loss = self.criterion(logits, targets)
                self.fabric.backward(loss)

                self.optimizer.step()
                self.fabric.log("loss", loss.item(), step=i + offset)

                loop.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item(), accuracy=best_acc)

            if self.scheduler:
                self.scheduler.step()

            proto_points: Tensor = self.compute_protopoints()
            # proto_points: C x d

            current_val_acc = self.validation(
                proto_points=proto_points,
                step=epoch,
            )

            state = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "accuracy": current_val_acc,
            }

            if current_val_acc > best_acc:
                self.fabric.save(
                    os.path.join(self.checkpoint_save_path, "best.ckpt"),
                    state,
                )
                best_acc = current_val_acc

            self.fabric.save(
                os.path.join(self.checkpoint_save_path, "last.ckpt"),
                state,
            )

    def get_targets(self, sampler: EpisodeSampler) -> torch.Tensor:
        return torch.arange(
            sampler.nc,
            dtype=torch.long,
            device=self.fabric.device,
        ).repeat_interleave(
            sampler.nq,
        )

    @torch.no_grad()
    def compute_protopoints(self) -> Optional[Tensor]:
        if "support" not in self.dataloaders:
            return

        dataloader = self.dataloaders["support"]
        proto_buckets = defaultdict(list)
        for batch, targets in tqdm(dataloader, desc="computing proto points"):
            batch_size = batch.shape[0]

            embeddings = self.model(batch)
            # embeddings: B x d
            for i in range(batch_size):
                proto_buckets[targets[i].item()].append(embeddings[i])

        proto_points = []
        for key in sorted(proto_buckets.keys()):
            proto_points.append(torch.stack(proto_buckets[key], dim=0).mean(dim=0))

        proto_points = torch.stack(proto_points, dim=0)
        # proto_points: C x d
        return proto_points

    def validation(
        self,
        proto_points: Tensor,
        step: int = 0,
    ) -> float:
        return self._run_single_stage("val", proto_points, step=step)

    def test(
        self,
        proto_points: Tensor,
        step: int = 0,
    ) -> float:
        return self._run_single_stage("test", proto_points, step=step)

    @torch.no_grad()
    def _run_single_stage(
        self,
        stage: Literal["val", "test"],
        proto_points: Tensor,
        step: int = 0,
    ) -> float:
        """_summary_

        Args:
            stage (Literal[&quot;val&quot;, &quot;test&quot;]): _description_
            proto_points (Tensor): C x d Tensor where C is the number of labels (needs to be ordered correctly)
            step (int, optional): _description_. Defaults to 0.

        Returns:
            float: _description_
        """
        if stage not in self.dataloaders:
            return (math.inf, 0)

        self.model.eval()

        dataloader = self.dataloaders[stage]

        acc = list()
        for batch, targets in tqdm(
            dataloader, desc=f"running {stage} stage with step: {step}"
        ):
            batch_size = batch.shape[0]

            preds = self.model.predict(batch, proto_points)
            # preds: B,

            for i in range(batch_size):
                acc.append((preds[i] == targets[i]).item())

        acc = sum(acc) / len(acc)

        self.fabric.log(f"{stage}/accuracy", acc, step=step)

        return acc

    @torch.no_grad()
    def analyse_model(
        self,
        k: int = 2,
        n_max: int = 5,
    ):
        stage = "val"
        self.model.eval()

        # analyse best potential `k`
        ds: ProtoDataset = self.dataloaders[stage].dataset

        pop_proto_points = []

        for label_idx in tqdm(ds.label_ids, desc="computing population mean"):
            sample_ids = ds.get_samples_for_cls(label_idx)
            proto_point = []
            for idx in sample_ids:
                data, _ = ds[idx]
                proto_point.append(self.model(data.unsqueeze(0)))
            proto_point = torch.cat(proto_point).mean(dim=0)
            pop_proto_points.append(proto_point)

        pop_proto_points = torch.stack(pop_proto_points, dim=0)
        # pop_proto_points: C x d

        k_shots = [2**i for i in range(n_max + 1)]
        pop_to_sample_proto_dists = []
        selected_proto_points = None

        assert k in k_shots

        for k_shot in k_shots:
            sample_proto_points = []
            for label_idx in tqdm(
                ds.label_ids, desc=f"computing sample mean with k={k_shot}"
            ):
                sample_ids = ds.get_samples_for_cls(label_idx)
                sample_ids = random.sample(sample_ids, k=min(k_shot, len(sample_ids)))
                proto_point = []
                for idx in sample_ids:
                    data, _ = ds[idx]
                    proto_point.append(self.model(data.unsqueeze(0)))
                proto_point = torch.cat(proto_point).mean(dim=0)
                sample_proto_points.append(proto_point)
            sample_proto_points = torch.stack(sample_proto_points, dim=0)
            # sample_proto_points: C x d
            if k_shot == k:
                selected_proto_points = sample_proto_points

            pop_to_sample_proto_dists.append(
                (pop_proto_points - sample_proto_points).norm(dim=1, p=2)
                # C
            )
        pop_to_sample_proto_dists = torch.stack(pop_to_sample_proto_dists, dim=0)
        # pop_to_sample_proto_dists: (n_max+1) x C

        # model class variance
        # selected_proto_points: C x d
        means = []
        stds = []
        for label_idx in tqdm(ds.label_ids, desc="computing class mean/std"):
            sample_ids = ds.get_samples_for_cls(label_idx)
            dists = []
            for idx in sample_ids:
                data, _ = ds[idx]
                embed = self.model(data.unsqueeze(0))
                class_dists = self.model.dist_layer(
                    embed,
                    selected_proto_points,
                ).flatten()
                # class_dists: C

                dists.append(class_dists)

            dists = torch.stack(dists, dim=0)
            # dists: N x C

            means.append(dists.mean(dim=0))
            stds.append(dists.std(dim=0))

        means = torch.stack(means, dim=0)
        # means: C x C

        stds = torch.stack(stds, dim=0)
        # stds: C x C

        divergences = torch.zeros_like(means)

        for i in range(means.shape[0]):
            mu_1 = means[i, i]
            std_1 = stds[i, i]
            for j in range(means.shape[1]):
                divergences[i, j] = kl_divergence_between_gaussian(
                    mu_1,
                    std_1,
                    means[i, j],
                    stds[i, j],
                )

        return pop_to_sample_proto_dists, k_shots, means, stds, divergences

    def run(self):
        self.train()
        proto_points: Tensor = self.compute_protopoints()
        # proto_points: C x d
        self.test(proto_points)
