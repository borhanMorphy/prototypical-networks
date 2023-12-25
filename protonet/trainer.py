from typing import Dict, Literal, Union, Optional, Tuple
from multiprocessing import cpu_count
import os
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import lightning as L

from .sampler import EpisodeSampler
from .model import ProtoNet


class ProtoTrainer():
    def __init__(
        self,
        model: ProtoNet,
        samplers: Dict[Literal["train", "val", "test"], EpisodeSampler],
        criterion,
        optimizer,
        num_epochs: int = 1,
        fabric: L.Fabric = None,
        scheduler = None,
        num_workers: Union[int, Literal["max"]] = 0,
        checkpoint_save_path: Optional[str] = None,
    ):
        assert "train" in samplers

        self.samplers = samplers
        self.criterion = criterion
        self.fabric = fabric or L.Fabric(
            accelerator="cuda",
            precision="16-mixed",
            loggers=L.fabric.loggers.TensorBoardLogger(
                root_dir=".",
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

        for stage, sampler in self.samplers.items():
            self.dataloaders[stage] = self.fabric.setup_dataloaders(
                DataLoader(
                    sampler.ds,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
        self.num_epochs = num_epochs
        self.checkpoint_save_path = checkpoint_save_path or os.getcwd()


    def train(self):
        sampler = self.samplers["train"]
        dataloader = self.dataloaders["train"]
        best_acc = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            if self.scheduler:
                self.fabric.log("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)

            loop = tqdm(enumerate(dataloader), total=sampler.num_episodes, leave=True)
            offset = sampler.num_episodes * epoch
            for i, batch in loop:
                self.optimizer.zero_grad()

                targets = self.get_targets(sampler)

                logits = self.model.forward_train(
                    batch[:sampler.num_query_per_episode, :],
                    batch[-sampler.num_support_per_episode:, :].unflatten(
                        dim=0, sizes=(sampler.nc, sampler.ns)
                    ),
                )
                loss = self.criterion(logits, targets)
                self.fabric.backward(loss)

                self.optimizer.step()
                self.fabric.log("loss", loss.item(), step=i + offset)

                loop.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item())

            if self.scheduler:
                self.scheduler.step()

            current_val_loss, current_val_acc = self.validation(step=epoch)

            state = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "accuracy": current_val_acc,
                "loss": current_val_loss,
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

    def validation(self, step: int = 0) -> Tuple[float, float]:
        return self._run_single_stage("val", step=step)

    def test(self, step: int = 0) -> Tuple[float, float]:
        return self._run_single_stage("test", step=step)

    @torch.no_grad()
    def _run_single_stage(self, stage: Literal["val", "test"], step: int = 0) -> Tuple[float, float]:
        if stage not in self.samplers:
            return (math.inf, 0)

        self.model.eval()

        sampler = self.samplers[stage]
        dataloader = self.dataloaders[stage]

        acc = list()
        losses = list()
        for i, batch in tqdm(enumerate(dataloader), total=sampler.num_episodes):
            targets = self.get_targets(sampler)
            
            logits = self.model.forward_train(
                # query_samples: (Nc * Nq) x *shape
                batch[:sampler.num_query_per_episode, :],
                # support_samples: Nc x Ns x *shape
                batch[-sampler.num_support_per_episode:, :].unflatten(
                    dim=0, sizes=(sampler.nc, sampler.ns)
                ),
            )
            preds = logits.argmax(dim=1)
            loss = self.criterion(logits, targets)
            acc.append((preds == targets).sum().item() / sampler.num_query_per_episode)
            losses.append(loss.item())

        loss = sum(losses) / len(losses)
        acc = sum(acc) / len(acc)

        self.fabric.log(f"{stage}/loss", loss, step=step)
        self.fabric.log(f"{stage}/accuracy", acc, step=step)

        return loss, acc


    def run(self):
        self.train()
        self.test()
