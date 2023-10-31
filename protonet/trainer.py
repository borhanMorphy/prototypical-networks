from typing import Dict, Literal

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import lightning as L

from .sampler import EposideSampler
from .model import ProtoNet


class ProtoTrainer():
    def __init__(
        self,
        model: ProtoNet,
        samplers: Dict[Literal["train", "val", "test"], EposideSampler],
        criterion,
        optimizer,
        num_epochs: int = 1,
        # TODO add fabric args
    ):
        assert "train" in samplers

        self.samplers = samplers
        self.criterion = criterion
        self.fabric = L.Fabric(
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

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.5,
        )

        self.dataloaders = {}
        for stage, sampler in self.samplers.items():
            self.dataloaders[stage] = self.fabric.setup_dataloaders(
                DataLoader(
                    sampler.ds,
                    batch_sampler=sampler,
                    num_workers=8,
                    pin_memory=True,
                )
            )
        self.num_epochs = num_epochs


    def train(self):
        sampler = self.samplers["train"]
        dataloader = self.dataloaders["train"]

        # no need to change targets, it would be same across all batches
        targets = torch.arange(
            sampler.nc,
            dtype=torch.long,
            device=self.fabric.device,
        ).repeat_interleave(
            sampler.nq,
        )

        for epoch in range(self.num_epochs):
            self.model.train()

            self.fabric.log("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
            for i, batch in tqdm(enumerate(dataloader), total=sampler.num_episodes):
                self.optimizer.zero_grad()

                logits = self.model.forward_train(
                    batch[:sampler.num_query_per_episode, :],
                    batch[-sampler.num_support_per_episode:, :].unflatten(
                        dim=0, sizes=(sampler.ns, sampler.nc)
                    ),
                )
                loss = self.criterion(logits, targets)
                self.fabric.backward(loss)

                self.optimizer.step()
                self.fabric.log("loss", loss.item(), step=i)
            self.scheduler.step()
            self.validation(step=epoch)

    def validation(self, step: int = 0):
        self._run_single_stage("val", step=step)

    def test(self, step: int = 0):
        self._run_single_stage("test", step=step)

    @torch.no_grad()
    def _run_single_stage(self, stage: Literal["val", "test"], step: int = 0):
        if stage not in self.samplers:
            return

        self.model.eval()

        sampler = self.samplers[stage]
        dataloader = self.dataloaders[stage]

        # no need to change targets, it would be same across all batches
        targets = torch.arange(
            sampler.nc, # nc-way
            dtype=torch.long,
            device=self.fabric.device,
        ).repeat_interleave(
            sampler.nq, # nq query
        )

        acc = list()
        losses= list()
        for i, batch in tqdm(enumerate(dataloader), total=sampler.num_episodes):
            logits = self.model.forward_train(
                batch[:sampler.num_query_per_episode, :],
                batch[-sampler.num_support_per_episode:, :].unflatten(
                    dim=0, sizes=(sampler.ns, sampler.nc)
                ),
            )
            preds = logits.argmax(dim=0)
            loss = self.criterion(logits, targets)
            acc.append((preds == targets).sum().item() / sampler.num_query_per_episode)
            losses.append(loss.item())

        self.fabric.log(f"{stage}/loss", sum(losses) / len(losses), step=step)
        self.fabric.log(f"{stage}/accuracy", sum(acc) / len(acc), step=step)


    def run(self):
        self.train()
        self.test()
