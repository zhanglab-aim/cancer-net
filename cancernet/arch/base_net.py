"""Scaffolding for building PyTorch Lightning modules."""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Tuple, List


class BaseNet(pl.LightningModule):
    """A basic scaffold for our modules, with default optimizer, scheduler, and loss
    function, and simple logging.
    """

    def __init__(self, lr: float = 0.01, scheduler: str="lambda"):
        super().__init__()

        self.lr = lr
        self.scheduler=scheduler

    def configure_optimizers(self) -> Tuple[List, List]:
        """Set up optimizers and schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler=="lambda":
            lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.scheduler=="pnet": ## Take scheduler from pnet
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.25)
        else:
            scheduler=None

        return [optimizer], [scheduler]

    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        and accuracy information that will be aggregated at epoch end.

        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss
        y_hat = self(batch)

        loss = F.nll_loss(y_hat, batch.y)

        # assess accuracy
        pred = y_hat.max(1)[1]
        correct = pred.eq(batch.y).sum().item()

        total = len(batch.y)

        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        """Generic function for summarizing and logging the loss and accuracy over an
        epoch.

        Creates log entries with name `f"{kind}_loss"` and `f"{kind}_accuracy"`.

        This function is used to implement the training, validation, and test epoch-end
        functions.
        """
        with torch.no_grad():
            # calculate average loss and average accuracy
            total_loss = sum(_["loss"] * _["total"] for _ in outputs)
            total = sum(_["total"] for _ in outputs)
            avg_loss = total_loss / total

            correct = sum(_["correct"] for _ in outputs)
            avg_acc = correct / total

        # log
        self.log(f"{kind}_loss", avg_loss)
        self.log(f"{kind}_accuracy", avg_acc)

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")
