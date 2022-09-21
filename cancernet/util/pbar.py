"""Define a PyTorchLightning progress bar that works with tqdm but displays only the
current epoch and does not follow the progress within each epoch."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from tqdm.auto import tqdm
from typing import Optional


class ProgressBar(ProgressBarBase):
    """A simple, tqdm-based progress bar that focuses on keeping track of how many
    epochs have been processed and how many are left, rather than monitoring progress
    within epoch.

    :param mininterval: minimum interval between progress bar updates (in seconds)
    """

    def __init__(self, mininterval: float = 0.5):
        super().__init__()

        self.mininterval = mininterval

        self._pbar: Optional[tqdm] = None
        self._enabled: bool = True

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def print(self, *args, **kwargs):
        if self._pbar is not None:
            self._pbar.write(*args, **kwargs)

    def on_train_start(self, trainer: "pl.Trainer", *_):
        total = trainer.max_epochs
        self._pbar = tqdm(
            desc=self.train_description,
            disable=not self._enabled,
            mininterval=self.mininterval,
            total=total,
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_
    ):
        if self._pbar is not None and not self._pbar.disable:
            self._pbar.update(1)
            self._pbar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, *_):
        self._pbar.close()
