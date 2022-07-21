import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GlobalAttention

from typing import Sequence


class GATNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        lr: float = 0.01,
    ):
        assert len(dims) == 4

        super().__init__()

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.fc1 = nn.Linear(dims[2], dims[3])
        self.fc2 = nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)

        self.gate_nn = nn.Sequential(
            nn.Linear(dims[2], 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

        self.lr = lr

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        # x2 = global_mean_pool(x1, data.batch)
        x2 = self.pool(x1, data.batch)
        x = F.dropout(x2, p=0.1, training=self.training)

        # back to 128-dimensions, then down to the number of classes
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return some intermediate results
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def step(self, batch, kind: str) -> dict:
        # run the model and calculate loss
        y_hat = self(batch)

        loss = F.nll_loss(y_hat, batch.y)

        # assess accuracy
        pred = y_hat.max(1)[1]
        correct = pred.eq(batch.y).sum().item()

        total = len(batch)

        # things to log
        log = {f"{kind}_loss": loss}

        batch_dict = {
            "loss": loss,
            "log": log,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        # calculate average loss and average accuracy
        avg_loss = torch.stack([_["loss"] for _ in outputs]).mean()

        correct = sum(_["correct"] for _ in outputs)
        total = sum(_["total"] for _ in outputs)
        avg_acc = correct / total

        # log
        self.log(f"{kind}_avg_loss", avg_loss)
        self.log(f"{kind}_avg_accuracy", avg_acc)

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")
