"""Various utility functions for assessing fit quality."""

import torch
import numpy as np

from sklearn.metrics import roc_curve, auc
from typing import Iterable, Tuple


def get_roc(
    model: torch.nn.Module, loader: Iterable, seed: int = 1
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Run a model on the a dataset and calculate ROC and AUC.

    The model output values are exponentiated before calculating the ROC. XXX Why?

    :param model: model to test
    :param loader: data loader
    :param seed: PyTorch random seed
    :return: a tuple `(fpr, tpr, auc_value, ys, outs)`, where `(fpr, tpr)` are vectors
        representing the ROC curve; `auc_value` is the AUC; `ys` and `outs` are the
        expected (ground-truth) outputs and the (exponentiated) model outputs,
        respectively
    """
    # keep everything reproducible!
    torch.manual_seed(seed)

    outs = []
    ys = []
    device = next(iter(model.parameters())).device
    for tb in loader:
        tb = tb.to(device)
        outs.append(torch.exp(model(tb)).detach().cpu().clone().numpy())
        ys.append(tb.y.detach().cpu().clone().numpy())

    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    if len(outs.shape) > 1:
        outs = np.hstack([1-outs, outs])
    fpr, tpr, _ = roc_curve(ys, outs[:, 1])
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value, ys, outs
