"""Various utility functions for assessing fit quality."""

import torch
import numpy as np

from sklearn.metrics import roc_curve, auc
from typing import Iterable, Tuple, Optional


def get_roc(
    model: torch.nn.Module, loader: Iterable, seed: Optional[int] = 1, exp: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Run a model on the a dataset and calculate ROC and AUC.

    The model output values are by default exponentiated before calculating the ROC.
    XXX Why?

    When the model outputs a list of values instead of a tensor, the last element in the
    sequence is used by this function.

    :param model: model to test
    :param loader: data loader
    :param seed: PyTorch random seed; set to `None` to avoid setting the seed
    :param exp: if `True`, exponential model outputs before calculating ROC
    :return: a tuple `(fpr, tpr, auc_value, ys, outs)`, where `(fpr, tpr)` are vectors
        representing the ROC curve; `auc_value` is the AUC; `ys` and `outs` are the
        expected (ground-truth) outputs and the (exponentiated) model outputs,
        respectively
    """
    if seed is not None:
        # keep everything reproducible!
        torch.manual_seed(seed)

    # make sure the model is in evaluation mode
    model.eval()

    outs = []
    ys = []
    device = next(iter(model.parameters())).device
    for tb in loader:
        tb = tb.to(device)
        output = model(tb)

        # handle multiple outputs
        if not torch.is_tensor(output):
            assert hasattr(output, "__getitem__")
            output = output[-1]

        output = output.detach().cpu().numpy()
        if exp:
            output = np.exp(output)
        outs.append(output)

        ys.append(tb.y.detach().cpu().numpy())

    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    if len(outs.shape) == 1 or outs.shape[0] == 1 or outs.shape[1] == 1:
        outs = np.column_stack([1 - outs, outs])
    fpr, tpr, _ = roc_curve(ys, outs[:, 1])
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value, ys, outs
