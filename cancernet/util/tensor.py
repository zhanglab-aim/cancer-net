"""Some tensor utilities."""

import torch


def scatter_nd(
    indices: torch.Tensor, weights: torch.Tensor, shape: tuple
) -> torch.Tensor:
    """For a list of indices and weights, return a sparse matrix of desired shape with
    weights only at the desired indices. Named after `tensorflow.scatter_nd`.
    """
    # Pytorch has scatter_add that does the same thing but only in one dimension
    # need to convert nd-indices to linear, then use Pytorch's function, then reshape

    ind1d = indices[:, 0]
    n = shape[0]
    for i in range(1, len(shape)):
        ind1d = ind1d * shape[i] + indices[:, i]
        n *= shape[i]

    # ensure all tensors are on the same device
    ind1d = ind1d.to(weights.device)

    # generate the flat output, then reshape
    res = weights.new_zeros(n)
    res = res.scatter_add_(0, ind1d, weights).reshape(*shape)
    return res
