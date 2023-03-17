import torch
from torch import Tensor

from torch.utils.data.sampler import Sampler
from typing import Iterator, Sized

class SequenceSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)
