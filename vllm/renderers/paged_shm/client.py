import numpy as np
import torch

from .storage import PagedSHMStorage

class PagedSHMClient:
    def __init__(self, name: str, size: int, block_size: int, pin: bool = False):
        self.storage = PagedSHMStorage(name=name, size=size, block_size=block_size, pin=pin)

    def write(self, data: bytes | np.ndarray | torch.Tensor, timeout_ms: int | None = None):
        pass

    def read(self, output: np.ndarray | torch.Tensor | None = None) -> np.ndarray | torch.Tensor:
        pass

    def read_to_device(self, ):
