import enum
import random
import psutil

import numpy as np
import torch

from cacheflow.parallel_utils.parallel_state import model_parallel_is_initialized
from cacheflow.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        id = self.counter
        self.counter += 1
        return id

    def reset(self) -> None:
        self.counter = 0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    return psutil.virtual_memory().total
