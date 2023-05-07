import enum

import psutil
import torch


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


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    return psutil.virtual_memory().total
