from functools import lru_cache
from typing import Tuple

import torch

from .interface import Platform, PlatformEnum


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM

    @staticmethod
    @lru_cache(maxsize=8)
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        return torch.cuda.get_device_capability(device_id)
