import os
from functools import lru_cache
from typing import Tuple

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", None) in ["fork", None]:
    logger.warning("`fork` method is not supported by ROCm. "
                   "VLLM_WORKER_MULTIPROC_METHOD is overridden to"
                   " `spawn` instead.")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM

    @staticmethod
    @lru_cache(maxsize=8)
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        return torch.cuda.get_device_capability(device_id)

    @staticmethod
    @lru_cache(maxsize=8)
    def get_device_name(device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)
