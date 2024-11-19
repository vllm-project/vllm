import os
from functools import lru_cache

import torch

from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

logger = init_logger(__name__)

if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", None) in ["fork", None]:
    logger.warning("`fork` method is not supported by ROCm. "
                   "VLLM_WORKER_MULTIPROC_METHOD is overridden to"
                   " `spawn` instead.")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        selected_backend = (_Backend.ROCM_FLASH if selected_backend
                            == _Backend.FLASH_ATTN else selected_backend)
        if selected_backend == _Backend.ROCM_FLASH:
            if not cls.has_device_capability(90):
                # not Instinct series GPUs.
                logger.info("flash_attn is not supported on NAVI GPUs.")
        else:
            logger.info("%s is not supported in AMD GPUs.", selected_backend)
        return _Backend.ROCM_FLASH

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory
