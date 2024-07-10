"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import Tuple

import pynvml

from .interface import Platform, PlatformEnum


def with_nvml_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


@lru_cache(maxsize=8)
@with_nvml_context
def get_physical_device_capability(device_id: int = 0) -> Tuple[int, int]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return pynvml.nvmlDeviceGetCudaComputeCapability(handle)


def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return get_physical_device_capability(physical_device_id)
