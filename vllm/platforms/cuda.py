"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

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


class CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA

    @staticmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetCudaComputeCapability(handle)
