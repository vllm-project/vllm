"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import List, Tuple

import pynvml

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA


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

    @staticmethod
    @with_nvml_context
    def is_full_nvlink(physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle, peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError as error:
                        logger.error(
                            "NVLink detection failed. This is normal if your"
                            " machine has no NVLink equipped.",
                            exc_info=error)
                        return False
        return True
