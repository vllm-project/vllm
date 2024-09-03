import os
from functools import lru_cache, wraps
from typing import List, Tuple

import torch
from amdsmi import (AmdSmiException, amdsmi_get_gpu_board_info,
                    amdsmi_get_processor_handles, amdsmi_init,
                    amdsmi_shut_down, amdsmi_topo_get_link_type)

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", None) in ["fork", None]:
    logger.warning("`fork` method is not supported by ROCm. "
                   "VLLM_WORKER_MULTIPROC_METHOD is overridden to"
                   " `spawn` instead.")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Prevent use of clashing `{CUDA/HIP}_VISIBLE_DEVICES``
if "HIP_VISIBLE_DEVICES" in os.environ:
    val = os.environ["HIP_VISIBLE_DEVICES"]
    if cuda_val := os.environ.get("CUDA_VISIBLE_DEVICES", None):
        assert val == cuda_val
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = val


# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_nvml_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM

    @staticmethod
    @lru_cache(maxsize=8)
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        return torch.cuda.get_device_capability(device_id)

    @staticmethod
    @with_nvml_context
    def is_full_nvlink(physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by xgmi (1 hop)
        """
        # On ROCm, we instead query if GPUs are connected by 1 hop XGMI
        handles = [
            amdsmi_get_processor_handles()[i] for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(
                            handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.",
                                     exc_info=error)
                        return False
        return True

    @staticmethod
    @with_nvml_context
    @lru_cache(maxsize=8)
    def get_device_name(device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        # Note: this may not be exactly the same as the torch device name
        # E.g. `AMD Instinct MI300X OAM` vs `AMD Instinct MI300X`
        return amdsmi_get_gpu_board_info(handle)["product_name"]
