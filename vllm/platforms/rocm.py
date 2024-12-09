import os
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, List

import torch
from amdsmi import (AmdSmiException, amdsmi_get_gpu_board_info,
                    amdsmi_get_processor_handles, amdsmi_init,
                    amdsmi_shut_down, amdsmi_topo_get_link_type)

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)

# import custom ops, trigger op registration
try:
    import vllm._rocm_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._rocm_C with %r", e)

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


def with_amdsmi_context(fn):

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
    device_name: str = "rocm"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    supported_quantization: list[str] = [
        "awq", "gptq", "fp8", "compressed_tensors", "compressed-tensors",
        "fbgemm_fp8", "gguf"
    ]

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

    @staticmethod
    @with_amdsmi_context
    def is_full_nvlink(physical_device_ids: List[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
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

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        # Note: this may not be exactly the same as the torch device name
        # E.g. `AMD Instinct MI300X OAM` vs `AMD Instinct MI300X`
        return amdsmi_get_gpu_board_info(handle)["product_name"]

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = \
                    "vllm.worker.multi_step_worker.MultiStepWorker"
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.worker.Worker"
            else:
                parallel_config.worker_cls = "vllm.worker.worker.Worker"

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ.")
        envs.VLLM_USE_TRITON_AWQ = True
