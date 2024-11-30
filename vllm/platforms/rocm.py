import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch

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

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

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

    @classmethod
    def get_executor_cls(cls,
                         distributed_executor_backend: Optional[str] = None,
                         is_async: Optional[bool] = None):
        if distributed_executor_backend == "ray":
            if is_async:
                return "vllm.executor.ray_gpu_executor.RayGPUExecutorAsync"
            else:
                return "vllm.executor.ray_gpu_executor.RayGPUExecutor"
        if distributed_executor_backend == "mp":
            if is_async:
                return "vllm.executor.multiproc_gpu_executor." \
                       "MultiprocessingGPUExecutorAsync"
            else:
                assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
                    "multiprocessing distributed executor backend does not "
                    "support VLLM_USE_RAY_SPMD_WORKER=1")
                return "vllm.executor.multiproc_gpu_executor." \
                       "MultiprocessingGPUExecutor"
        if is_async:
            return "vllm.executor.gpu_executor.GPUExecutorAsync"
        else:
            return "vllm.executor.gpu_executor.GPUExecutor"
