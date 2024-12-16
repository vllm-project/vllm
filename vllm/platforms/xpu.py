from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.IPEX:
            logger.info("Cannot use %s backend on XPU.", selected_backend)
        return _Backend.IPEX

    @staticmethod
    def get_device_capability(device_id: int = 0) -> DeviceCapability:
        major, minor, *_ = torch.xpu.get_device_capability(
            device_id)['version'].split('.')
        return DeviceCapability(major=int(major), minor=int(minor))

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # check and update model config
        model_config = vllm_config.model_config
        if model_config.dtype == torch.bfloat16:
            logger.warning(
                "bfloat16 is not fully supported on XPU, casting to float16.")
            model_config.dtype = torch.float16
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on XPU, fallback to the eager "
                "mode.")
            model_config.enforce_eager = True

        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "XPU does not support speculative decoding")

        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "ray"):
            logger.warning(
                "%s is not supported on XPU, fallback to ray distributed"
                " executor backend.",
                parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "ray"
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.worker.xpu_worker.XPUWorker"

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on XPU.")
        return False
