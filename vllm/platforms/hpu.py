from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        return _Backend.HPU_ATTN

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.is_multi_step:
            raise NotImplementedError(
                "Multi-step execution is not implemented for HPU")

        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Speculative decoding is not implemented for HPU")

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.worker.hpu_worker.HPUWorker"

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on HPU.")
        return False
