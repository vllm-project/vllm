from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

import vllm.envs as envs

logger = init_logger(__name__)


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    supported_quantization: list[str] = ["tpu_int8"]

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if envs.VLLM_USE_V1:
            if selected_backend != _Backend.PALLAS_VLLM_V1:
                logger.info("[V1] Cannot use %s backend on TPU.",
                            selected_backend)
            return _Backend.PALLAS_VLLM_V1
        else:
            if selected_backend != _Backend.PALLAS:
                logger.info("Cannot use %s backend on TPU.", selected_backend)
            return _Backend.PALLAS

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if envs.VLLM_USE_V1:
            return False
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_ONCE compilation level
        if (compilation_config.level == CompilationLevel.NO_COMPILATION
                or compilation_config.level == CompilationLevel.PIECEWISE):
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        assert compilation_config.level < CompilationLevel.PIECEWISE,\
            ("Current compilation level = {} but needs to be less"
             " than  ".format(compilation_config.level, CompilationLevel.PIECEWISE))

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm.v1.worker.tpu_worker.TRUWorker"
            else:
                if scheduler_config.is_multi_step:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_tpu_worker.MultiStepTPUWorker"
                else:
                    parallel_config.worker_cls = "vllm.worker.tpu_worker.TPUWorker"

        # Adjust scheduler config for V1
        # TODO: Add support for these
        if envs.VLLM_USE_V1:
            if vllm_config.cache_config.enable_prefix_caching:
                logger.info("[V1][TPU] Disable prefix caching")
                vllm_config.cache_config.enable_prefix_caching = False

            if vllm_config.scheduler_config.chunked_prefill_enabled:
                logger.info("[V1][TPU] Disable chunked prefill")
                vllm_config.scheduler_config.chunked_prefill_enabled = False

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False
