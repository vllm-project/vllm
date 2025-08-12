# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "compressed_tensors"
    ]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if (selected_backend != _Backend.PALLAS
                and selected_backend != _Backend.PALLAS_VLLM_V1):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        if use_v1:
            logger.info("Using Pallas V1 backend.")
            return "vllm.v1.attention.backends.pallas.PallasAttentionBackend"
        else:
            logger.info("Using Pallas backend.")
            return "vllm.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not envs.VLLM_USE_V1

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
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        if vllm_config.model_config.dtype in (torch.float16, torch.float32):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.", vllm_config.model_config.dtype)
            vllm_config.model_config.dtype = torch.bfloat16

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on vLLM V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_tpu_worker.MultiStepTPUWorker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.tpu_worker.TPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.tpu_worker.TPUWorker"

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True
