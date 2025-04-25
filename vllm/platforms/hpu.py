# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional

import torch

from vllm import envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"
    ray_device_key: str = "HPU"
    device_control_env_var: str = "HABANA_VISIBLE_MODULES"
    supported_quantization: list[str] = [
        "compressed-tensors", "fp8", "inc", "awq_hpu", "gptq_hpu"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if use_v1:
            logger.info("Using HPUAttentionV1 backend.")
            return "vllm.v1.attention.backends.hpu_attn.HPUAttentionBackendV1"
        logger.info("Using HPUAttention backend.")
        return "vllm.attention.backends.hpu_attn.HPUAttentionBackend"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        if scheduler_config.is_multi_step:
            parallel_config.worker_cls = \
                "vllm.worker.multi_step_hpu_worker.MultiStepHPUWorker"

        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm.v1.worker.hpu_worker.HPUWorker"
            else:
                if scheduler_config.is_multi_step:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_hpu_worker.MultiStepHPUWorker"
                elif vllm_config.speculative_config:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.hpu_worker.HPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.hpu_worker.HPUWorker"

        # NOTE(kzawora): default block size for Gaudi should be 128
        # smaller sizes still work, but very inefficiently
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
        if (parallel_config.distributed_executor_backend == 'mp'
                and envs.VLLM_WORKER_MULTIPROC_METHOD == 'fork'):
            if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD",
                              None) is not None:
                logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                               "might cause application hangs on exit. Using "
                               "VLLM_WORKER_MULTIPROC_METHOD=fork anyway, "
                               "as it was explicitly requested.")
            else:
                logger.warning(
                    "On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                    "might cause application hangs on exit. Setting "
                    "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                    "To override that behavior, please set "
                    "VLLM_WORKER_MULTIPROC_METHOD=fork explicitly.")
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on HPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_hpu.PunicaWrapperHPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.hpu_communicator.HpuCommunicator"  # noqa

    @classmethod
    def supports_structured_output(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on HPU is experimental
        return True
