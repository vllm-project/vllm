from typing import TYPE_CHECKING

import psutil
import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.TORCH_SDPA:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        return _Backend.TORCH_SDPA

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm.envs as envs
        from vllm.utils import GiB_bytes
        model_config = vllm_config.model_config
        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on CPU, fallback to the eager "
                "mode.")
            model_config.enforce_eager = True

        cache_config = vllm_config.cache_config

        if cache_config.enable_prefix_caching:
            logger.warning(
                "Prefix caching is not supported on CPU, disable it.")
            cache_config.enable_prefix_caching = False

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                cache_config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                    "for CPU backend is not set, using 4 by default.")
            else:
                cache_config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore # noqa
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")

        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.chunked_prefill_enabled:
            logger.warning(
                "Chunked prefill is not supported on CPU, disable it.")
            scheduler_config.chunked_prefill_enabled = False

        parallel_config = vllm_config.parallel_config
        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "mp"):
            logger.warning(("%s is not supported on CPU, fallback to mp "
                            "distributed executor backend."),
                           parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "mp"
