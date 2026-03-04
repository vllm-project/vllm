# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    dist_backend: str = "gloo"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "mps"

    @classmethod
    def import_kernels(cls) -> None:
        # No vllm._C on macOS — all ops use PyTorch native fallbacks.
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # MPS uses unified memory; pinning is not applicable.
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.mps.recommended_max_memory()

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return float(torch.mps.current_allocated_memory())

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # MPS has a single device; nothing to set.
        pass

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.MPS_ATTN:
            logger.info("Cannot use %s backend on MPS.", selected_backend)
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on MPS.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on MPS.")
        return AttentionBackendEnum.MPS_ATTN.get_path()

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        # async_scheduling must be disabled before VllmConfig.__post_init__
        # runs the auto-detection logic, so we use apply_config_platform_defaults
        # (called early) rather than check_and_update_config (called late).
        vllm_config.scheduler_config.async_scheduling = False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationMode

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        # MPS is single-device only
        if parallel_config.world_size > 1:
            raise RuntimeError(
                "MPS platform does not support multi-device parallelism. "
                "world_size must be 1."
            )

        # Worker class
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.mps_worker.MPSWorker"

        # Disable features not supported on MPS
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on MPS, disabled.")
            parallel_config.enable_dbo = False

        # Block size
        if cache_config.block_size is None:
            cache_config.block_size = 16

        # FP8 KV cache not supported
        if cache_config.cache_dtype.startswith("fp8"):
            logger.warning(
                "MPS backend doesn't support KV cache quantization, "
                "falling back to auto."
            )
            cache_config.cache_dtype = "auto"

        # KV cache space — use VLLM_CPU_KVCACHE_SPACE env or auto-size.
        #
        # MPS uses unified memory shared between CPU and GPU.  When total
        # MPS-allocated memory (model weights + KV cache + intermediates)
        # exceeds ~40-45% of system RAM, the Metal memory manager begins
        # thrashing — causing 50-100x throughput degradation.
        #
        # Conservative default: 25% of system RAM for KV cache, which
        # leaves headroom for model weights (~10-15% for typical models)
        # and OS/system usage.
        import psutil

        from vllm.utils.mem_constants import GiB_bytes
        from vllm.utils.mem_utils import format_gib

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        if kv_cache_space is None:
            total_mem = psutil.virtual_memory().total
            DEFAULT_MPS_MEM_UTILIZATION = 0.25
            kv_cache_space = int(total_mem * DEFAULT_MPS_MEM_UTILIZATION)
            logger.warning_once(
                "VLLM_CPU_KVCACHE_SPACE not set. "
                "Using %s GiB for KV cache on MPS. "
                "Set VLLM_CPU_KVCACHE_SPACE (in GiB) to override.",
                format_gib(kv_cache_space),
            )
        else:
            kv_cache_space *= GiB_bytes
        cache_config.cpu_kvcache_space_bytes = kv_cache_space

        # Disable compilation / CUDA graphs
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.mode = CompilationMode.NONE

        # Disable multi-stream for shared experts
        os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

        # MPS requires spawn — fork() in a multi-threaded process crashes
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        assert vllm_config.device_config.device_type == "mps"
