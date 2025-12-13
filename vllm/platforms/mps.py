# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS (Metal Performance Shaders) platform support for Apple Silicon."""

import subprocess
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    dist_backend: str = "gloo"
    device_control_env_var = "GPU_DEVICE_ORDINAL"

    # MPS has limited operator support, use eager backend
    simple_compile_backend: str = "eager"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # MPS supports float16, bfloat16 (on newer Apple Silicon), and float32
        # but does NOT support float64
        if (
            subprocess.check_output(
                ["sysctl -n hw.optional.arm.FEAT_BF16"], shell=True
            ).strip()
            == b"1"
        ):
            return [torch.bfloat16, torch.float16, torch.float32]
        return [torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return (
            torch.backends.mps.get_name()
            if hasattr(torch.backends.mps, "get_name")
            else "Apple MPS"
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        if use_mla:
            raise NotImplementedError("MLA is not supported on MPS.")
        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on MPS.")
        # Use MPS attention backend
        logger.info("Using MPS attention backend.")
        return AttentionBackendEnum.MPS_ATTN.get_path()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total MPS device memory."""
        import psutil

        from vllm.utils.mem_constants import GiB_bytes

        # Check for user-specified KV cache space first
        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        if kv_cache_space is None:
            # Use a portion of system memory for MPS
            # Apple Silicon shares memory between CPU and GPU
            total_memory = psutil.virtual_memory().total
            # Use 50% of system memory for KV cache by default
            # (same as CPU backend since MPS shares unified memory)
            kv_cache_space = total_memory // 2
            logger.warning_once(
                "Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) "
                "for MPS backend is not set, using %.1f GiB by default.",
                kv_cache_space / GiB_bytes,
            )
        else:
            kv_cache_space = int(kv_cache_space * GiB_bytes)

        return kv_cache_space

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device for the current platform."""
        # MPS only has one device, but set it anyway for consistency
        # and to avoid potential side effects from not setting it
        torch.mps._set_default_device(device)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import os

        model_config = vllm_config.model_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config

        if cache_config.block_size is None:
            cache_config.block_size = 16  # Smaller block size for MPS

        if cache_config.cache_dtype != "auto":
            logger.warning(
                "MPS backend doesn't support KV cache quantization, "
                "falling back to auto."
            )
            cache_config.cache_dtype = "auto"

        cache_config.cpu_kvcache_space_bytes = MpsPlatform.get_device_total_memory()

        parallel_config = vllm_config.parallel_config

        # MPS only supports single device
        if parallel_config.world_size > 1:
            raise ValueError("MPS backend only supports single device (world_size=1)")

        # Force single process execution
        if parallel_config.distributed_executor_backend is None:
            parallel_config.distributed_executor_backend = "uni"

        if parallel_config.worker_cls == "auto":
            # Use MPS worker which handles macOS-specific initialization
            parallel_config.worker_cls = "vllm.v1.worker.mps_worker.MPSWorker"

        # Disable custom all reduce
        parallel_config.disable_custom_all_reduce = True

        # Disable DBO
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on MPS, disabled.")
            parallel_config.enable_dbo = False

        # Disable CUDA graph equivalent features
        from vllm.config import CompilationMode

        vllm_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = vllm_config.compilation_config
        # Use eager mode for MPS - torch.compile has limited support
        compilation_config.mode = CompilationMode.NONE
        compilation_config.backend = "eager"

        if vllm_config.lora_config is not None:
            compilation_config.mode = CompilationMode.NONE

        # MPS platform runs on MPS device
        assert vllm_config.device_config.device_type == "mps"

        # Environment variables for MPS executor
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Enable MPS fallback for unsupported ops
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if model_config is not None and model_config.use_mla:
            logger.info(
                "MLA is enabled on MPS platform; forcing chunked "
                "prefill and prefix caching to be disabled."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # Use CPU punica wrapper as fallback
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device specific communicator class for distributed communication."""
        # Use CPU communicator for MPS since gloo is the backend
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    @classmethod
    def supports_structured_output(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        # Return False to use direct attention backend calls
        # instead of torch.ops.vllm.unified_attention which requires CUDA ops
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return False

    @classmethod
    def import_kernels(cls) -> None:
        """Import any platform-specific C kernels."""
        # MPS doesn't have custom C kernels, skip import
        pass
