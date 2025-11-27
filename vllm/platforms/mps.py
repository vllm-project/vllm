# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.attention.backends.registry import _Backend
    from vllm.config import VllmConfig
else:
    _Backend = None
    VllmConfig = None


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    dist_backend: str = "gloo"  # Use gloo like CPU since MPS doesn't have NCCL equivalent
    device_control_env_var = "MPS_DEVICE_ID"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # MPS supports fp16, bf16, and fp32
        # Check if bfloat16 is supported on this Apple Silicon chip
        try:
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.arm.FEAT_BF16"],
                stderr=subprocess.DEVNULL
            ).strip()
            if result == b"1":
                return [torch.bfloat16, torch.float16, torch.float32]
        except Exception:
            pass
        return [torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Apple Silicon chip."""
        try:
            # Get the chip model
            result = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return result
        except Exception:
            return "Apple Silicon"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "_Backend",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        from vllm.attention.backends.registry import _Backend

        # Check if Metal backend can be used
        try:
            import vllm._metal_C  # noqa: F401
            use_metal = True
        except ImportError:
            logger.warning("Metal C extension not available, falling back to Torch SDPA")
            use_metal = False

        if use_metal:
            if selected_backend and selected_backend != _Backend.METAL:
                logger.info("Cannot use %s backend on MPS, using Metal backend instead.", selected_backend)
            if use_mla:
                raise NotImplementedError("MLA is not supported on Metal.")
            if use_sparse:
                raise NotImplementedError("Sparse Attention is not supported on Metal.")

            # Metal backend only supports float16 and float32, not bfloat16
            if dtype == torch.bfloat16 or (kv_cache_dtype and "bfloat16" in kv_cache_dtype):
                raise ValueError(
                    "Metal backend does not support bfloat16. "
                    "Please use --dtype float16 (or --kv-cache-dtype float16) when using MPS/Metal backend."
                )

            logger.info("Using Metal native backend for MPS with paged attention.")
            return "vllm.v1.attention.backends.metal_attn.MetalAttentionBackend"
        else:
            # Fallback to TorchSDPA
            if selected_backend and selected_backend != _Backend.TORCH_SDPA:
                logger.info("Cannot use %s backend on MPS, using Torch SDPA instead.", selected_backend)
            if use_mla:
                raise NotImplementedError("MLA is not supported on MPS.")
            if use_sparse:
                raise NotImplementedError("Sparse Attention is not supported on MPS.")

            logger.info("Using Torch SDPA backend for MPS.")
            return "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total unified memory available on Apple Silicon."""
        try:
            # Get the total physical memory in bytes
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            total_memory = int(result)

            # Reserve some memory for the system (20%)
            # Apple Silicon has unified memory shared between CPU and GPU
            usable_memory = int(total_memory * 0.8)

            logger.info(
                "Detected %d GB of unified memory, reserving 20%% for system, "
                "using %d GB for vLLM.",
                total_memory // (1024 ** 3),
                usable_memory // (1024 ** 3)
            )

            return usable_memory
        except Exception as e:
            logger.warning("Failed to get system memory: %s, using default 8GB", str(e))
            return 8 * 1024 ** 3  # Default to 8GB

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device for the current platform."""
        # MPS doesn't have a set_device method like CUDA
        # Just ensure the device is MPS
        if device.type != "mps":
            logger.warning("Attempting to set non-MPS device %s on MPS platform", device)

    @classmethod
    def inference_mode(cls):
        """Use torch.no_grad() as MPS may not support inference_mode fully."""
        # Use no_grad() as it's more compatible with MPS
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        """Check and update configuration for MPS platform."""
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        # Set block size for MPS (similar to CPU)
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # Check for unsupported features
        if scheduler_config.enable_chunked_prefill and cache_config.cache_dtype != "auto":
            logger.warning(
                "Chunked-prefill on MPS is not compatible with FP8 KV cache, "
                "using auto cache dtype."
            )
            cache_config.cache_dtype = "auto"

        if cache_config.enable_prefix_caching:
            logger.warning(
                "Prefix caching may have limited support on MPS backend."
            )

        # MPS doesn't support distributed execution
        if parallel_config.world_size > 1:
            raise RuntimeError(
                "MPS backend does not support distributed execution. "
                "Apple Silicon GPUs do not support multi-GPU setups."
            )

        # Set worker class for MPS
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.mps_worker.MPSWorker"

        # Disable DBO (Dual-Batch Overlap)
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on MPS, disabled.")
            parallel_config.enable_dbo = False

        # Handle compilation config
        from vllm.config import CompilationMode

        vllm_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = vllm_config.compilation_config
        # Disable all compilation for MPS since torch.compile/inductor
        # doesn't fully support MPS yet
        compilation_config.mode = CompilationMode.NONE
        compilation_config.backend = ""
        logger.info("Disabling compilation for MPS (torch.compile not fully supported).")

        if vllm_config.lora_config is not None:
            logger.warning("LoRA may have limited support on MPS backend.")

        assert vllm_config.device_config.device_type == "mps"

        # Set environment variables
        import os
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Pinning memory is not applicable for MPS (unified memory)."""
        logger.info("Pin memory is not applicable on MPS (unified memory architecture).")
        return False

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Get current memory usage on MPS device."""
        try:
            # MPS uses unified memory, so check system memory usage
            # PyTorch doesn't provide direct MPS memory stats yet
            if hasattr(torch.mps, 'current_allocated_memory'):
                return float(torch.mps.current_allocated_memory())
            else:
                # Fallback: estimate from system memory
                logger.debug("MPS memory tracking not available, using estimated value.")
                return 0.0
        except Exception as e:
            logger.debug("Failed to get MPS memory usage: %s", str(e))
            return 0.0

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """Get punica wrapper for MPS (use CPU implementation)."""
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device communicator class for MPS."""
        # Use CPU communicator since MPS doesn't support multi-GPU
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    @classmethod
    def opaque_attention_op(cls) -> bool:
        """Register attention as one giant opaque custom op."""
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Check if hybrid KV cache is supported."""
        return False  # Disable for now, can enable later if needed
