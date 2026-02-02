# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    # Intel XPU's device key is "GPU" for Ray.
    # see https://github.com/ray-project/ray/blob/6a5eb5865eeb9ccf058a79b44f107e327e360673/python/ray/_private/accelerators/intel_gpu.py#L20 # noqa: E501
    ray_device_key: str = "GPU"
    dist_backend: str = "ccl"  # ccl | xccl
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C
        with contextlib.suppress(ImportError):
            import vllm._moe_C  # noqa: F401

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout("NHD")
        logger.info(
            "Setting VLLM_KV_CACHE_LAYOUT to 'NHD' for XPU; "
            "only NHD layout is supported by XPU attention kernels."
        )

        dtype = attn_selector_config.dtype
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on XPU.")
        if selected_backend == AttentionBackendEnum.TRITON_ATTN:
            logger.info_once("Using Triton backend.")
            return AttentionBackendEnum.TRITON_ATTN.get_path()
        elif dtype == torch.float32:
            logger.warning_once(
                "Flash Attention on XPU does not support float32 dtype. "
                "Falling back to Triton Attention backend."
            )
            return AttentionBackendEnum.TRITON_ATTN.get_path()
        elif selected_backend == AttentionBackendEnum.FLASH_ATTN:
            logger.info_once("Using Flash Attention backend.")
            return AttentionBackendEnum.FLASH_ATTN.get_path()
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}, "
                f"with use_mla: {attn_selector_config.use_mla}"
            )

        logger.info("Using Flash Attention backend.")
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        # XPU only supports FLASH_ATTN for vision attention.
        return [
            AttentionBackendEnum.FLASH_ATTN,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: "
                f"{cls.get_supported_vit_attn_backends()}."
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        logger.info_once(
            f"Using backend {AttentionBackendEnum.FLASH_ATTN} for vit attention"
        )
        return AttentionBackendEnum.FLASH_ATTN

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.xpu.set_device(device)

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        # capacity format differs from cuda's and will cause unexpected
        # failure, so use None directly
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        xpu_use_triton_kernel = os.getenv("XPU_USE_TRITON_KERNEL", "0") == "1"
        if not xpu_use_triton_kernel:
            return "vllm.lora.punica_wrapper.punica_xpu.PunicaWrapperXPU"
        else:
            return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        # in V1(or with ipex chunked prefill) block_size is 64
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 64

        # lazy import to avoid circular import
        from vllm.config import CompilationMode, CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []

        assert compilation_config.cudagraph_mode == CUDAGraphMode.NONE, (
            "CUDA graph mode should be NONE on XPU"
        )

        if vllm_config.lora_config is not None:
            compilation_config.mode = CompilationMode.NONE
        # decrease triton kernel compilation scratch space for speculative decoding
        if vllm_config.speculative_config is not None:
            os.environ["IGC_ForceOCLSIMDWidth"] = "16"  # noqa: SIM112
        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        # Only override worker_cls if it's still the default "auto"
        # This allows custom workers (like vllm-omni workers) to be used on XPU
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"
        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.enable_permute_local_kv = True

        if model_config and model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return False

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.xpu.reset_peak_memory_stats(device)
        return torch.xpu.max_memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e5m2

    @classmethod
    def is_data_center_gpu(cls) -> bool:
        device_name = cls.get_device_name().lower()
        return device_name.count("data center gpu") > 0

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"  # noqa

    @classmethod
    def device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            device_name = cls.get_device_name().lower()
            # client gpu a770
            if device_name.count("a770") > 0:
                raise ValueError(
                    "Intel Arc A770 have bfloat16 accuracy known issue. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on XPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from XPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()
