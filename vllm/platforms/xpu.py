# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
from typing import TYPE_CHECKING

import torch

# import custom ops, trigger op registration
import vllm_xpu_kernels._C  # noqa
import vllm_xpu_kernels._moe_C  # noqa
import vllm_xpu_kernels._xpu_C  # noqa

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig
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
    dist_backend: str = "xccl"  # xccl only
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
        num_heads: int | None = None,
    ) -> str:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout("NHD")
        logger.info(
            "Setting VLLM_KV_CACHE_LAYOUT to 'NHD' for XPU; "
            "only NHD layout is supported by XPU attention kernels."
        )

        # TurboQuant KV cache: route directly to TQ backend
        kv_cache_dtype = attn_selector_config.kv_cache_dtype
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):
            logger.info_once("Using TurboQuant attention backend.")
            return AttentionBackendEnum.TURBOQUANT.get_path()

        dtype = attn_selector_config.dtype
        if attn_selector_config.use_sparse:
            logger.info_once("Using XPU MLA Sparse backend.")
            return AttentionBackendEnum.XPU_MLA_SPARSE.get_path()
        if attn_selector_config.use_mla:
            logger.info_once("Using Triton MLA backend on V1 engine.")
            return AttentionBackendEnum.TRITON_MLA.get_path()
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
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
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
    def manual_seed_all(cls, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

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
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config

        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []

        attention_config = vllm_config.attention_config
        if attention_config.backend is None:
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN
        if not supports_xpu_graph():
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning(
                "XPU Graph is not supported in the current PyTorch version, "
                "disabling cudagraph_mode."
            )
        elif not envs.VLLM_XPU_ENABLE_XPU_GRAPH:
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning(
                "XPU Graph is disabled by environment variable, "
                "please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it."
            )
        elif parallel_config.world_size_across_dp > 1:
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning(
                "XPU Graph doesn't support capture communication ops, "
                "disabling cudagraph_mode."
            )
        else:
            if (
                attention_config.backend == AttentionBackendEnum.FLASH_ATTN
                and compilation_config.cudagraph_mode
                not in {CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE}
            ):
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                logger.warning(
                    "FMHA sycl-tla kernels cannot be captured with XPU graphs, "
                    "falling back to PIECEWISE graph mode on XPU platform."
                )

        # Disable fusion passes not yet supported on XPU.
        pass_config = compilation_config.pass_config
        fusion_passes_to_disable = {
            "enable_sp": "Sequence parallelism",
            "fuse_gemm_comms": "Async TP",
            "fuse_allreduce_rms": "AllReduce + RMSNorm fusion",
            "fuse_norm_quant": "RMSNorm + quant fusion",
            "fuse_act_quant": "Activation + quant fusion",
            "fuse_attn_quant": "Attention + quant fusion",
            "fuse_act_padding": "Activation + padding fusion",
            "fuse_rope_kvcache": "RoPE + KV cache fusion",
        }
        for flag, feature_name in fusion_passes_to_disable.items():
            if getattr(pass_config, flag):
                logger.warning(
                    "Feature %r is not yet supported on XPU and will be disabled.",
                    feature_name,
                )
                setattr(pass_config, flag, False)

        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        # Only override worker_cls if it's still the default "auto"
        # This allows custom workers (like vllm-omni workers) to be used on XPU
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"
        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.enable_permute_local_kv = True

        # In some cases, the internal memory type cache can misdetect GPU
        # memory as host memory, also leading to invalid memory access.
        # This cache can be disabled by setting UCX_MEMTYPE_CACHE=n.
        # ref. https://openucx.readthedocs.io/en/master/faq.html
        os.environ["UCX_MEMTYPE_CACHE"] = "n"

        # spawn is the only supported multiprocessing method on XPU
        if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        super().update_block_size_for_backend(vllm_config)
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.utils.math_utils import cdiv

        cache_config = vllm_config.cache_config
        # special fix for GDN since kernel only supports block size dividable by 64
        attn_layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        kernel_block_size = None
        for layer in attn_layers.values():
            b = layer.get_attn_backend()
            if b.get_name() == "GDN_ATTN":
                kernel_block_size = 64
                break

        if kernel_block_size is None:
            return
        new_block_size = (
            cdiv(cache_config.block_size, kernel_block_size) * kernel_block_size
        )
        if new_block_size == cache_config.block_size:
            return

        if cache_config.mamba_cache_mode == "align":
            cache_config.mamba_block_size = new_block_size
        original_mamba_page_size_padded = cache_config.mamba_page_size_padded
        if cache_config.mamba_page_size_padded is not None:
            attn_page_size_1_token = (
                cache_config.mamba_page_size_padded // cache_config.block_size
            )
            cache_config.mamba_page_size_padded = (
                new_block_size * attn_page_size_1_token
            )
        cache_config.block_size = new_block_size
        logger.info(
            "[XPU]Setting attention block size to %d tokens to ensure multiple of %d, "
            "set mamba_page_size_padded to %d bytes accordingly, before was %d bytes.",
            new_block_size,
            kernel_block_size,
            cache_config.mamba_page_size_padded,
            original_mamba_page_size_padded,
        )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats(device)
        return torch.xpu.max_memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def is_data_center_gpu(cls) -> bool:
        device_name = cls.get_device_name().lower()
        return device_name.count("data center gpu") > 0

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        from vllm.utils.torch_utils import supports_xccl

        if not supports_xccl():
            logger.warning(
                "xccl is not enabled in this torch build, communication"
                " is not available."
            )
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return True

    @classmethod
    def get_default_ir_op_priority(
        cls, vllm_config: "VllmConfig"
    ) -> "IrOpPriorityConfig":
        from vllm.config.compilation import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        # Native used by default when compiling,
        # use fused kernels where available when no codegen
        cc = vllm_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        default = ["native"] if using_inductor else ["xpu_kernels", "native"]

        return IrOpPriorityConfig.with_default(default)

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

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.xpu.get_device_properties(device_id).max_compute_units
