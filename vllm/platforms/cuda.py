# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from collections.abc import Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, TypeVar

import torch
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import vllm._C  # noqa
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml
from vllm.utils.torch_utils import cuda_device_count_stateless

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.attention.backends.registry import _Backend
    from vllm.config import VllmConfig
else:
    _Backend = None

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

pynvml = import_pynvml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.has_device_capability(80):
            # Ampere and Hopper or later NVIDIA GPUs.
            return [torch.bfloat16, torch.float16, torch.float32]
        if self.has_device_capability(60):
            # Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported
            return [torch.float16, torch.float32]
        # Kepler and Maxwell NVIDIA GPUs, only FP32 is supported,
        # though vLLM doesn't support these GPUs.
        return [torch.float32]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `VLLM_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend for non-blackwell GPUs,
            # else we default to CutlassMLA. For each case, we force the
            # required block_size.
            use_flashmla = False
            use_cutlass_mla = False
            use_flashinfer_mla = False

            if envs.VLLM_ATTENTION_BACKEND is None:
                # Default case
                if cls.is_device_capability(100):
                    # Blackwell => Force CutlassMLA.
                    use_cutlass_mla = True
                    # TODO: This does not work, because the
                    # global_force_attn_backend_context_manager is not set.
                    # See vllm/attention/selector.py:_cached_get_attn_backend
                    envs.VLLM_ATTENTION_BACKEND = "CUTLASS_MLA"
                else:
                    # Not Blackwell
                    use_flashmla = True
            else:
                # Forced case
                use_flashmla = envs.VLLM_ATTENTION_BACKEND == "FLASHMLA"
                use_cutlass_mla = envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA"
                use_flashinfer_mla = envs.VLLM_ATTENTION_BACKEND == "FLASHINFER_MLA"

            from vllm.attention.ops.flashmla import is_flashmla_dense_supported

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size % 128 != 0:
                cache_config.block_size = 128
                logger.info(
                    "Forcing kv cache block size to 128 for CUTLASS_MLA backend."
                )

            if (
                use_flashinfer_mla
                and cache_config.block_size != 32
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashInferMLA backend."
                )

            # TODO(Chen): remove this hacky code
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse backend."
                )
        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if (
            parallel_config.all2all_backend == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
            and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "WideEP: Disabling CUDA Graphs since DeepEP high-throughput "
                "kernels are optimized for prefill and are incompatible with "
                "CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "use --all2all-backend with another option, such as "
                "deepep_low_latency, pplx, or allgather_reducescatter."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_vit_attn_backend(cls, head_size: int, dtype: torch.dtype) -> "_Backend":
        from vllm.attention.backends.registry import _Backend

        # For Blackwell GPUs, force TORCH_SDPA for now.
        # See https://github.com/facebookresearch/xformers/issues/1317#issuecomment-3199392579 # noqa: E501
        if cls.has_device_capability(100):
            return _Backend.TORCH_SDPA

        if dtype not in (torch.float16, torch.bfloat16):
            return _Backend.XFORMERS

        if cls.has_device_capability(80):
            FLASH_ATTN_V1 = (
                "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501
            )
            from vllm.attention.selector import is_attn_backend_supported

            is_default_fa_supported = is_attn_backend_supported(
                FLASH_ATTN_V1, head_size, dtype, allow_import_error=False
            )
            if is_default_fa_supported:
                return _Backend.FLASH_ATTN
            else:
                # Fallback to XFORMERS
                return _Backend.XFORMERS
        else:
            # Fallback for Volta/Turing GPUs or FA not supported
            return _Backend.XFORMERS

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_v1,
        use_mla,
        has_sink,
        use_sparse,
    ) -> str:
        from vllm.attention.backends.registry import _Backend

        if use_mla:
            # explicitly reject non-MLA backends when MLA is enabled to avoid
            # silently selecting an incompatible backend (e.g., FLASHINFER).
            if selected_backend in {
                _Backend.FLASHINFER,
                _Backend.FLASH_ATTN,
                _Backend.TRITON_ATTN,
                _Backend.TREE_ATTN,
                _Backend.XFORMERS,
            }:
                raise ValueError(
                    f"Attention backend {selected_backend} incompatible with MLA. "
                    "Please use one of the MLA backends: FLASHINFER_MLA, CUTLASS_MLA, "
                    "FLASHMLA, FLASH_ATTN_MLA, or TRITON_MLA. Alternatively, set "
                    "VLLM_MLA_DISABLE=1 to disable MLA for this model."
                )

            from vllm.attention.ops.flashmla import is_flashmla_dense_supported
            from vllm.attention.utils.fa_utils import flash_attn_supports_mla

            if use_sparse:
                logger.info_once("Using Sparse MLA backend.")
                return (
                    "vllm.v1.attention.backends.mla.flashmla_sparse."
                    "FlashMLASparseBackend"
                )

            use_cutlassmla = selected_backend == _Backend.CUTLASS_MLA or (
                selected_backend is None
                and cls.is_device_capability(100)
                and block_size % 128 == 0
            )
            use_flashinfermla = selected_backend == _Backend.FLASHINFER_MLA or (
                selected_backend is None
                and cls.is_device_capability(100)
                and (block_size == 32 or block_size % 64 == 0)
            )
            use_flashmla = selected_backend == _Backend.FLASHMLA or (
                selected_backend is None and is_flashmla_dense_supported()[0]
            )
            use_flashattn = selected_backend == _Backend.FLASH_ATTN_MLA or (
                selected_backend is None and flash_attn_supports_mla()
            )
            use_triton = selected_backend == _Backend.TRITON_MLA or (
                selected_backend is None
            )

            if use_cutlassmla:
                logger.info_once("Using Cutlass MLA backend.", scope="local")
                return "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
            if use_flashinfermla:
                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                set_kv_cache_layout("HND")
                logger.info_once("Using FlashInfer MLA backend.")
                return (
                    "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
                )
            if use_flashmla:
                if block_size % 64 != 0:
                    logger.warning(
                        "FlashMLA backend is not supported for block size %d"
                        " (currently only supports block size 64).",
                        block_size,
                    )
                else:
                    logger.info_once("Using FlashMLA backend.")
                    return "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
            if use_flashattn:
                logger.info_once("Using FlashAttention MLA backend.")
                return (
                    "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
                )
            if use_triton:
                logger.info_once("Using Triton MLA backend.")
                return "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"

        FLASHINFER_V1 = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"  # noqa: E501
        FLEX_ATTENTION_V1 = (
            "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"  # noqa: E501
        )
        TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"  # noqa: E501
        FLASH_ATTN_V1 = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501
        TREE_ATTN_V1 = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"  # noqa: E501
        XFORMERS_V1 = "vllm.v1.attention.backends.xformers.XFormersAttentionBackend"  # noqa: E501

        use_fp8_kv_cache = kv_cache_dtype is not None and kv_cache_dtype.startswith(
            "fp8"
        )

        if selected_backend == _Backend.FLASHINFER:
            logger.info_once("Using FlashInfer backend.")
            if cls.has_device_capability(100):
                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                set_kv_cache_layout("HND")
            return FLASHINFER_V1
        elif selected_backend == _Backend.FLEX_ATTENTION:
            logger.info_once("Using FlexAttention backend.")
            return FLEX_ATTENTION_V1
        elif selected_backend == _Backend.TRITON_ATTN:
            logger.info_once("Using Triton backend.")
            return TRITON_ATTN
        elif selected_backend == _Backend.FLASH_ATTN:
            logger.info_once("Using Flash Attention backend.")
            return FLASH_ATTN_V1
        elif selected_backend == _Backend.TREE_ATTN:
            logger.info_once("Using Tree Attention backend.")
            return TREE_ATTN_V1
        elif selected_backend == _Backend.XFORMERS:
            logger.info_once("Using XFormers backend.")
            return XFORMERS_V1

        from vllm.attention.selector import is_attn_backend_supported

        # Default backends for V1 engine
        # Prefer FlashInfer for Blackwell GPUs if installed
        if cls.is_device_capability(100):
            if is_default_backend_supported := is_attn_backend_supported(
                FLASHINFER_V1, head_size, dtype
            ):
                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                logger.info_once(
                    "Using FlashInfer backend with HND KV cache layout on "
                    "V1 engine by default for Blackwell (SM 10.0) GPUs."
                )
                set_kv_cache_layout("HND")

                return FLASHINFER_V1

            if not is_default_backend_supported.can_import:
                logger.warning_once(
                    "FlashInfer failed to import on Blackwell (SM 10.0) GPUs; "
                    "it is recommended to install FlashInfer for better "
                    "performance."
                )

        # FlashAttention is the default for SM 8.0+ GPUs
        if cls.has_device_capability(80):
            if (has_sink or use_fp8_kv_cache) and not cls.is_device_capability(90):
                logger.info_once("Using Triton backend.")
                return TRITON_ATTN
            elif is_default_backend_supported := is_attn_backend_supported(
                FLASH_ATTN_V1, head_size, dtype, allow_import_error=False
            ):
                logger.info_once("Using Flash Attention backend.")
                return FLASH_ATTN_V1

        # FlexAttention is the default for older GPUs
        else:
            logger.info_once("Using FlexAttention backend.")
            return FLEX_ATTENTION_V1

        assert not is_default_backend_supported

        use_flex_attention_reason = {}
        if not is_default_backend_supported.head_size:
            use_flex_attention_reason["head_size"] = head_size
        if not is_default_backend_supported.dtype:
            use_flex_attention_reason["dtype"] = dtype

        logger.info_once(
            "Using FlexAttention backend for %s.",
            ", ".join(f"{k}={v}" for k, v in use_flex_attention_reason.items()),
        )
        return FLEX_ATTENTION_V1

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
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
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available."
        )
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

CudaPlatform.log_warnings()
