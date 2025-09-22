# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from datetime import timedelta
from functools import cache, wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import torch
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import vllm._C  # noqa
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import cuda_device_count_stateless, import_pynvml

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

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
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
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
            if vllm_config.speculative_config:
                if not envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Speculative decoding is not supported on vLLM V0.")
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
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
                use_flashmla = (envs.VLLM_ATTENTION_BACKEND == "FLASHMLA")
                use_cutlass_mla = (
                    envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA")
                use_flashinfer_mla = (
                    envs.VLLM_ATTENTION_BACKEND == "FLASHINFER_MLA")

            from vllm.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size != 128:
                cache_config.block_size = 128
                logger.info("Forcing kv cache block size to 128 for "
                            "CUTLASS_MLA backend.")

            if use_flashinfer_mla and cache_config.block_size not in [32, 64]:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashInferMLA "
                    "backend.")

        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                and parallel_config.data_parallel_size > 1
                and compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "Data Parallel: disabling cudagraphs since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead.")
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_vit_attn_backend(cls, head_size: int,
                             dtype: torch.dtype) -> _Backend:
        if dtype not in (torch.float16, torch.bfloat16):
            return _Backend.XFORMERS

        if cls.has_device_capability(80):
            FLASH_ATTN_V1 = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501
            from vllm.attention.selector import is_attn_backend_supported
            is_default_fa_supported = is_attn_backend_supported(
                FLASH_ATTN_V1, head_size, dtype, allow_import_error=False)
            if is_default_fa_supported:
                return _Backend.FLASH_ATTN
            else:
                # Fallback to XFORMERS
                return _Backend.XFORMERS
        else:
            # Fallback for Volta/Turing GPUs or FA not supported
            return _Backend.XFORMERS

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla,
                             has_sink) -> str:
        if use_mla:
            if not use_v1:
                raise RuntimeError(
                    "MLA attention backends require the V1 engine. "
                    "Set VLLM_USE_V1=1 to enable them.")

            from vllm.attention.ops.flashmla import is_flashmla_supported
            from vllm.attention.utils.fa_utils import flash_attn_supports_mla

            use_cutlassmla = selected_backend == _Backend.CUTLASS_MLA or (
                selected_backend is None and cls.is_device_capability(100)
                and block_size == 128)
            use_flashinfermla = selected_backend == _Backend.FLASHINFER_MLA or (
                selected_backend is None and cls.is_device_capability(100)
                and block_size in [32, 64])
            use_flashmla = selected_backend in [
                _Backend.FLASHMLA, _Backend.FLASHMLA_VLLM_V1
            ] or (selected_backend is None and is_flashmla_supported()[0])
            use_flashattn = selected_backend == _Backend.FLASH_ATTN_MLA or (
                selected_backend is None and flash_attn_supports_mla())
            use_triton = selected_backend == _Backend.TRITON_MLA or (
                selected_backend is None)

            if use_cutlassmla:
                logger.info_once("Using Cutlass MLA backend on V1 engine.")
                return ("vllm.v1.attention.backends.mla."
                        "cutlass_mla.CutlassMLABackend")
            if use_flashinfermla:
                from vllm.v1.attention.backends.utils import (
                    set_kv_cache_layout)
                set_kv_cache_layout("HND")
                logger.info_once("Using FlashInfer MLA backend on V1 engine.")
                return ("vllm.v1.attention.backends.mla."
                        "flashinfer_mla.FlashInferMLABackend")
            if use_flashmla:
                if block_size != 64:
                    logger.warning(
                        "FlashMLA backend is not supported for block size %d"
                        " (currently only supports block size 64).",
                        block_size)
                else:
                    logger.info_once("Using FlashMLA backend on V1 engine.")
                    return ("vllm.v1.attention.backends.mla."
                            "flashmla.FlashMLABackend")
            if use_flashattn:
                logger.info_once(
                    "Using FlashAttention MLA backend on V1 engine.")
                return ("vllm.v1.attention.backends.mla."
                        "flashattn_mla.FlashAttnMLABackend")
            if use_triton:
                logger.info_once("Using Triton MLA backend on V1 engine.")
                return ("vllm.v1.attention.backends.mla."
                        "triton_mla.TritonMLABackend")
        if use_v1:
            FLASHINFER_V1 = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"  # noqa: E501
            FLEX_ATTENTION_V1 = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"  # noqa: E501
            TRITON_ATTN_VLLM_V1 = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"  # noqa: E501
            FLASH_ATTN_V1 = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501
            TREE_ATTN_V1 = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"  # noqa: E501
            XFORMERS_V1 = "vllm.v1.attention.backends.xformers.XFormersAttentionBackend"  # noqa: E501

            if selected_backend == _Backend.FLASHINFER:
                logger.info_once("Using FlashInfer backend on V1 engine.")
                if cls.has_device_capability(100):
                    from vllm.v1.attention.backends.utils import (
                        set_kv_cache_layout)
                    set_kv_cache_layout("HND")
                return FLASHINFER_V1
            elif selected_backend == _Backend.FLEX_ATTENTION:
                logger.info_once("Using FlexAttention backend on V1 engine.")
                return FLEX_ATTENTION_V1
            elif selected_backend == _Backend.TRITON_ATTN_VLLM_V1:
                logger.info_once("Using Triton backend on V1 engine.")
                return TRITON_ATTN_VLLM_V1
            elif selected_backend == _Backend.FLASH_ATTN:
                logger.info_once("Using Flash Attention backend on V1 engine.")
                return FLASH_ATTN_V1
            elif selected_backend == _Backend.TREE_ATTN:
                logger.info_once("Using Tree Attention backend on V1 engine.")
                return TREE_ATTN_V1
            elif selected_backend == _Backend.XFORMERS_VLLM_V1:
                logger.info_once("Using XFormers backend on V1 engine.")
                return XFORMERS_V1

            from vllm.attention.selector import is_attn_backend_supported

            # Default backends for V1 engine
            # Prefer FlashInfer for Blackwell GPUs if installed
            if cls.is_device_capability(100):
                if is_default_backend_supported := is_attn_backend_supported(
                        FLASHINFER_V1, head_size, dtype):
                    from vllm.v1.attention.backends.utils import (
                        set_kv_cache_layout)

                    logger.info_once(
                        "Using FlashInfer backend with HND KV cache layout on "
                        "V1 engine by default for Blackwell (SM 10.0) GPUs.")
                    set_kv_cache_layout("HND")

                    return FLASHINFER_V1

                if not is_default_backend_supported.can_import:
                    logger.warning_once(
                        "FlashInfer failed to import for V1 engine on "
                        "Blackwell (SM 10.0) GPUs; it is recommended to "
                        "install FlashInfer for better performance.")

            # FlashAttention is the default for SM 8.0+ GPUs
            if cls.has_device_capability(80):
                if has_sink and not cls.is_device_capability(90):
                    logger.info_once("Using Triton backend on V1 engine.")
                    return TRITON_ATTN_VLLM_V1
                if is_default_backend_supported := is_attn_backend_supported(
                        FLASH_ATTN_V1, head_size, dtype,
                        allow_import_error=False):
                    logger.info_once("Using Flash Attention backend on "
                                     "V1 engine.")
                    return FLASH_ATTN_V1

            # FlexAttention is the default for older GPUs
            else:
                logger.info_once("Using FlexAttention backend on V1 engine.")
                return FLEX_ATTENTION_V1

            assert not is_default_backend_supported

            use_flex_attention_reason = {}
            if not is_default_backend_supported.head_size:
                use_flex_attention_reason["head_size"] = head_size
            if not is_default_backend_supported.dtype:
                use_flex_attention_reason["dtype"] = dtype

            logger.info_once(
                "Using FlexAttention backend for %s on V1 engine.",
                ", ".join(f"{k}={v}"
                          for k, v in use_flex_attention_reason.items()),
            )
            return FLEX_ATTENTION_V1

        raise RuntimeError(
            "V0 attention backends have been removed. Set VLLM_USE_V1=1 "
            "to select a supported backend.")

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

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
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        assert is_nccl_available()
        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
        return pg

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: "ModelConfig") -> bool:
        fp8_attention = kv_cache_dtype.startswith("fp8")
        attention_backend = envs.VLLM_ATTENTION_BACKEND

        supported = False
        if model_config is not None and model_config.use_mla:
            # Default to CutlassMLA for blackwell,
            # FlashMLA otherwise
            if attention_backend is None:
                if cls.is_device_capability(100):
                    attention_backend = "CUTLASS_MLA"
                else:
                    attention_backend = "FLASHMLA"

            # Only FlashMLA and CUTLASS_MLA support fp8
            if attention_backend in [
                    "FLASHMLA", "CUTLASS_MLA", "FLASHINFER_MLA"
            ]:
                supported = True
            else:
                supported = (not fp8_attention)
        else:
            # Default to FlashAttention
            if attention_backend is None:
                attention_backend = "FLASH_ATTN_VLLM_V1"

            # All Blackwell backends support fp8
            if cls.is_device_capability(100):
                supported = True
            elif attention_backend == "FLASH_ATTN_VLLM_V1":
                if fp8_attention:
                    from vllm.attention.utils.fa_utils import (
                        flash_attn_supports_fp8)
                    supported = flash_attn_supports_fp8()
                else:
                    supported = True
            elif attention_backend == "FLASHINFER":
                supported = True
            elif attention_backend == "TRITON_ATTN_VLLM_V1":
                supported = cls.supports_fp8()
        return supported

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        if torch_dtype == torch.bfloat16:  # noqa: SIM102
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
                    "`dtype` flag in CLI, for example: --dtype=half.")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
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
        capability: Union[tuple[int, int], int],
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
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
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
                            " your machine has no NVLink equipped.")
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
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_ids)
            ]
            if (len(set(device_names)) > 1
                    and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
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
            " not found. Assuming no NVLink available.")
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
