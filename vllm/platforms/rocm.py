# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import cuda_device_count_stateless

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config import VllmConfig
else:
    AttentionBackendEnum = None

logger = init_logger(__name__)

try:
    from amdsmi import (
        AmdSmiException,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
        amdsmi_topo_get_link_type,
    )
except ImportError as e:
    logger.warning("Failed to import from amdsmi with %r", e)

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)

# import custom ops, trigger op registration
try:
    import vllm._rocm_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._rocm_C with %r", e)

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: list[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_SWA_REASON = ()
_ROCM_PARTIALLY_SUPPORTED_MODELS: dict[str, str] = {}
_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",  # MI300X VF
    "0x74a2": "AMD_Instinct_MI308X",
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",  # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",
    "0x74bd": "AMD_Instinct_MI300X_HF",
    "0x744c": "AMD_Radeon_RX7900XTX",
}

# Prevent use of clashing `{CUDA/HIP}_VISIBLE_DEVICES`
if "HIP_VISIBLE_DEVICES" in os.environ:
    val = os.environ["HIP_VISIBLE_DEVICES"]
    if cuda_val := os.environ.get("CUDA_VISIBLE_DEVICES", None):
        assert val == cuda_val
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = val

# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


@cache
def on_gfx1x() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx11", "gfx12"])


@cache
def on_mi3xx() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx942", "gfx950"])


@cache
def on_gfx9() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])


@cache
def on_gfx950() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx950"])


@cache
def use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
    sliding_window: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> bool:
    from vllm._aiter_ops import rocm_aiter_ops

    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    ON_GFX9 = any(arch in GPU_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
    ON_GFX11_GFX12 = any(arch in GPU_ARCH for arch in ["gfx11", "gfx12"])

    # custom paged attn always supported on V0. On V1, requires sliding window
    # disabled due to observed numerical discrepancy.
    if ON_GFX9:
        return (
            (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and (envs.VLLM_ROCM_CUSTOM_PAGED_ATTN)
            and not (rocm_aiter_ops.is_pa_attn_enabled())
            and sinks is None
        )

    else:
        return (
            ON_GFX11_GFX12
            and (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and head_size == 128
            and block_size == 16
            and (gqa_ratio >= 3 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and alibi_slopes is None
            and kv_cache_dtype == "auto"
            and envs.VLLM_ROCM_CUSTOM_PAGED_ATTN
            and sinks is None
        )


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    # rocm shares the same device control env var as CUDA
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    supported_quantization: list[str] = [
        "awq",
        "gptq",
        "fp8",
        "compressed-tensors",
        "fbgemm_fp8",
        "gguf",
        "quark",
        "ptpc_fp8",
        "mxfp4",
        "petit_nvfp4",
        "torchao",
    ]
    # bitsandbytes not supported on gfx9 (warp size 64 limitation)
    if not on_gfx9():
        supported_quantization += ["bitsandbytes"]

    @classmethod
    def get_vit_attn_backend(
        cls, head_size: int, dtype: torch.dtype
    ) -> AttentionBackendEnum:
        from importlib.util import find_spec

        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.attention.backends.registry import AttentionBackendEnum

        if rocm_aiter_ops.is_mha_enabled():
            # Note: AITER FA is only supported for Qwen-VL models.
            # TODO: Add support for other VL models in their model class.
            return AttentionBackendEnum.ROCM_AITER_FA

        if on_gfx9() and find_spec("flash_attn") is not None:
            return AttentionBackendEnum.FLASH_ATTN

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        attn_type: str | None = None,
    ) -> str:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.attention.backends.registry import AttentionBackendEnum

        if use_sparse:
            if kv_cache_dtype.startswith("fp8"):
                raise ValueError(
                    "ROCMAiterMLASparseBackend doesn't support fp8 kv_cache_dtype."
                )
            assert block_size == 1, (
                "Sparse MLA backend on ROCm only supports block size 1 for now."
            )
            logger.info_once("Using Sparse MLA backend on V1 engine.")
            return AttentionBackendEnum.ROCM_AITER_MLA_SPARSE.get_path()

        if use_mla:
            if selected_backend is None:
                selected_backend = (
                    AttentionBackendEnum.ROCM_AITER_MLA
                    if rocm_aiter_ops.is_mla_enabled() or block_size == 1
                    else AttentionBackendEnum.TRITON_MLA
                )
            if selected_backend == AttentionBackendEnum.TRITON_MLA:
                if block_size != 1:
                    logger.info_once("Using Triton MLA backend.")
                    return AttentionBackendEnum.TRITON_MLA.get_path()
                raise ValueError(
                    f" The selected backend, {selected_backend.name},"
                    f"does not support block size {block_size}."
                )
            if selected_backend == AttentionBackendEnum.ROCM_AITER_MLA:
                logger.info("Using AITER MLA backend.")
                return AttentionBackendEnum.ROCM_AITER_MLA.get_path()
            if selected_backend == AttentionBackendEnum.ROCM_AITER_TRITON_MLA:
                logger.info("Using AITER TRITON MLA backend.")
                return AttentionBackendEnum.ROCM_AITER_TRITON_MLA.get_path()

            raise ValueError(
                f" The selected backend, {selected_backend.name},"
                f"is not MLA type while requested for MLA backend."
            )

        if selected_backend == AttentionBackendEnum.FLEX_ATTENTION:
            logger.info("Using FlexAttention backend.")
            return AttentionBackendEnum.FLEX_ATTENTION.get_path()

        if selected_backend == AttentionBackendEnum.TRITON_ATTN:
            logger.info("Using Triton Attention backend on V1 engine.")
            return AttentionBackendEnum.TRITON_ATTN.get_path()

        if selected_backend == AttentionBackendEnum.ROCM_ATTN:
            logger.info("Using Rocm Attention backend on V1 engine.")
            return AttentionBackendEnum.ROCM_ATTN.get_path()

        if selected_backend == AttentionBackendEnum.ROCM_AITER_FA:
            if on_gfx9():
                logger.info("Using Aiter Flash Attention backend on V1 engine.")
                return AttentionBackendEnum.ROCM_AITER_FA.get_path()
            else:
                raise ValueError(
                    f"The selected backend, {selected_backend.name}, "
                    "is only supported on gfx9 architectures."
                )

        if selected_backend == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN:
            logger.info("Using Aiter Unified Attention backend on V1 engine.")
            return AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path()

        # Handle automatic backend selection based on environment variables
        if selected_backend is None:
            # Priority 1: Check for AITER Unified Attention (must check before MHA)
            if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:
                logger.info("Using Aiter Unified Attention backend on V1 engine.")
                return AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path()

            # Priority 2: Check for AITER MHA (Flash Attention)
            # Only use if explicitly enabled (not just VLLM_ROCM_USE_AITER=1)
            if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MHA and on_gfx9():
                logger.info("Using Aiter Flash Attention backend on V1 engine.")
                return AttentionBackendEnum.ROCM_AITER_FA.get_path()

            # Priority 3: Check for ROCM_ATTN (prefill-decode split)
            if envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION:
                logger.info("Using Rocm Attention backend on V1 engine.")
                return AttentionBackendEnum.ROCM_ATTN.get_path()

            # Priority 4: Check for AITER enabled without specific flags
            # This defaults to AITER FA only if MHA is not explicitly disabled
            if (
                envs.VLLM_ROCM_USE_AITER
                and on_gfx9()
                and envs.VLLM_ROCM_USE_AITER_MHA is not False
            ):
                logger.info("Using Aiter Flash Attention backend on V1 engine.")
                return AttentionBackendEnum.ROCM_AITER_FA.get_path()

            # Default: Triton Unified Attention
            logger.info("Using Triton Attention backend on V1 engine.")
            return AttentionBackendEnum.TRITON_ATTN.get_path()

        raise RuntimeError(
            "V0 attention backends have been removed. Set VLLM_USE_V1=1 "
            "to select a supported backend."
        )

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @with_amdsmi_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        device_name: str = asic_info["device_id"]
        if device_name in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[device_name]
        return asic_info["market_name"]

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.config.compilation import CUDAGraphMode

        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
        is_eager_execution = compilation_config == CUDAGraphMode.NONE

        use_aiter_rms_norm = rocm_aiter_ops.is_rmsnorm_enabled()

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
        #  Aiter rms norm perform best when CUDA Graph capture is enabled.
        if (
            use_aiter_rms_norm
            and not is_eager_execution
            and "-rms_norm" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rms_norm")

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        if model_arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model architecture '{model_arch}' is not supported by ROCm for now."
            )

        if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
            msg = _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch]
            logger.warning(
                "Model architecture '%s' is partially supported by ROCm: %s",
                model_arch,
                msg,
            )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ."
            )
        os.environ["VLLM_USE_TRITON_AWQ"] = "1"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_mx(cls) -> bool:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])

    @classmethod
    def supports_fp8(cls) -> bool:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95", "gfx12"])

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        supported_archs = ["gfx94", "gfx95"]
        return any(gfx in gcn_arch for gfx in supported_archs)

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def is_navi(cls) -> bool:
        return "gfx1" in torch.cuda.get_device_properties(0).gcnArchName

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
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True
