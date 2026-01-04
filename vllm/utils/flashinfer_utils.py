# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import functools
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import requests
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    create_flashinfer_prepare_finalize,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
# Referenced from https://github.com/flashinfer-ai/flashinfer/blob/0c9a92c3d9a7e043ab6f3f7b2273269caf6ab044/flashinfer/jit/cubin_loader.py#L35  # noqa: E501
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"
    CUTEDSL = "CUTEDSL"


@functools.cache
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.VLLM_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # When not using flashinfer cubin,
    # Also check if nvcc is available since it's required to JIT compile flashinfer
    if not has_flashinfer_cubin() and shutil.which("nvcc") is None:
        logger.debug_once(
            "FlashInfer unavailable since nvcc was not found "
            "and not using pre-downloaded cubins"
        )
        return False
    return True


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


# Create lazy wrappers for each function
flashinfer_trtllm_fp8_block_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"
)
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"
)
flashinfer_cutlass_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "cutlass_fused_moe"
)
flashinfer_cutedsl_grouped_gemm_nt_masked = _lazy_import_wrapper(
    "flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"
)
flashinfer_fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_batched_quantize = _lazy_import_wrapper("flashinfer", "nvfp4_batched_quantize")
silu_and_mul_scaled_nvfp4_experts_quantize = _lazy_import_wrapper(
    "flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"
)
scaled_fp4_grouped_quantize = _lazy_import_wrapper(
    "flashinfer", "scaled_fp4_grouped_quantize"
)
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer", "nvfp4_block_scale_interleave"
)
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe"
)


# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_all2all() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return `True` if FlashInfer MoE module is available."""
    return (
        has_flashinfer()
        and importlib.util.find_spec("flashinfer.fused_moe") is not None
    )


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """Return ``True`` if FlashInfer cutedsl module is available."""
    return (
        has_flashinfer() and importlib.util.find_spec("flashinfer.cute_dsl") is not None
    )


@functools.cache
def has_flashinfer_trtllm_fused_moe() -> bool:
    """Return `True` if FlashInfer TRTLLM fused MoE is available."""
    if not has_flashinfer_moe():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_cutedsl():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"),
        ("flashinfer", "scaled_fp4_grouped_quantize"),
        ("flashinfer", "silu_and_scaled_nvfp4_experts_quantize"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return `True` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    # If we have pre-downloaded cubins, we can assume the cubins are available.
    if has_flashinfer_cubin():
        return True

    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code,
            )
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


@functools.cache
def supports_trtllm_attention() -> bool:
    """
    TRTLLM attention is supported if the platform is SM100,
    NVIDIA artifactory is accessible, and batch-invariant mode is not enabled.
    """
    # Batch-invariant mode disables TRTLLM attention
    if vllm_is_batch_invariant():
        return False

    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
    return (
        current_platform.is_device_capability_family(100) and has_nvidia_artifactory()
    )


def force_use_trtllm_attention() -> bool | None:
    """
    This function should only be called during initialization stage when vllm config
    is set.
    Return `None` if --attention-config.use_trtllm_attention is not set,
    return `True` if TRTLLM attention is forced to be used,
    return `False` if TRTLLM attention is forced to be not used.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.attention_config.use_trtllm_attention


def can_use_trtllm_attention(num_qo_heads: int, num_kv_heads: int) -> bool:
    """Check if the current configuration supports TRTLLM attention."""
    if force_use_trtllm_attention() is False:
        return False
    has_trtllm = supports_trtllm_attention()
    # num_kv_heads=1 is not supported due to TMA descriptor building limitations.
    # When num_kv_heads=1, the KV cache strides become degenerate (stride_heads ==
    # stride_batch), which causes CUDA's cuTensorMapEncodeTiled to fail because
    # TMA descriptors cannot handle degenerate 4D tensors with singleton dimensions.
    # See: https://fburl.com/352mrydz
    if has_trtllm and num_kv_heads == 1:
        logger.warning_once(
            "TRTLLM attention does not support num_kv_heads=1. "
            "This configuration causes TMA descriptor building to fail due to "
            "degenerate tensor strides. Falling back to FlashInfer attention."
        )
    return has_trtllm and (num_qo_heads % num_kv_heads == 0) and (num_kv_heads != 1)


def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    kv_cache_dtype: str,
    q_dtype: torch.dtype,
    is_prefill: bool,
    # None means auto-detection, True means force on, False means force off
    force_use_trtllm: bool | None = None,
    has_sinks: bool = False,
    has_spec: bool = False,
) -> bool:
    """Return `True` if TRTLLM attention is used."""

    # CLI argument is set to 0 - respect it
    if force_use_trtllm is not None and not force_use_trtllm:
        return False

    # Decode context parallel is not supported
    if dcp_world_size > 1:
        logger.warning_once(
            "Trtllm does not support returning LSE and as a result "
            "does not support DCP, reverting to FlashInfer"
        )
        return False

    # The platform is not supported
    if not supports_trtllm_attention():
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported on this platform, "
                "but --attention-config.use_trtllm_attention is set to 1"
            )
        return False

    # The combination of query and key heads is not supported
    if num_qo_heads % num_kv_heads != 0:
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported for this combination of "
                "query and key heads, but --attention-config.use_trtllm_attention is "
                "set to 1"
            )
        return False

    # num_kv_heads=1 is not supported
    if num_kv_heads == 1:
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention does not support num_kv_heads=1, "
                "but --attention-config.use_trtllm_attention is set to 1"
            )
        return False

    if has_spec and not is_prefill:
        # Speculative decoding requires TRTLLM attention for decodes
        logger.info_once("Using TRTLLM attention (enabled for speculative decoding).")
        return True

    # Must use TRTLLM attention if query is FP8 quantized
    if q_dtype == current_platform.fp8_dtype():
        logger.info_once("Using TRTLLM attention (query is quantized).")
        return True

    # If sinks are being used, we must use TRTLLM attention as it's
    # the only backend that supports them
    if has_sinks:
        logger.info_once("Using TRTLLM attention (required for attention sinks).")
        return True

    if force_use_trtllm is None:
        # CLI argument not set - use auto-detection
        if is_prefill:
            # Prefill auto-detection
            use_trtllm = kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM prefill attention (auto-detected).")
        else:
            # Decode auto-detection
            use_trtllm = num_tokens <= 256 and kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM decode attention (auto-detected).")
        return use_trtllm

    # CLI argument is set to 1 - respect it
    logger.info_once(
        "Using TRTLLM attention (--attention-config.use_trtllm_attention is set to 1)"
    )
    return True


if has_flashinfer():

    @torch.library.custom_op(
        "vllm::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_

        return flashinfer_mm_fp4_(
            A, B, A_scale, B_scale, g_scale, dtype, block_size=16, backend=backend
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::bmm_fp8",
        mutates_args=[],
        device_types="cuda",
    )
    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import bmm_fp8 as bmm_fp8_

        return bmm_fp8_(A, B, A_scale, B_scale, dtype, None, backend)

    @torch.library.register_fake(
        "vllm::bmm_fp8",
    )
    def bmm_fp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=dtype, device=A.device
        )


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if backend == "cutlass":
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        backend=backend,
    )


def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.device.type == "cuda" and b.device.type == "cuda"
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device.type == "cuda" and scale_b.device.type == "cuda"

    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])

    if bias is not None:
        output = output + bias
    return output


def calculate_tile_tokens_dim(num_tokens, top_k, num_experts):
    from flashinfer import next_positive_power_of_2

    # FlashInfer 0.2.10 has issues with larger tile sizes. Set to 8 for now.
    # TODO: Revert this to dynamic calculation once a new version of FlashInfer
    # with the necessary kernels is released.
    tile_tokens_dim = 8

    # A factor considering tokens are not perfectly balanced among experts.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert
    # assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-max_tile_tokens_dim tokens per CTA tile
    # as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_flashinfer_fp8_moe_weights(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor
):
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i])
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )


def apply_flashinfer_per_tensor_scale_fp8(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    top_k: int,
    num_expert_group: int | None,
    topk_group: int | None,
    global_num_experts: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from flashinfer.fused_moe import RoutingMethodType

    import vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe  # noqa: E501, F401

    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_scalar to be initialized"
    )
    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_gate_scalar to be initialized"
    )
    assert layer.output1_scales_scalar is not None, (
        "Expected output2_scales_scalar to be initialized"
    )

    from vllm.model_executor.models.llama4 import Llama4MoE

    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, (
        "FusedMoE flashinfer kernels are only supported for Llama4"
    )
    return torch.ops.vllm.flashinfer_fused_moe_per_tensor_scale_fp8(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        input_scale=layer.w13_input_scale,
        gemm1_weights=layer.w13_weight,
        gemm2_weights=layer.w2_weight,
        output1_scales_scalar=layer.output1_scales_scalar,
        output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
        output2_scales_scalar=layer.output2_scales_scalar,
        num_experts=global_num_experts,
        top_k=top_k,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        use_routing_scales_on_input=apply_router_weight_on_input,
        routing_method_type=RoutingMethodType.Llama4,
    )


def get_moe_scaling_factors(
    input_scale: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output1_scales_scalar = gemm1_weights_scale * input_scale * (1.0 / activation_scale)
    output1_scales_gate_scalar = gemm1_weights_scale * input_scale
    output2_scales_scalar = activation_scale * gemm2_weights_scale

    return output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar


def register_moe_scaling_factors(layer: torch.nn.Module) -> None:
    output1_scales, output1_gate_scales, output2_scales = get_moe_scaling_factors(
        layer.w13_input_scale,
        layer.w13_weight_scale,
        layer.w2_input_scale,
        layer.w2_weight_scale,
    )
    layer.register_parameter(
        "output1_scales_scalar", torch.nn.Parameter(output1_scales, requires_grad=False)
    )
    layer.register_parameter(
        "output1_scales_gate_scalar",
        torch.nn.Parameter(output1_gate_scales, requires_grad=False),
    )
    layer.register_parameter(
        "output2_scales_scalar", torch.nn.Parameter(output2_scales, requires_grad=False)
    )
    layer.register_parameter(
        "w2_input_scale_inv",
        torch.nn.Parameter(1.0 / layer.w2_input_scale, requires_grad=False),
    )


def build_flashinfer_fp8_cutlass_moe_prepare_finalize(
    moe: FusedMoEConfig | None, use_deepseek_fp8_block_scale: bool = False
) -> mk.FusedMoEPrepareAndFinalize:
    """Create a FlashInfer CUTLASS fused-MoE prepare finalize kernel"""
    use_dp = moe.moe_parallel_config.dp_size > 1 if moe is not None else False
    # Propagate block-scale flag so prepare/finalize can skip act quantization
    # and inform the kernel to consume per-block weight scales.
    return create_flashinfer_prepare_finalize(
        use_dp, use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale
    )


def select_cutlass_fp8_gemm_impl(
    moe: FusedMoEConfig | None,
    quant_config: FusedMoEQuantConfig,
    out_dtype: torch.dtype | None = None,
    use_deepseek_fp8_block_scale: bool = False,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    """Return a GEMM *experts* implementation for fused-MoE layers"""

    if moe is not None:
        return FlashInferExperts(
            out_dtype=moe.in_dtype,
            quant_config=quant_config,
            ep_rank=moe.moe_parallel_config.ep_rank,
            ep_size=moe.moe_parallel_config.ep_size,
            tp_rank=moe.moe_parallel_config.tp_rank,
            tp_size=moe.moe_parallel_config.tp_size,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        )

    assert out_dtype is not None, "If moe config is None, out_dtype must be passed"
    return FlashInferExperts(
        out_dtype=out_dtype,
        quant_config=quant_config,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
    )


def flashinfer_cutlass_moe_fp8(
    hidden_states: torch.Tensor,
    layer: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    moe: FusedMoEConfig | None = None,
) -> torch.Tensor:
    quant_config = layer.quant_method.get_fused_moe_quant_config(layer)
    assert quant_config is not None

    # Construct modular kernel with block-scale support when requested.
    fused_experts = mk.FusedMoEModularKernel(
        build_flashinfer_fp8_cutlass_moe_prepare_finalize(
            moe=moe, use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale
        ),
        select_cutlass_fp8_gemm_impl(
            moe=moe,
            quant_config=quant_config,
            out_dtype=hidden_states.dtype,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        ),
        moe_parallel_config=layer.moe_parallel_config,
    )

    return fused_experts(
        hidden_states,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )


def get_flashinfer_moe_backend() -> FlashinferMoeBackend:
    backend_map = {
        "throughput": FlashinferMoeBackend.CUTLASS,
        "latency": FlashinferMoeBackend.TENSORRT_LLM,
        "masked_gemm": FlashinferMoeBackend.CUTEDSL,
    }

    flashinfer_moe_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
    if flashinfer_moe_backend in backend_map:
        if (
            flashinfer_moe_backend == "latency"
            and not current_platform.is_device_capability_family(100)
        ):
            logger.info_once(
                "Flashinfer TRTLLM MOE backend is only supported on "
                "SM100 and later, using CUTLASS backend instead",
                scope="local",
            )
            return FlashinferMoeBackend.CUTLASS
        return backend_map[flashinfer_moe_backend]
    elif current_platform.is_device_capability(90):
        return FlashinferMoeBackend.CUTLASS

    raise ValueError(
        f"Unknown flashinfer moe backend: {flashinfer_moe_backend!r}. "
        f"Expected one of {list(backend_map.keys())}."
    )


def is_flashinfer_supporting_global_sf(backend: FlashinferMoeBackend | None) -> bool:
    # TODO(shuw@nvidia): Update when new backends are added.
    backends_supporting_global_sf = (
        FlashinferMoeBackend.CUTLASS,
        FlashinferMoeBackend.TENSORRT_LLM,
    )
    return backend in backends_supporting_global_sf


__all__ = [
    "has_flashinfer_cubin",
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_trtllm_fp8_per_tensor_scale_moe",
    "flashinfer_fp4_quantize",
    "nvfp4_batched_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_cutedsl",
    "has_flashinfer_trtllm_fused_moe",
    "has_flashinfer_comm",
    "has_flashinfer_all2all",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_nvidia_artifactory",
    "force_use_trtllm_attention",
    "supports_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
    "calculate_tile_tokens_dim",
    "swap_w13_to_w31",
    "rotate_flashinfer_fp8_moe_weights",
    "apply_flashinfer_per_tensor_scale_fp8",
    "get_moe_scaling_factors",
    "register_moe_scaling_factors",
    "build_flashinfer_fp8_cutlass_moe_prepare_finalize",
    "select_cutlass_fp8_gemm_impl",
    "flashinfer_cutlass_moe_fp8",
    "get_flashinfer_moe_backend",
    "is_flashinfer_supporting_global_sf",
    "FlashinferMoeBackend",
]
