# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer side of the QuantizedActivation contract for normalization layers.

Given a normalization module (e.g. RMSNorm), the input hidden_states (plus
optional residual), and the downstream linear it feeds (e.g. qkv_proj or
gate_up_proj), fuse the residual add + normalization + downstream input
quantization into a single kernel when the linear advertises a consumable
input quantization key (see quant_activation.py).

Crucially, when a residual tensor is provided, the residual highway is updated
in-place and returned in full precision (e.g. BF16/FP16) so no quantization
error accumulates across layers, while hidden_states is returned as a
pre-quantized QuantizedActivation for the immediate consumer GEMM.
"""

from collections.abc import Callable
from typing import Any, overload

import torch

from vllm import envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
    get_input_quant_scale,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import round_up

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

_FLASHINFER_ADD_RMSNORM_FP4QUANT: Any | None = None
if (
    current_platform.is_cuda()
    and hasattr(torch, "float4_e2m1fn_x2")
    and has_flashinfer()
):
    try:
        from flashinfer import add_rmsnorm_fp4quant as _fi_add_rmsnorm_fp4quant
    except ImportError:
        pass
    else:
        _FLASHINFER_ADD_RMSNORM_FP4QUANT = _fi_add_rmsnorm_fp4quant


def _should_defer_to_allreduce_rms_fusion(residual: torch.Tensor | None) -> bool:
    """Defer manual norm+quant fusion when compiler AllReduce+RMSNorm+Quant is active."""
    if residual is None:
        return False
    try:
        from vllm.config import CompilationMode, get_current_vllm_config
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() <= 1:
            return False
        comp_cfg = get_current_vllm_config().compilation_config
        return (
            comp_cfg.mode >= CompilationMode.VLLM_COMPILE
            and comp_cfg.pass_config.fuse_allreduce_rms
        )
    except Exception:
        return False


def _rms_norm_common_supported(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> bool:
    """Common precondition checks for fused RMSNorm + Quant kernels."""
    if not isinstance(norm_fn, RMSNorm):
        return False
    if not norm_fn.has_weight or norm_fn.variance_size_override is not None:
        return False
    if envs.VLLM_BATCH_INVARIANT:
        return False
    if _should_defer_to_allreduce_rms_fusion(residual):
        return False
    if x.dtype != norm_fn.weight.dtype:
        return False
    if residual is not None and (
        residual.dtype != x.dtype or not residual.is_contiguous()
    ):
        return False
    return True


def _rms_norm_fp8_static_supported(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> bool:
    return (
        _rms_norm_common_supported(norm_fn, x, residual)
        and get_input_quant_scale(linear) is not None
    )


def _rms_norm_fp8_static(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> QuantizedActivation | tuple[QuantizedActivation, torch.Tensor]:
    """RMSNorm (with optional fused residual add) + static per-tensor FP8 quant."""
    scale = get_input_quant_scale(linear)
    assert scale is not None, "input_scale is required for static FP8 quantization"
    result = torch.empty(x.shape, dtype=FP8_DTYPE, device=x.device)

    if residual is None:
        torch.ops._C.rms_norm_static_fp8_quant(
            result, x, norm_fn.weight.data, scale, norm_fn.variance_epsilon
        )
    else:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            result,
            x,
            residual,
            norm_fn.weight.data,
            scale,
            norm_fn.variance_epsilon,
        )

    qa = QuantizedActivation(
        data=result,
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )
    return qa if residual is None else (qa, residual)


def _rms_norm_fp8_dynamic_token(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> QuantizedActivation | tuple[QuantizedActivation, torch.Tensor]:
    """RMSNorm (with optional fused residual add) + dynamic per-token FP8 quant."""
    d = x.shape[-1]
    num_tokens = x.numel() // d
    result = torch.empty(x.shape, dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)

    torch.ops._C.rms_norm_dynamic_per_token_quant(
        result,
        x,
        norm_fn.weight.data,
        scales,
        norm_fn.variance_epsilon,
        None,
        residual,
    )

    qa = QuantizedActivation(
        data=result,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8DynamicTokenSym,
    )
    return qa if residual is None else (qa, residual)


def _rms_norm_fp8_dynamic_block_supported(
    group_size: int,
) -> Callable[[torch.nn.Module, torch.Tensor, LinearBase, torch.Tensor | None], bool]:
    def _supported(
        norm_fn: torch.nn.Module,
        x: torch.Tensor,
        linear: LinearBase,
        residual: torch.Tensor | None,
    ) -> bool:
        return (
            _rms_norm_common_supported(norm_fn, x, residual)
            and x.ndim in (2, 3)
            and x.shape[-1] % group_size == 0
        )

    return _supported


def _rms_norm_fp8_dynamic_block(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
    group_size: int,
    quant_key: QuantKey,
) -> QuantizedActivation | tuple[QuantizedActivation, torch.Tensor]:
    """RMSNorm (with optional fused residual add) + dynamic per-block FP8 quant."""
    d = x.shape[-1]
    num_tokens = x.numel() // d
    num_groups = d // group_size

    result = torch.empty(x.shape, dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty(
        (num_tokens, num_groups), dtype=torch.float32, device=x.device
    )

    torch.ops._C.rms_norm_per_block_quant(
        result,
        x,
        norm_fn.weight.data,
        scales,
        norm_fn.variance_epsilon,
        None,
        residual,
        group_size,
        False,
    )

    qa = QuantizedActivation(
        data=result,
        scale=scales.view(x.shape[:-1] + (num_groups,)),
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=quant_key,
    )
    return qa if residual is None else (qa, residual)


def _rms_norm_fp8_dynamic_128(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> QuantizedActivation | tuple[QuantizedActivation, torch.Tensor]:
    return _rms_norm_fp8_dynamic_block(
        norm_fn, x, linear, residual, 128, kFp8Dynamic128Sym
    )


def _rms_norm_fp8_dynamic_64(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> QuantizedActivation | tuple[QuantizedActivation, torch.Tensor]:
    return _rms_norm_fp8_dynamic_block(
        norm_fn, x, linear, residual, 64, kFp8Dynamic64Sym
    )


def _rms_norm_nvfp4_dynamic_supported(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> bool:
    return (
        _rms_norm_common_supported(norm_fn, x, residual)
        and residual is not None
        and x.ndim in (2, 3)
        and x.shape[-1] % 16 == 0
        and get_input_quant_scale(linear) is not None
    )


def _rms_norm_nvfp4_dynamic(
    norm_fn: RMSNorm,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Fused residual add + RMSNorm + NVFP4 dynamic block quantization."""
    assert residual is not None
    assert _FLASHINFER_ADD_RMSNORM_FP4QUANT is not None

    d = x.shape[-1]
    out_shape = x.shape
    x_2d = x.reshape(-1, d)
    res_2d = residual.reshape(-1, d)
    num_tokens = x_2d.shape[0]

    # NVFP4 packs 2 FP4 values into 1 uint8 byte
    result = torch.empty((num_tokens, d // 2), dtype=FP4_DTYPE, device=x.device)

    # Swizzled 128x4 tile block scale layout expected by NVFP4 consumer kernels
    rounded_m = round_up(num_tokens, 128)
    rounded_n = round_up(d // 16, 4)
    block_scale = torch.empty(
        (rounded_m, rounded_n // 4), dtype=torch.int32, device=x.device
    ).view(FP8_DTYPE)
    block_scale_unswizzled = torch.empty(
        (num_tokens, d // 16), dtype=FP8_DTYPE, device=x.device
    )

    input_global_scale_inv = get_input_quant_scale(linear)
    assert input_global_scale_inv is not None, (
        "input_global_scale_inv is required for NVFP4 quantization"
    )

    _FLASHINFER_ADD_RMSNORM_FP4QUANT(
        x_2d,
        res_2d,
        norm_fn.weight.data,
        y_fp4=result.view(torch.float4_e2m1fn_x2),
        block_scale=block_scale.view(torch.float8_e4m3fn),
        global_scale=input_global_scale_inv.reshape(1),
        eps=norm_fn.variance_epsilon,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=True,
        output_both_sf_layouts=False,
        block_scale_unswizzled=block_scale_unswizzled,
    )

    qa = QuantizedActivation(
        data=result.view(out_shape[:-1] + (d // 2,)),
        scale=block_scale,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=kNvfp4Dynamic,
    )
    return qa, residual


# (normalization module type, consumer input quantization key) -> fused producer
_FUSED_NORM_QUANT: dict[tuple[type, QuantKey], Callable] = {
    (RMSNorm, kFp8StaticTensorSym): _rms_norm_fp8_static,
    (RMSNorm, kFp8DynamicTokenSym): _rms_norm_fp8_dynamic_token,
}

_FUSED_NORM_QUANT_SUPPORT: dict[
    tuple[type, QuantKey],
    Callable[[torch.nn.Module, torch.Tensor, LinearBase, torch.Tensor | None], bool],
] = {
    (RMSNorm, kFp8StaticTensorSym): _rms_norm_fp8_static_supported,
    (RMSNorm, kFp8DynamicTokenSym): lambda norm_fn, x, linear, residual: (
        _rms_norm_common_supported(norm_fn, x, residual)
    ),
}

if current_platform.is_cuda_alike() and hasattr(
    torch.ops._C, "rms_norm_per_block_quant"
):
    _FUSED_NORM_QUANT[(RMSNorm, kFp8Dynamic128Sym)] = _rms_norm_fp8_dynamic_128
    _FUSED_NORM_QUANT_SUPPORT[(RMSNorm, kFp8Dynamic128Sym)] = (
        _rms_norm_fp8_dynamic_block_supported(128)
    )
    _FUSED_NORM_QUANT[(RMSNorm, kFp8Dynamic64Sym)] = _rms_norm_fp8_dynamic_64
    _FUSED_NORM_QUANT_SUPPORT[(RMSNorm, kFp8Dynamic64Sym)] = (
        _rms_norm_fp8_dynamic_block_supported(64)
    )

if (
    current_platform.is_cuda()
    and current_platform.has_device_capability(100)
    and _FLASHINFER_ADD_RMSNORM_FP4QUANT is not None
):
    _FUSED_NORM_QUANT[(RMSNorm, kNvfp4Dynamic)] = _rms_norm_nvfp4_dynamic
    _FUSED_NORM_QUANT_SUPPORT[(RMSNorm, kNvfp4Dynamic)] = (
        _rms_norm_nvfp4_dynamic_supported
    )


@overload
def maybe_fused_norm_quant(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
    residual: None = None,
) -> "torch.Tensor | QuantizedActivation": ...


@overload
def maybe_fused_norm_quant(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor,
) -> "tuple[torch.Tensor | QuantizedActivation, torch.Tensor]": ...


def maybe_fused_norm_quant(
    norm_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
    residual: torch.Tensor | None = None,
) -> (
    "torch.Tensor | QuantizedActivation | "
    "tuple[torch.Tensor | QuantizedActivation, torch.Tensor]"
):
    """Apply norm_fn, fusing the downstream linear's input quant when possible.

    When ``residual`` is provided, the residual is updated in-place in full
    precision (e.g. BF16/FP16) and returned as the second tuple element, while
    the normalized output is returned as a ``QuantizedActivation`` matching
    ``linear``'s input quantization key. Falls back to ``norm_fn(x)`` or
    ``norm_fn(x, residual)`` when no fused kernel matches or when ``linear``
    requires unquantized input (such as when LoRA is active).
    """
    key = get_input_quant_key(linear)
    if key is not None:
        registry_key = (type(norm_fn), key)
        producer = _FUSED_NORM_QUANT.get(registry_key)
        support = _FUSED_NORM_QUANT_SUPPORT.get(registry_key)
        if producer is not None and (
            support is None or support(norm_fn, x, linear, residual)
        ):
            return producer(norm_fn, x, linear, residual)
    if residual is None:
        return norm_fn(x)
    return norm_fn(x, residual)
