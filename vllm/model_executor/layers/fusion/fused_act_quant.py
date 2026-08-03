# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Producer side of the QuantizedActivation contract for activation layers.

Given an activation module and the downstream linear it feeds, fuse the
activation with that linear's input quantization into a single kernel when the
linear advertises a consumable input_quant_key (see quant_activation.py).
Falls back to the plain activation when nothing matches, so a model forward can
always call maybe_fused_act_quant unconditionally.

This is the manual-fusion counterpart to ActivationQuantFusionPass: when fusion
fires here the silu_and_mul pattern is already consumed, so the compiler pass
finds nothing to rewrite and the two never double-fuse.
"""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def _silu_and_mul_fp8_static(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """SiluAndMul + FP8 static per-tensor quantization."""
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    result = torch.empty(out_shape, dtype=FP8_DTYPE, device=x.device)
    # TODO(mgoin): read the consumer scale via the contract instead of reaching
    # into the kernel-specific input_scale attribute.
    scale = linear.input_scale
    torch.ops._C.silu_and_mul_quant(result, x, scale)
    return QuantizedActivation(
        data=result,
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=kFp8StaticTensorSym,
    )


def _silu_and_mul_fp8_dynamic_block(
    x: torch.Tensor, linear: LinearBase, group_size: int, quant_key: QuantKey
) -> QuantizedActivation:
    """SiluAndMul + FP8 dynamic per-block quantization."""
    assert x.ndim == 2, f"Input must be 2D [batch, hidden*2], got {x.shape}"

    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    num_tokens = x.shape[0]
    num_groups = d // group_size

    result = torch.empty((num_tokens, d), dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty((num_tokens, num_groups), dtype=torch.float32, device=x.device)

    torch.ops._C.silu_and_mul_per_block_quant(
        out=result,
        input=x,
        scales=scales,
        group_size=group_size,
        scale_ub=None,
        is_scale_transposed=False,
    )

    return QuantizedActivation(
        data=result.view(out_shape),
        scale=scales.view(out_shape[:-1] + (num_groups,)),
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=quant_key,
    )


def _silu_and_mul_fp8_dynamic_128(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """SiluAndMul + FP8 dynamic per-group (group=128) quantization."""
    return _silu_and_mul_fp8_dynamic_block(x, linear, 128, kFp8Dynamic128Sym)


def _silu_and_mul_fp8_dynamic_64(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """SiluAndMul + FP8 dynamic per-group (group=64) quantization."""
    return _silu_and_mul_fp8_dynamic_block(x, linear, 64, kFp8Dynamic64Sym)


def _silu_and_mul_nvfp4_dynamic(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """SiluAndMul + NVFP4 dynamic quantization."""
    assert x.ndim == 2, f"Input must be 2D [batch, hidden*2], got {x.shape}"

    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    num_tokens = x.shape[0]

    # NVFP4 packs 2 values into 1 byte
    result = torch.empty((num_tokens, d // 2), dtype=FP4_DTYPE, device=x.device)

    # Block scale output shape: swizzled layout for tensor cores
    # Each group of 16 elements shares one FP8 scale
    num_k_tiles = (d + 63) // 64
    block_scale = torch.empty(
        (num_tokens, num_k_tiles * 4), dtype=FP8_DTYPE, device=x.device
    )

    input_global_scale = getattr(linear, "input_global_scale", None)
    if input_global_scale is None:
        input_global_scale = torch.tensor([1.0], dtype=torch.float32, device=x.device)

    torch.ops._C.silu_and_mul_nvfp4_quant(result, block_scale, x, input_global_scale)

    return QuantizedActivation(
        data=result.view(out_shape[:-1] + (d // 2,)),
        scale=block_scale,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=kNvfp4Dynamic,
    )


# (activation module type, consumer input_quant_key) -> fused producer.
# Mirrors ActivationQuantFusionPass.FUSED_OPS; add a row to migrate a scheme.
_FUSED_ACT_QUANT: dict[tuple[type, QuantKey], Callable] = {
    (SiluAndMul, kFp8StaticTensorSym): _silu_and_mul_fp8_static,
}

# Add CUDA-specific entries for dynamic block quantization
if current_platform.is_cuda_alike():
    _FUSED_ACT_QUANT.update(
        {
            (SiluAndMul, kFp8Dynamic128Sym): _silu_and_mul_fp8_dynamic_128,
            (SiluAndMul, kFp8Dynamic64Sym): _silu_and_mul_fp8_dynamic_64,
        }
    )

# Add NVFP4 if supported (requires SM100+)
if current_platform.is_cuda() and hasattr(torch.ops._C, "silu_and_mul_nvfp4_quant"):
    _FUSED_ACT_QUANT[(SiluAndMul, kNvfp4Dynamic)] = _silu_and_mul_nvfp4_dynamic


def maybe_fused_act_quant(
    act_fn: torch.nn.Module,
    x: torch.Tensor,
    linear: LinearBase,
) -> "torch.Tensor | QuantizedActivation":
    """Apply act_fn, fusing the downstream linear's input quant when possible.

    Returns a QuantizedActivation when a fused kernel matches
    (act_fn, linear.input_quant_key), else the plain activated tensor.
    """
    key = getattr(linear, "input_quant_key", None)
    if key is not None:
        producer = _FUSED_ACT_QUANT.get((type(act_fn), key))
        if producer is not None:
            return producer(x, linear)
    return act_fn(x)
