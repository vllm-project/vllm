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
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


def _silu_and_mul_fp8_static(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
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


# (activation module type, consumer input_quant_key) -> fused producer.
# Mirrors ActivationQuantFusionPass.FUSED_OPS; add a row to migrate a scheme.
_FUSED_ACT_QUANT: dict[tuple[type, QuantKey], Callable] = {
    (SiluAndMul, kFp8StaticTensorSym): _silu_and_mul_fp8_static,
}


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
