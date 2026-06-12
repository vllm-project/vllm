# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual RMSNorm + input-quant fusion (RFC #43224, issue #43500).

Replaces the ``RMSNormQuantFusionPass`` compiler fusion with an explicit
call site in model code: the decoder layer asks :func:`rms_norm_input_quant`
to normalize its hidden states, and when the downstream linear layer has
registered an ``input_quant_key``, the norm and the activation quantization
are executed as a single fused kernel. The result is passed to the linear
layer as a :class:`QuantizedActivation`, which the scaled-mm kernels accept
directly without re-quantizing.

Adapted from the prototype in https://github.com/vllm-project/vllm/pull/42469.
"""

from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

__all__ = ["QuantizedActivation", "rms_norm_input_quant"]


@dataclass
class QuantizedActivation:
    """An activation that has already been quantized by a fused producer.

    Linear-layer kernels accept this in place of a plain tensor and skip
    their own input quantization step.
    """

    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def rms_norm_input_quant(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module | None,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Apply ``norm`` (with optional residual add) fused with the input
    quantization of ``linear``, when ``linear`` advertises one.

    Falls back to the plain (unfused) norm when ``linear`` is ``None`` or has
    no ``input_quant_key``, preserving the exact pre-fusion behavior.
    ``linear=None`` keeps call sites safe for decoder-layer subclasses that
    swap in modules without the expected projection attribute (e.g. Aria's
    MoE layer replacing ``mlp.gate_up_proj``).

    Returns ``(hidden_states, residual)`` like the fused-add RMSNorm path;
    ``hidden_states`` is a :class:`QuantizedActivation` when fusion applied.
    """
    quant_key: QuantKey | None = getattr(linear, "input_quant_key", None)
    if quant_key is None:
        if residual is None:
            return norm(x), x
        out, residual = norm(x, residual)
        return out, residual

    if quant_key == kFp8StaticTensorSym:
        return _rms_norm_fp8_static_per_tensor(norm, x, residual, linear)
    if quant_key == kFp8DynamicTokenSym:
        return _rms_norm_fp8_dynamic_per_token(norm, x, residual)

    raise NotImplementedError(
        f"rms_norm + {quant_key} manual fusion is not wired; "
        "supported: fp8 static per-tensor, fp8 dynamic per-token"
    )


def _rms_norm_fp8_static_per_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
    if residual is None:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_q,
            x,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
        residual = x
    else:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_q,
            x,
            residual,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
    return (
        QuantizedActivation(
            data=out_q,
            scale=linear.input_scale,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kFp8StaticTensorSym,
        ),
        residual,
    )


def _rms_norm_fp8_dynamic_per_token(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[QuantizedActivation, torch.Tensor]:
    # The fused kernel updates `residual` in place (like fused_add_rms_norm)
    # and returns per-token scales of shape (num_tokens, 1).
    if residual is None:
        residual = x
        out_q, scales = ops.rms_norm_dynamic_per_token_quant(
            x,
            norm.weight.data,
            norm.variance_epsilon,
            current_platform.fp8_dtype(),
        )
    else:
        out_q, scales = ops.rms_norm_dynamic_per_token_quant(
            x,
            norm.weight.data,
            norm.variance_epsilon,
            current_platform.fp8_dtype(),
            residual=residual,
        )
    return (
        QuantizedActivation(
            data=out_q,
            scale=scales,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kFp8DynamicTokenSym,
        ),
        residual,
    )
