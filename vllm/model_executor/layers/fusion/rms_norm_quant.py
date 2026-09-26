# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual RMSNorm + input-quant fusion (RFC #43224, issue #43500).

Producer side of the QuantizedActivation contract for RMSNorm. A decoder
layer asks :func:`rms_norm_input_quant` to normalize its hidden states; when
the downstream linear advertises an ``input_quant_key`` its kernel can consume
(wired by ``expose_input_quant_key``), the norm and the activation
quantization run as a single fused kernel. The result is handed to the linear
as a :class:`QuantizedActivation`, which the scaled-mm kernels read directly
via ``as_quantized_activation``, skipping their own input requantization.

This replaces the ``RMSNormQuantFusionPass`` compiler fusion with an explicit
call site in model code, using the same fused kernels the pass emitted, so
there is no perf delta versus compile-time fusion.

Currently wired for FP8 static per-tensor (``kFp8StaticTensorSym``) -- the key
the upstream scaled-mm FP8 kernels (cutlass/flashinfer) advertise. Other keys
(dynamic per-token, per-block, nvfp4) take the plain-norm fallback until their
producer kernels are added in follow-ups.
"""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


def rms_norm_input_quant(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module | None,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Apply ``norm`` (with optional residual add) fused with the input
    quantization of ``linear``, when ``linear``'s kernel advertises a key.

    Falls back to the plain (unfused) norm when ``linear`` is ``None`` or has
    no consumable ``input_quant_key``, preserving the exact pre-fusion
    behavior. ``linear=None`` keeps call sites safe for decoder-layer
    subclasses that swap in modules without the expected projection attribute
    (e.g. Aria's MoE layer replacing ``mlp.gate_up_proj``).

    Returns ``(hidden_states, residual)`` like the fused-add RMSNorm path;
    ``hidden_states`` is a :class:`QuantizedActivation` when fusion applied.
    """
    quant_key = getattr(linear, "input_quant_key", None)
    if quant_key == kFp8StaticTensorSym:
        return _rms_norm_fp8_static_per_tensor(norm, x, residual, linear)

    # No consumable key (or unsupported one) -> plain norm, exact prior path.
    if residual is None:
        return norm(x), x
    out, residual = norm(x, residual)
    return out, residual


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
