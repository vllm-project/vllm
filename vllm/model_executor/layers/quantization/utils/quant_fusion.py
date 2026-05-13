# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


@dataclass
class QuantizedActivation:
    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def rms_norm_input_quant(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    quant_key: QuantKey | None = None,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    if quant_key is None:
        if residual is None:
            return norm(x), x
        out, residual = norm(x, residual)
        return out, residual
    if quant_key == kFp8StaticTensorSym:
        return _rms_norm_fp8_static_per_tensor(norm, x, residual, input_scale)
    raise NotImplementedError(f"rms_norm + {quant_key} not wired")


def _rms_norm_fp8_static_per_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    input_scale: torch.Tensor,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
    if residual is None:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_q,
            x,
            norm.weight.data,
            input_scale,
            norm.variance_epsilon,
        )
        residual = x
    else:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_q,
            x,
            residual,
            norm.weight.data,
            input_scale,
            norm.variance_epsilon,
        )
    qa = QuantizedActivation(
        data=out_q,
        scale=input_scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )
    return qa, residual
