# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


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
    linear: torch.nn.Module,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    if getattr(linear, "input_quant_key", None) is None:
        if residual is None:
            return norm(x), x
        out, residual = norm(x, residual)
        return out, residual
    return linear.quant_method.rms_norm_quantize_input(linear, norm, x, residual)
