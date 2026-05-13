# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def manual_input_quant_enabled() -> bool:
    return os.environ.get("VLLM_HOIST_INPUT_QUANT", "0") == "1"


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
    if residual is None:
        residual = x
        out = norm(x)
    else:
        out, residual = norm(x, residual)

    if getattr(linear, "input_quant_key", None) is None:
        return out, residual
    return linear.quant_method.quantize_input(linear, out), residual
