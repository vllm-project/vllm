# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.platforms import current_platform


def apply_weights_fp8(
    scaled_mm_func: Callable[..., torch.Tensor],
    quant_fp8_func: QuantFP8,
    w: torch.Tensor,
    x: torch.Tensor,
    w_s: torch.Tensor,
    x_s: torch.Tensor,
    bias: torch.Tensor,
    x_s_ub: torch.Tensor | None,
    maybe_out_dtype: torch.dtype | None,
) -> torch.Tensor:
    #   ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_s computed from x.
    #   If static, layer.input_scale is scalar and x_s is input_scale.
    # View input as 2D matrix for fp8 methods
    x_2d = x.view(-1, x.shape[-1])
    output_shape = [*x.shape[:-1], w.shape[1]]

    out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype

    # If input not quantized
    # TODO(luka) remove this path if not used anymore
    x_2d_q = x_2d
    if x.dtype != current_platform.fp8_dtype():
        x_2d_q, x_s = quant_fp8_func(
            x_2d,
            x_s,
            x_s_ub,
        )

    return scaled_mm_func(
        A=x_2d_q,
        B=w,
        out_dtype=out_dtype,
        As=x_s,
        Bs=w_s,
        bias=bias,
        output_shape=output_shape,
    )
