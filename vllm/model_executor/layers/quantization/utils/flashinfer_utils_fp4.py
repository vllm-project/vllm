# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def apply_mxfp4_flashinfer_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    size_n: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.utils.flashinfer import (
        flashinfer_mxfp4_quantize,
        flashinfer_scaled_fp4_mm,
    )

    x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    x_fp4, x_scale = flashinfer_mxfp4_quantize(x)

    out = flashinfer_scaled_fp4_mm(
        x_fp4,
        weight,
        x_scale,
        weight_scale,
        alpha=None,
        out_dtype=input.dtype,
        backend="cute-dsl",
        block_size=32,
        use_nvfp4=False,
    )

    if bias is not None:
        out = out + bias
    return out.view(out_shape)
