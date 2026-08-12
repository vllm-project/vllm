# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

from .quant_activation import QuantizedActivation


def maybe_allocate_fp8_block_quant(
    x: torch.Tensor,
    *linears: torch.nn.Module,
) -> QuantizedActivation | None:
    """Allocate a shared DeepGEMM FP8 activation for compatible consumers."""
    if (
        not linears
        or DeepGemmQuantScaleFMT.from_oracle() != DeepGemmQuantScaleFMT.UE8M0
    ):
        return None

    quant_key = getattr(linears[0], "input_quant_key", None)
    if quant_key != kFp8Dynamic128Sym or any(
        getattr(linear, "input_quant_key", None) != quant_key for linear in linears[1:]
    ):
        return None

    group_size = quant_key.scale.group_shape.col
    if x.shape[-1] % group_size != 0:
        return None

    num_rows = x.numel() // x.shape[-1]
    num_groups = x.shape[-1] // group_size
    num_scale_packs = (num_groups + 3) // 4
    aligned_rows = ((num_rows + 3) // 4) * 4
    data = torch.empty_like(x, dtype=current_platform.fp8_dtype())
    scale = torch.empty_strided(
        (num_rows, num_scale_packs),
        (1, aligned_rows),
        dtype=torch.int32,
        device=x.device,
    )
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=quant_key,
    )
