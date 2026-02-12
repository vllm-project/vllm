# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_dynamic_per_token_quant(
    output: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [num_tokens]
    scale: torch.Tensor,  # [num_tokens, 1]
    epsilon: float,
    scale_ub: torch.Tensor | None = None,  # []
    residual: torch.Tensor | None = None,  # [num_tokens, hidden_size]
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    # only support fp8 quant for now
    fp8_dtype = _get_fp8_dtype()
    assert output.dtype in [fp8_dtype, torch.int8]
    assert output.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert output.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32

    if residual is not None:
        assert residual.dtype == input.dtype

    quant_dtype = output.dtype
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = _get_int8_min_max()
        min_scaling_factor = _get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)
        s_blk = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / qtype_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
                residual[tile_m, tile_n] = x_blk.to(residual.dtype)
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            if quant_dtype == torch.int8:
                s_inv_blk = 1.0 / s_blk[:, None]
                y_blk = x_blk * s_inv_blk
                y_blk = y_blk.round()
            else:
                y_blk = x_blk / s_blk[:, None]

            output[tile_m, tile_n] = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
                output.dtype
            )
