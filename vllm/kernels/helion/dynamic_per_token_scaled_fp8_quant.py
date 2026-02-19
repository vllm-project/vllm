# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def dynamic_per_token_scaled_fp8_quant(
    output: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
):
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert output.shape == input.shape
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert output.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            # hl.atomic_max(scale, [tile_m, 0], s_blk)
            # scale[tile_m, 0] = torch.maximum(scale[tile_m, 0], s_blk)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk * (1.0 / s_blk[:, None])

            output[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(output.dtype)
