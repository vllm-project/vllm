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
    fp8_max_t = torch.tensor(fp8_max, dtype=torch.float32, device=input.device)
    s_1_t = torch.tensor(1.0, dtype=torch.float32, device=input.device)
    s_512_t = torch.tensor(512.0, dtype=torch.float32, device=input.device)
    min_scaling_factor_t = s_1_t / (fp8_max_t * s_512_t)
    scale.zero_()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        x_blk = input[tile_m, tile_n]
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(dtype=torch.float32)
        hl.atomic_max(scale, [tile_m, 0], s_blk)

    hl.barrier()

    for tile_m in hl.tile(num_tokens):
        s_blk = scale[tile_m, :]
        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        fp8_max_s = hl.load(fp8_max_t, [])
        s_blk = s_blk / fp8_max_s
        min_scaling_factor_s = hl.load(min_scaling_factor_t, [])
        s_blk = s_blk.clamp(min=min_scaling_factor_s)
        scale[tile_m, :] = s_blk

    hl.barrier()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        x_blk = input[tile_m, tile_n]
        s_blk = scale[tile_m, :]
        y_blk = x_blk.to(torch.float32) / s_blk

        output[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(output.dtype)
