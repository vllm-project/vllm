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
)
def dynamic_per_tensor_scaled_fp8_quant(
    output: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [1]
):
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert output.shape == input.shape
    assert scale.shape[0] == 1
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert output.stride()[-1] == 1

    _, fp8_max = get_fp8_min_max()
    fp8_max_t = torch.tensor(fp8_max, dtype=torch.float32, device=input.device)

    scale.zero_()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        abs_value = torch.abs(input[tile_m, tile_n])
        # multiple reduction dimensions not supported yet
        max_value_per_row = torch.amax(abs_value, dim=1)
        max_value = torch.amax(max_value_per_row).to(dtype=torch.float32)
        fp8_max_s = hl.load(fp8_max_t, [])
        hl.atomic_max(scale, [0], max_value / fp8_max_s)

    hl.barrier()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        inv_scale = (1.0 / scale[0]).to(dtype=torch.float32)
        x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
        output[tile_m, tile_n] = (x_blk * inv_scale).to(output.dtype)
