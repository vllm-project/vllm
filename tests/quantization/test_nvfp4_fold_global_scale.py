# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-shard NVFP4 weight global scales folded into the FP8 block scales."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    fold_nvfp4_shard_global_scales,
)

E4M3 = torch.float8_e4m3fn
# One extra E4M3 rounding per block scale: relative error <= 2**-4.
RTOL = 2**-4


def _multipliers(block_scales, global_scales, sizes, divisor):
    """What a kernel multiplies each row's FP4 codes by."""
    out, start = [], 0
    for size, g in zip(sizes, global_scales):
        rows = block_scales[start : start + size].float()
        out.append(rows / g if divisor else rows * g)
        start += size
    return torch.cat(out)


@pytest.mark.parametrize("divisor", [True, False])
@pytest.mark.parametrize(
    "sizes, global_scales",
    [
        ([256, 64, 64], [4640.0, 1944.0, 2288.0]),
        # Kimi-Linear in_proj_qkvgfab at TP=1 (q, k, v, b, f_a)
        ([4096, 4096, 4096, 32, 128], [5152.0, 5344.0, 4640.0, 1944.0, 2288.0]),
    ],
)
def test_linear_fold_keeps_per_shard_dequantization(sizes, global_scales, divisor):
    torch.manual_seed(0)
    if not divisor:  # ModelOpt / Quark store multipliers (amax / 2688)
        global_scales = [1.0 / g for g in global_scales]
    block = (torch.rand(sum(sizes), 8) * 400 + 1).to(E4M3)
    ref = _multipliers(block, global_scales, sizes, divisor)

    folded = block.clone()
    common = fold_nvfp4_shard_global_scales(
        folded, torch.tensor(global_scales), sizes, divisor=divisor
    )

    expected = min(global_scales) if divisor else max(global_scales)
    assert common.item() == pytest.approx(expected)
    got = folded.float() / common if divisor else folded.float() * common
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=0)
    # Collapsing to one scale without folding, as before, is far off.
    stale = (
        block.float() / max(global_scales)
        if divisor
        else block.float() * max(global_scales)
    )
    assert not torch.allclose(stale, ref, rtol=0.1)


def test_equal_global_scales_are_left_alone():
    block = (torch.rand(64, 4) * 400 + 1).to(E4M3)
    folded = block.clone()
    common = fold_nvfp4_shard_global_scales(
        folded, torch.tensor([3000.0, 3000.0]), [32, 32], divisor=True
    )
    assert common.item() == 3000.0
    assert torch.equal(folded.view(torch.uint8), block.view(torch.uint8))


def test_unknown_layout_falls_back_to_max():
    block = (torch.rand(64, 4) * 400 + 1).to(E4M3)
    folded = block.clone()
    # Partition sizes do not cover the rows: do not touch the scales.
    common = fold_nvfp4_shard_global_scales(
        folded, torch.tensor([3000.0, 1500.0]), [32, 16], divisor=True
    )
    assert common.item() == 3000.0
    assert torch.equal(folded.view(torch.uint8), block.view(torch.uint8))
