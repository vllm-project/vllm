# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x Triton post-expert finalize kernel."""

import pytest
import torch

from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)


def _ref_finalize(rows, norm_weight, weights, eps, weight_first, gain_center=1.0):
    """``sum_k weights * norm(rows[:, k])`` in fp32."""
    x = rows.float()
    gain = 1 if norm_weight is None else norm_weight.float() + gain_center
    if weight_first:
        x = x * gain
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    if not weight_first:
        y = y * gain
    return (y * weights.float()[..., None]).sum(1)


_SHAPES = [(4096, 8), (8192, 9), (513, 3)]  # tiled, looped, masked-column kernel
_LAYOUTS = ["dense", "expert_map", "permuted"]
_WEIGHTS = [None, "first", "after"]


# A Latin square: every shape/layout, shape/weight and layout/weight pair once.
@pytest.mark.parametrize(
    "tokens,hidden,topk,layout,weight_mode",
    [
        (33, *_SHAPES[i], _LAYOUTS[j], _WEIGHTS[(i + j) % 3])
        for i in range(3)
        for j in range(3)
    ]
    + [(0, 4096, 8, "permuted", "after")],
)
@torch.inference_mode()
def test_triton_finalize_masks_remote_experts(
    tokens, hidden, topk, layout, weight_mode
):
    """Remote route workspaces can be NaN and must never enter normalization."""
    torch.manual_seed(42)
    routed = torch.randn(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, topk, device="cuda")
    ids = torch.arange(topk, device="cuda").expand(tokens, -1).contiguous()
    # Deliberately non-contiguous ownership, so range-based masking is wrong.
    expert_map = torch.arange(topk, device="cuda").roll(2)
    expert_map[1::2] = -1
    if layout == "dense":
        local = torch.ones_like(ids, dtype=torch.bool)
    else:
        local = expert_map[ids] >= 0
        if tokens:
            local[0] = False
            ids[0] = 1
    routed.masked_fill_(~local[..., None], float("nan"))
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) / 10
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    if layout == "permuted":
        idx = torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        idx = idx.view(tokens, topk).masked_fill(~local, -1)
        rows, kwargs = routed.view(tokens * topk, hidden), {"permuted_idx": idx}
    else:
        mapped = expert_map if layout == "expert_map" else None
        rows, kwargs = routed, {"topk_ids": ids, "expert_map": mapped}
    gain = None if weight_mode is None else norm_weight
    weight_first = weight_mode == "first"

    fused_post_expert_finalize(
        rows, gain, weights, 1e-8, 1.0, weight_first, output, **kwargs
    )

    expected = _ref_finalize(
        routed.masked_fill(~local[..., None], 0), gain, weights, 1e-8, weight_first
    ).to(output.dtype)
    torch.testing.assert_close(output, expected, atol=0.015625, rtol=0.01)
    assert torch.isfinite(output).all()
