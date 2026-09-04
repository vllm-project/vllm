# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER biased_grouped_topk expert-group count.

The AITER kernel is only instantiated for ``NUM_GRP`` in {1, 2, 4, 8}, so the
group count derived from ``num_experts`` has to be rounded to a supported
value that still divides ``num_experts``. Grouping is a no-op on this path
(``topk_group == num_expert_group``), so any such value routes identically;
the constraint is purely about which kernel exists.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    AITER_MAX_EXPERTS_PER_GROUP as MAX_EXPERTS_PER_GROUP,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    AITER_SUPPORTED_NUM_GRP as SUPPORTED_NUM_GRP,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _aiter_can_use_biased_grouped_topk,
    _aiter_get_num_expert_group,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.mark.parametrize(
    ("num_experts", "expected"),
    [
        (8, 1),  # fits in a single group
        (32, 1),  # exactly at the per-group limit
        (64, 2),
        (128, 4),
        (256, 8),
        (72, 8),  # naive ceil gives 3 -> unsupported, rounds up to 8
        (96, 8),  # naive ceil gives 3 -> unsupported, rounds up to 8
        (160, 8),  # naive ceil gives 5 -> unsupported, rounds up to 8
        (192, 8),  # naive ceil gives 6 -> unsupported, rounds up to 8
        # No supported NUM_GRP divides these within the group-size limit, so
        # the naive value is kept and the call site declines the kernel.
        (33, 3),
        (129, 43),
        (320, 10),
        (384, 12),
    ],
)
def test_known_expert_counts(num_experts, expected):
    g = _aiter_get_num_expert_group(num_experts)

    assert g == expected
    # The router asserts both unconditionally; rounding must never break them.
    assert num_experts % g == 0
    assert num_experts // g <= MAX_EXPERTS_PER_GROUP


@pytest.mark.parametrize(
    ("num_experts", "supported"),
    [
        (8, True),
        (32, True),
        (64, True),
        (72, True),
        (96, True),
        (128, True),
        (160, True),
        (192, True),
        (256, True),
        (33, False),
        (129, False),
        (257, False),
        (320, False),
        (384, False),
    ],
)
def test_guard_admits_exactly_the_shapes_with_a_kernel(num_experts, supported):
    """An unsupported group count must never reach the kernel.

    ``topk >= g`` alone does not stop it: num_experts=33 gives g=3, which
    passes at topk >= 3. Membership in SUPPORTED_NUM_GRP is what declines.
    """
    g = _aiter_get_num_expert_group(num_experts)
    assert (g in SUPPORTED_NUM_GRP) == supported

    # Ample topk, so only the membership check can decline.
    assert _aiter_can_use_biased_grouped_topk(num_experts, topk=64) == supported
    if supported:
        assert _aiter_can_use_biased_grouped_topk(num_experts, topk=g)
        assert not _aiter_can_use_biased_grouped_topk(num_experts, topk=g - 1)


@pytest.mark.parametrize(
    "num_experts",
    [
        64,  # g == 2, not rounded -- covers the no-op claim itself
        128,  # g == 4, not rounded
        256,  # g == 8, not rounded
        96,  # naive 3 -> rounded to 8
        192,  # naive 6 -> rounded to 8
    ],
)
def test_rounded_group_count_routes_like_the_reference(num_experts):
    """The rounded group count must route identically to ``grouped_topk``.

    Gating is continuous random, so exact score ties -- the one place the two
    may legitimately differ -- have measure zero.
    """
    torch.manual_seed(num_experts)
    device = "cuda"
    topk = 8
    num_tokens = 83  # not a multiple of any warp/tile size
    g = _aiter_get_num_expert_group(num_experts)
    assert _aiter_can_use_biased_grouped_topk(num_experts, topk)

    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)

    ref_weights, ref_ids = grouped_topk(
        hidden_states=torch.empty(num_tokens, 1, device=device),
        gating_output=gating,
        topk=topk,
        renormalize=True,
        num_expert_group=g,
        topk_group=g,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )

    weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
    ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    rocm_aiter_ops.biased_grouped_topk(
        gating,
        bias,
        weights,
        ids,
        num_expert_group=g,
        topk_group=g,
        need_renorm=True,
    )

    # Neither side promises an order within a token's top-k, so sort by expert
    # id and carry the weights through the same permutation -- sorting the two
    # independently would not catch a weight attached to the wrong expert.
    order = ids.argsort(dim=-1)
    ref_order = ref_ids.argsort(dim=-1)
    torch.testing.assert_close(
        ids.gather(1, order), ref_ids.to(torch.int32).gather(1, ref_order)
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        ref_weights.gather(1, ref_order),
        atol=1e-4,
        rtol=1e-4,
    )
