# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Quark OCP MXFP4 weights loaded into AITER MoE layouts."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.inkling.amd.moe import InklingMoE


def _fake_moe() -> tuple[InklingMoE, SimpleNamespace]:
    moe = InklingMoE.__new__(InklingMoE)
    nn.Module.__init__(moe)
    moe.n_routed_experts = 4
    routed = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_rank=1, tp_size=2)
        ),
        # Logical intermediate per rank is 3; AITER pads it to 4.
        w13_weight=nn.Parameter(torch.zeros(4, 8, 2, dtype=torch.uint8), False),
        w2_weight=nn.Parameter(torch.zeros(4, 5, 3, dtype=torch.uint8), False),
        w13_weight_scale=nn.Parameter(torch.ones(4, 8, 2, dtype=torch.uint8), False),
        w2_weight_scale=nn.Parameter(torch.ones(4, 5, 3, dtype=torch.uint8), False),
    )
    moe.experts = SimpleNamespace(routed_experts=routed)
    moe._local_expert_slots = (  # type: ignore[method-assign]
        lambda: dict(enumerate(range(4)))
    )
    return moe, routed


def test_loads_logical_tp_shards_and_preserves_aiter_padding():
    moe, routed = _fake_moe()

    w13 = torch.arange(4 * 12 * 2, dtype=torch.uint8).view(4, 12, 2)
    w2 = torch.arange(4 * 5 * 4, dtype=torch.uint8).view(4, 5, 4)
    moe.load_expert_weight("experts.w13_weight", w13)
    moe.load_expert_weight("experts.w2_weight", w2)

    # TP rank 1 consumes the second six interleaved gate/up rows. AITER keeps
    # one padding row in each destination half and one padding w2 column.
    torch.testing.assert_close(routed.w13_weight[:, :3], w13[:, 6:12:2])
    torch.testing.assert_close(routed.w13_weight[:, 4:7], w13[:, 7:12:2])
    assert torch.count_nonzero(routed.w13_weight[:, 3]) == 0
    assert torch.count_nonzero(routed.w13_weight[:, 7]) == 0
    torch.testing.assert_close(routed.w2_weight[:, :, :2], w2[:, :, 2:4])
    assert torch.count_nonzero(routed.w2_weight[:, :, 2]) == 0


def test_unflattens_quark_scales_and_preserves_scale_padding():
    moe, routed = _fake_moe()

    flat_w13_scale = torch.arange(4 * 12 * 2, dtype=torch.uint8).view(4 * 12, 2)
    flat_w2_scale = torch.arange(4 * 5 * 4, dtype=torch.uint8).view(4 * 5, 4)
    moe.load_expert_weight("experts.w13_weight_scale", flat_w13_scale)
    moe.load_expert_weight("experts.w2_weight_scale", flat_w2_scale)

    w13_scale = flat_w13_scale.view(4, 12, 2)
    w2_scale = flat_w2_scale.view(4, 5, 4)
    torch.testing.assert_close(routed.w13_weight_scale[:, :3], w13_scale[:, 6:12:2])
    torch.testing.assert_close(routed.w13_weight_scale[:, 4:7], w13_scale[:, 7:12:2])
    assert torch.all(routed.w13_weight_scale[:, 3] == 1)
    assert torch.all(routed.w13_weight_scale[:, 7] == 1)
    torch.testing.assert_close(routed.w2_weight_scale[:, :, :2], w2_scale[:, :, 2:4])
    assert torch.all(routed.w2_weight_scale[:, :, 2] == 1)
