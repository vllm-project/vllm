# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.mimo_v2_mega_moe import (
    MiMoV2TPMegaMoEMethod,
    TPFallbackWeights,
)


def test_mimo_tp_fallback_matches_aiter_interleaved_layout():
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    experts, intermediate, hidden = 9, 256, 6144
    generator = torch.Generator().manual_seed(42)
    weight = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scales = torch.randint(
        0,
        255,
        (experts * 2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    layer = torch.nn.Module()
    layer.w13_weight = shuffle_weight(weight).view(torch.float4_e2m1fn_x2)
    layer.w13_weight_scale = shuffle_scale(scales)
    layer.w2_weight = torch.empty(0, dtype=torch.uint8)
    layer.w2_weight_scale = torch.empty(0, dtype=torch.uint8)
    fallback = TPFallbackWeights(layer)

    expected_weight = shuffle_weight(weight, is_guinterleave=True, gate_up=True)
    expected_scales = shuffle_scale(
        scales, experts_cnt=experts, is_guinterleave=True, gate_up=True
    )
    assert torch.equal(fallback.w1.view(torch.uint8), expected_weight)
    assert torch.equal(fallback.s1, expected_scales)
    assert fallback.w2 is layer.w2_weight
    assert fallback.s2 is layer.w2_weight_scale


def test_mimo_tp_mega_selects_padded_mega_or_fallback():
    method = object.__new__(MiMoV2TPMegaMoEMethod)
    method.capacity_rows = 8192
    method.max_rows = 256

    assert method.execution_path(0) == "empty"
    assert method.execution_path(1) == "mega_tp_ar"
    assert method.execution_path(256) == "mega_tp_ar"
    assert method.execution_path(257) == "aiter_a8w4"
    assert method.execution_path(8192) == "aiter_a8w4"
    with pytest.raises(ValueError, match="outside configured padded capacity"):
        method.execution_path(8193)
