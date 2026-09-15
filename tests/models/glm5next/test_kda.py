# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention


def _make_attention():
    attention = object.__new__(Glm5NextLinearAttention)
    weights = [
        torch.nn.Parameter(torch.zeros((2, 1, 3))),
        torch.nn.Parameter(torch.ones((2, 1, 3))),
        torch.nn.Parameter(torch.full((2, 1, 3), 2.0)),
    ]
    object.__setattr__(attention, "q_conv1d", SimpleNamespace(weight=weights[0]))
    object.__setattr__(attention, "k_conv1d", SimpleNamespace(weight=weights[1]))
    object.__setattr__(attention, "v_conv1d", SimpleNamespace(weight=weights[2]))
    object.__setattr__(attention, "_merged_conv_weight", None)
    return attention, weights


def test_merged_conv_weight_cached_when_unchanged():
    attention, _weights = _make_attention()

    merged_first = attention._get_merged_conv_weight()
    merged_second = attention._get_merged_conv_weight()

    assert merged_first is merged_second


def test_merged_conv_weight_rebuilds_after_load_weight_invalidation():
    attention, weights = _make_attention()

    merged_before = attention._get_merged_conv_weight()

    with torch.no_grad():
        weights[1].data.copy_(torch.full_like(weights[1], 3.0))

    attention.invalidate_merged_conv_weight()

    merged_after = attention._get_merged_conv_weight()

    assert merged_before is not merged_after
    torch.testing.assert_close(merged_after[2:4], torch.full((2, 3), 3.0))