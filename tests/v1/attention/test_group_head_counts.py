# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A KV cache group may hold layers with different query head counts — e.g.
Laguna, which overrides the head count per layer. Scheduler metadata describes
an attention scratchpad whose layout depends on that count, so layers differing
from the model-wide default need metadata of their own. Sharing one blob makes
the wider layers index past the end of the split-KV reduction scratchpad,
corrupting memory.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackendImpl,
    CPUAttentionMetadataBuilder,
)


def _layers(layer_num_heads: list[int]):
    """Stand-in attention layers, one per head count, as one KV cache group."""
    return {
        f"layer_{i}": SimpleNamespace(
            impl=MagicMock(spec=CPUAttentionBackendImpl, num_heads=num_heads)
        )
        for i, num_heads in enumerate(layer_num_heads)
    }


@pytest.mark.parametrize(
    "layer_num_heads,default_num_heads,expected",
    [
        # Uniform: every layer matches the default, so no extra metadata.
        ([8, 8], 8, set()),
        # Laguna-style: the wider layers each need their own metadata.
        ([8, 16, 8, 16], 8, {16}),
        # The default need not be the most common count.
        ([16, 32], 8, {16, 32}),
    ],
)
def test_cpu_group_extra_num_heads(layer_num_heads, default_num_heads, expected):
    layers = _layers(layer_num_heads)
    builder = SimpleNamespace(
        vllm_config=None,
        layer_names=list(layers),
        num_heads=default_num_heads,
    )
    with patch(
        "vllm.v1.attention.backends.cpu_attn.get_layers_from_vllm_config",
        return_value=layers,
    ):
        extra = CPUAttentionMetadataBuilder._group_extra_num_heads(builder)
    assert extra == expected
