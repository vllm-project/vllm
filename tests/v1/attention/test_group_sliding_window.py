# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A KV cache group may hold both windowed and global layers — e.g. Gemma-3 with
the hybrid KV cache manager disabled, where the sliding-window layers are
promoted to full-attention *storage* and merged into one group whose spec still
records the window. Backends must take the window from the layers, never from
that group spec, or the global layers get windowed too and silently lose access
to everything older than the window.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackendImpl,
    CPUAttentionMetadataBuilder,
)


def _layers(layer_windows: list[int]):
    """Stand-in attention layers, one per window, as one KV cache group."""
    return {
        f"layer_{i}": SimpleNamespace(
            impl=MagicMock(spec=CPUAttentionBackendImpl, sliding_window=window)
        )
        for i, window in enumerate(layer_windows)
    }


@pytest.mark.parametrize(
    "layer_windows,expected",
    [
        ([512, 512], 512),  # uniform sliding window -> shared by the group
        ([-1, -1], -1),  # all global -> no window
        ([512, -1], -1),  # mixed -> the group cannot assume either window
    ],
)
def test_cpu_group_sliding_window(layer_windows, expected):
    layers = _layers(layer_windows)
    builder = SimpleNamespace(vllm_config=None, layer_names=list(layers))
    with patch(
        "vllm.v1.attention.backends.cpu_attn.get_layers_from_vllm_config",
        return_value=layers,
    ):
        window = CPUAttentionMetadataBuilder._group_sliding_window(builder)
    assert window == expected
