# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test prefetch offloading correctness with DeepSeek V2 model."""

from ..utils import compare_two_settings


def test_prefetch_offload_deepseek():
    """Test prefetch CPU offloading with DeepSeek-V2-Lite.

    Compares outputs between:
    1. Baseline (no offloading)
    2. Prefetch offloading (group_size=8, num_in_group=2, prefetch_step=1)

    This tests prefetching-based offloading on a MoE model.
    """
    compare_two_settings(
        "deepseek-ai/DeepSeek-V2-Lite",
        [
            # Prefetch offloading configuration
            "--offload-group-size",
            "8",
            "--offload-num-in-group",
            "2",
            "--offload-prefetch-step",
            "1",
            # Selective offloading: only MoE expert weights
            "--offload-params",
            "w13_weight",
            "w2_weight",
            # torch.compile is automatically disabled when prefetch offloading
            # is enabled (via enable_if in @support_torch_compile decorator)
        ],
        [],  # Baseline: no offloading
    )
