# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test prefetch offloading correctness with Llama model."""

from ..utils import compare_two_settings


def test_prefetch_offload_llama():
    """Test prefetch CPU offloading with Llama-3.2-1B-Instruct.

    Compares outputs between:
    1. Baseline (no offloading)
    2. Prefetch offloading (group_size=8, num_in_group=2, prefetch_step=1)

    This tests prefetching-based offloading on a dense model.
    """
    compare_two_settings(
        "meta-llama/Llama-3.2-1B-Instruct",
        [
            # Prefetch offloading configuration
            "--offload-group-size",
            "8",
            "--offload-num-in-group",
            "2",
            "--offload-prefetch-step",
            "1",
            # Selective offloading: only MLP weights
            "--offload-params",
            "gate_up_proj",
            "down_proj",
        ],
        [],  # Baseline: no offloading
    )
