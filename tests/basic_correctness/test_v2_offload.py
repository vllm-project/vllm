# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test V2 offloading correctness with DeepSeek V2 model."""

from ..utils import compare_two_settings


def test_v2_offload_deepseek():
    """Test V2 CPU offloading with DeepSeek-V2-Lite.

    Compares outputs between:
    1. Baseline (no offloading)
    2. V2 offloading (group_size=8, num_in_group=2, prefetch_step=1)

    This tests the advanced offloading with prefetching on a MoE model.
    """
    compare_two_settings(
        "deepseek-ai/DeepSeek-V2-Lite",
        [
            # V2 offloading configuration
            "--offload-group-size",
            "8",
            "--offload-num-in-group",
            "2",
            "--offload-prefetch-step",
            "1",
            # torch.compile is automatically disabled when V2 offloading is
            # enabled (via enable_if in @support_torch_compile decorator)
        ],
        [],  # Baseline: no offloading
    )
