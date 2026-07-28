# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for hierarchical MoE expert staging vs baseline."""

from ..utils import compare_two_settings


def test_hierarchical_offload_moe():
    """Compare hierarchical expert staging against a no-offload baseline.

    Uses DeepSeek-V2-Lite with a small device slot count so experts must
    stage through pinned RAM. Skipped automatically when the model/hardware
    is unavailable via compare_two_settings infrastructure.
    """
    compare_two_settings(
        "deepseek-ai/DeepSeek-V2-Lite",
        [
            "--offload-backend",
            "hierarchical",
            "--tier-num-slots",
            "16",
            "--tier-ram-gb",
            "4",
            "--tier-policy",
            "quality",
            "--max-model-len",
            "512",
            "--enforce-eager",
        ],
        [],  # baseline
    )
