# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the MoE expert LRU cache (--moe-expert-cache-size).

Runs two vllm serve instances side-by-side via compare_two_settings:
  - baseline: standard MoE (all experts on GPU)
  - cache:    expert LRU cache enabled with a small GPU buffer

Token outputs must match exactly.
"""

import pytest

from ..utils import compare_two_settings

_MOE_MODEL = "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"


@pytest.mark.parametrize("cache_size", [4, 16])
def test_moe_expert_cache_correctness(cache_size: int) -> None:
    """Output tokens from the cache path must match the no-cache baseline."""
    compare_two_settings(
        model=_MOE_MODEL,
        arg1=["--enforce-eager"],
        arg2=[
            "--enforce-eager",
            "--moe-expert-cache-size",
            str(cache_size),
        ],
    )


def test_moe_expert_cache_disabled_by_default() -> None:
    """Verify that the default (cache_size=0) leaves the existing path intact."""
    compare_two_settings(
        model=_MOE_MODEL,
        arg1=["--enforce-eager"],
        arg2=["--enforce-eager", "--moe-expert-cache-size", "0"],
    )
