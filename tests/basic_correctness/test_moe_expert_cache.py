# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the MoE expert LRU cache (--moe-expert-cache-size).

Runs two vllm serve instances side-by-side via compare_two_settings:
  - baseline: standard MoE (all experts on GPU)
  - cache:    expert LRU cache enabled with a small GPU buffer

Token outputs must match exactly (the default ``token`` split is bit-exact).

The tiny Qwen3-MoE checkpoint (8 experts, top-2, 6 layers) keeps the main
cases runnable on small CI devices and covers the unquantized path; the FP8
case needs the full DeepSeek-Coder-V2-Lite checkpoint (~16 GB weights plus a
baseline server), so it runs only where a large device is available.
"""

import pytest
import torch

from ..utils import compare_two_settings

# 8 experts, top-2: cache 3 sits just above the one-token floor and forces
# splitting plus steady eviction; cache 4 (50%) exercises plain eviction.
_TINY_MOE_MODEL = "nm-testing/tinysmokeqwen3moe"

# fp8 quant_method with shared experts; routes top-6 of 64.
_FP8_MOE_MODEL = "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8"

_COMMON_ARGS = ["--max-model-len", "2048", "--dtype", "bfloat16"]

_HAS_LARGE_GPU = (
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory >= 40 * 2**30
)


@pytest.mark.parametrize("cache_size", [3, 4])
def test_moe_expert_cache_correctness(cache_size: int) -> None:
    """Output tokens from the cache path must match the no-cache baseline."""
    compare_two_settings(
        model=_TINY_MOE_MODEL,
        arg1=_COMMON_ARGS,
        arg2=_COMMON_ARGS + ["--moe-expert-cache-size", str(cache_size)],
    )


def test_moe_expert_cache_disabled_by_default() -> None:
    """Verify that the default (cache_size=0) leaves the existing path intact."""
    compare_two_settings(
        model=_TINY_MOE_MODEL,
        arg1=_COMMON_ARGS,
        arg2=_COMMON_ARGS + ["--moe-expert-cache-size", "0"],
    )


def test_moe_expert_cache_with_enforce_eager() -> None:
    """Cache with explicit --enforce-eager (no CUDA graphs)."""
    compare_two_settings(
        model=_TINY_MOE_MODEL,
        arg1=_COMMON_ARGS + ["--enforce-eager"],
        arg2=_COMMON_ARGS + ["--enforce-eager", "--moe-expert-cache-size", "4"],
    )


@pytest.mark.skipif(
    not _HAS_LARGE_GPU,
    reason="FP8 MoE e2e needs ~16 GB of weights plus KV for two servers; "
    "requires a >=40 GiB device",
)
def test_moe_expert_cache_fp8_correctness() -> None:
    """FP8 path: slot-indexed scales must reproduce baseline tokens exactly."""
    compare_two_settings(
        model=_FP8_MOE_MODEL,
        arg1=_COMMON_ARGS,
        arg2=_COMMON_ARGS + ["--moe-expert-cache-size", "16"],
    )
