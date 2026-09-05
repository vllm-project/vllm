# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER biased_grouped_topk expert-group count.

The AITER kernel is only instantiated for ``NUM_GRP`` in {1, 2, 4, 8}, so the
group count derived from ``num_experts`` has to be rounded to a supported
power of two that still divides ``num_experts``. Grouping is a no-op on this
path (``topk_group == num_expert_group``), so any such value routes
identically; the constraint is purely about which kernel exists.
"""

import pytest

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _aiter_get_num_expert_group,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

# Mirrors _AITER_MAX_EXPERTS_PER_GROUP in the router.
MAX_EXPERTS_PER_GROUP = 32
SUPPORTED_NUM_GRP = (1, 2, 4, 8)


@pytest.mark.parametrize(
    "num_experts",
    [
        1,
        8,
        16,
        32,  # exactly one full group
        33,
        64,
        72,
        96,
        128,
        129,
        160,
        192,
        256,
        257,
        320,
        384,
    ],
)
def test_group_count_divides_and_bounds_group_size(num_experts):
    g = _aiter_get_num_expert_group(num_experts)

    assert num_experts % g == 0, f"{g} does not divide {num_experts}"
    # The router asserts this unconditionally; rounding must never break it.
    assert num_experts // g <= MAX_EXPERTS_PER_GROUP


@pytest.mark.parametrize(
    "num_experts",
    [1, 8, 16, 32, 64, 72, 96, 128, 160, 192, 256, 320, 384],
)
def test_group_count_is_kernel_supported_when_one_fits(num_experts):
    """Whenever some supported NUM_GRP works, the chosen value must be one."""
    fits = [
        c
        for c in SUPPORTED_NUM_GRP
        if num_experts % c == 0 and num_experts // c <= MAX_EXPERTS_PER_GROUP
    ]
    if not fits:
        pytest.skip(f"no supported NUM_GRP fits {num_experts=}")

    assert _aiter_get_num_expert_group(num_experts) in fits


@pytest.mark.parametrize(
    ("num_experts", "expected"),
    [
        (8, 1),  # 8 experts fit in a single group
        (32, 1),  # exactly at the per-group limit
        (64, 2),
        (128, 4),
        (256, 8),
        (72, 8),  # naive ceil gives 3 -> unsupported, rounds up to 8
        (96, 8),  # naive ceil gives 3 -> unsupported, rounds up to 8
        (160, 8),  # naive ceil gives 5 -> unsupported, rounds up to 8
        (192, 8),  # naive ceil gives 6 -> unsupported, rounds up to 8
        # No supported NUM_GRP divides these while keeping the group size
        # within the limit, so the naive value is kept and the call site's
        # `topk >= num_expert_group` guard sends routing to the generic path.
        (320, 10),
        (384, 12),
        (33, 3),
        (129, 43),
    ],
)
def test_known_expert_counts(num_experts, expected):
    assert _aiter_get_num_expert_group(num_experts) == expected


@pytest.mark.parametrize("num_experts", [33, 129, 257, 320, 384])
def test_falls_back_when_no_supported_group_count_fits(num_experts):
    """With no usable NUM_GRP the naive divisor is kept, unrounded.

    Such a value has no instantiated kernel, but it is large enough that the
    ``topk >= num_expert_group`` check at the call site fails and routing
    falls back to the generic path rather than reaching AITER.
    """
    assert not any(
        num_experts % c == 0 and num_experts // c <= MAX_EXPERTS_PER_GROUP
        for c in SUPPORTED_NUM_GRP
    )

    g = _aiter_get_num_expert_group(num_experts)
    assert num_experts % g == 0
    assert num_experts // g <= MAX_EXPERTS_PER_GROUP
    assert g not in SUPPORTED_NUM_GRP


@pytest.mark.parametrize("num_experts", [72, 96, 160, 192])
def test_rounding_only_moves_to_supported_values(num_experts):
    """The naive ceil-divide result is unsupported for these counts."""
    naive = max(1, -(-num_experts // MAX_EXPERTS_PER_GROUP))
    while num_experts % naive != 0:
        naive += 1
    assert naive not in SUPPORTED_NUM_GRP

    assert _aiter_get_num_expert_group(num_experts) in SUPPORTED_NUM_GRP
