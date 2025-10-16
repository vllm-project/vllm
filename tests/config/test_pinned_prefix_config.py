# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import CacheConfig


def test_invalid_cap_ratio_over_one():
    # pinned_prefix_cap_ratio > 1.0 should raise ValueError (no HF access)
    with pytest.raises(
        ValueError, match=r"pinned_prefix_cap_ratio.*within \[0.0, 1.0\]"
    ):
        _ = CacheConfig(pinned_prefix_cap_ratio=1.5)


def test_negative_cap_ratio_raises():
    # negative value should raise because ratio must be within [0, 1]
    with pytest.raises(
        ValueError, match=r"pinned_prefix_cap_ratio.*within \[0.0, 1.0\]"
    ):
        _ = CacheConfig(pinned_prefix_cap_ratio=-0.1)
