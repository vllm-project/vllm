# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config.cache import CacheConfig


def test_num_gpu_blocks_override_none_is_default() -> None:
    """Verify that None (the default) is accepted and leaves the field as None."""
    cfg = CacheConfig()
    assert cfg.num_gpu_blocks_override is None


@pytest.mark.parametrize("value", [1, 16, 1024])
def test_num_gpu_blocks_override_positive_is_accepted(value: int) -> None:
    """Verify that positive integers are accepted."""
    cfg = CacheConfig(num_gpu_blocks_override=value)
    assert cfg.num_gpu_blocks_override == value


@pytest.mark.parametrize("value", [0, -1, -16])
def test_num_gpu_blocks_override_non_positive_raises(value: int) -> None:
    """Verify that 0 and negative integers raise ValidationError."""
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        CacheConfig(num_gpu_blocks_override=value)
