# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.cache import CacheConfig


def test_block_size_none_uses_default():
    config = CacheConfig()
    assert config.block_size == CacheConfig.DEFAULT_BLOCK_SIZE
    assert config.user_specified_block_size is False


def test_block_size_positive_is_accepted():
    config = CacheConfig(block_size=32)
    assert config.block_size == 32
    assert config.user_specified_block_size is True


@pytest.mark.parametrize("invalid_block_size", [0, -1, -16])
def test_block_size_non_positive_raises(invalid_block_size):
    with pytest.raises(ValueError, match="block_size"):
        CacheConfig(block_size=invalid_block_size)
