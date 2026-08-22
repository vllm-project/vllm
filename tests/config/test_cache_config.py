# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config.cache import CacheConfig


def test_num_gpu_blocks_override_none_is_default():
    assert CacheConfig().num_gpu_blocks_override is None
    assert CacheConfig(num_gpu_blocks_override=None).num_gpu_blocks_override is None


def test_num_gpu_blocks_override_positive_is_accepted():
    assert CacheConfig(num_gpu_blocks_override=16).num_gpu_blocks_override == 16


@pytest.mark.parametrize("bad", [0, -1, -16])
def test_num_gpu_blocks_override_non_positive_raises(bad):
    # Regression test for https://github.com/vllm-project/vllm/issues/43842:
    # a non-positive override previously reached BlockPool.__init__ and tripped
    # a bare assertion; it must now be rejected at config construction time.
    with pytest.raises(ValidationError):
        CacheConfig(num_gpu_blocks_override=bad)
