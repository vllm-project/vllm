# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config import CacheConfig


@pytest.mark.parametrize("mamba_block_size", [8, 16, 512])
def test_mamba_block_size_multiple_of_eight_accepted(
    mamba_block_size: int,
) -> None:
    cfg = CacheConfig(mamba_block_size=mamba_block_size)
    assert cfg.mamba_block_size == mamba_block_size
    assert cfg.user_specified_mamba_block_size is True


def test_mamba_block_size_none_accepted() -> None:
    cfg = CacheConfig()
    assert cfg.mamba_block_size is None
    assert cfg.user_specified_mamba_block_size is False


@pytest.mark.parametrize("mamba_block_size", [1, 7, 12, 15])
def test_mamba_block_size_not_multiple_of_eight_rejected(
    mamba_block_size: int,
) -> None:
    with pytest.raises(ValidationError, match="multiple of 8"):
        CacheConfig(mamba_block_size=mamba_block_size)
