# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import pytest

from vllm.config import SpeculativeConfig


def _config(method: str, num_speculative_tokens: int) -> SpeculativeConfig:
    config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=num_speculative_tokens
    )
    config.method = cast(Any, method)
    return config


@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("shallow, deep", [(3, 9), (1, 128)])
def test_parallel_drafting_hash_includes_speculative_depth(
    method: str, shallow: int, deep: int
):
    """Parallel drafting depth changes static query shapes."""
    assert (
        _config(method, shallow).compile_factors()
        != _config(method, deep).compile_factors()
    )


@pytest.mark.parametrize("method", ["ngram", "eagle"])
def test_sequential_drafting_hash_ignores_speculative_depth(method: str):
    """Sequential drafting depth is a loop count, not a static query shape."""
    assert _config(method, 3).compile_factors() == _config(method, 9).compile_factors()
