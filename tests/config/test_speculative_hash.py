# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import pytest

from vllm.config import SpeculativeConfig


def _config(
    method: str, num_speculative_tokens: int, *, parallel_drafting: bool
) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = cast(Any, method)
    config.num_speculative_tokens = num_speculative_tokens
    config.parallel_drafting = parallel_drafting
    config.draft_model_config = cast(Any, None)
    return config


@pytest.mark.parametrize("method", ["dflash", "dspark", "eagle3", "draft_model"])
@pytest.mark.parametrize("shallow, deep", [(3, 9), (1, 128)])
def test_parallel_drafting_hash_includes_speculative_depth(
    method: str, shallow: int, deep: int
):
    """Parallel drafting depth changes static query shapes."""
    assert (
        _config(method, shallow, parallel_drafting=True).compute_hash()
        != _config(method, deep, parallel_drafting=True).compute_hash()
    )


@pytest.mark.parametrize("method", ["mtp", "eagle"])
def test_sequential_drafting_hash_ignores_speculative_depth(method: str):
    """Sequential drafting depth is a loop count, not a static query shape."""
    assert (
        _config(method, 3, parallel_drafting=False).compute_hash()
        == _config(method, 9, parallel_drafting=False).compute_hash()
    )
