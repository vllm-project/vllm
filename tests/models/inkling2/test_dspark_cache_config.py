# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import CacheConfig
from vllm.models.inkling2.nvidia.dspark import _draft_cache_config


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "float16"])
def test_draft_cache_stays_bf16_without_fp8(cache_dtype):
    assert _draft_cache_config(CacheConfig(cache_dtype=cache_dtype)) is None


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_draft_cache_takes_fp8_and_drops_the_model_window(cache_dtype):
    cache_config = CacheConfig(cache_dtype=cache_dtype)
    cache_config.sliding_window = 1024

    draft = _draft_cache_config(cache_config)

    assert draft is not None and draft is not cache_config
    assert draft.cache_dtype == cache_dtype
    assert draft.sliding_window is None
    assert cache_config.sliding_window == 1024
