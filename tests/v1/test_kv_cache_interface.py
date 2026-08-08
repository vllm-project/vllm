# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("cache_dtype_str", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_plain_fp8_mla_real_page_size_uses_fp8_storage(cache_dtype_str: str):
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=768,
        dtype=torch.float32,
        cache_dtype_str=cache_dtype_str,
    )

    assert spec.real_page_size_bytes == 16 * 768


@pytest.mark.parametrize("cache_dtype_str", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_plain_fp8_sliding_window_mla_real_page_size_uses_fp8_storage(
    cache_dtype_str: str,
):
    spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=768,
        dtype=torch.float32,
        sliding_window=128,
        cache_dtype_str=cache_dtype_str,
    )

    assert spec.real_page_size_bytes == 16 * 768
