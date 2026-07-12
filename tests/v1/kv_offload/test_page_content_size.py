# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""content_size_bytes is the offload transfer width; it must include the
inline per-token-head scales."""

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


def _spec(**kwargs) -> FullAttentionSpec:
    kwargs.setdefault("dtype", torch.bfloat16)
    return FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, **kwargs)


def test_content_size_without_quant_matches_real_page():
    spec = _spec()
    assert spec.content_size_bytes == spec.real_page_size_bytes
    assert spec.page_size_bytes == spec.content_size_bytes


def test_content_size_includes_per_token_head_scales():
    spec = _spec(dtype=torch.uint8, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    scales = 2 * spec.block_size * spec.num_kv_heads * 4
    assert spec.content_size_bytes == spec.real_page_size_bytes + scales
    assert spec.page_size_bytes == spec.content_size_bytes


def test_page_size_padded_wins():
    spec = _spec(
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        page_size_padded=65536,
    )
    assert spec.page_size_bytes == 65536
