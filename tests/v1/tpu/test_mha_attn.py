# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test:

* Tests for MMEncoderAttention layer
"""

import pytest
import torch
import torch_xla
import torch_xla.core
import torch_xla.core.xla_model

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.attention.selector import _cached_get_attn_backend
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


def ref_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Native implementation of scaled dot product attention without mask:
    - query, key, value: [batch_size, seq_len, num_heads, head_size]
    - attn_mask: [batch_size, seq_len, seq_len]
    """
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    attn_weights = scale * torch.matmul(query, key.transpose(2, 3))
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.matmul(attn_weights, value).transpose(1, 2)
    return out


BATCH_SIZES = [1, 16]
SEQ_LENS = [1]
NUM_HEADS = [1, 16]
NUM_KV_HEADS = [1]
HEAD_SIZES = [64, 80]


@pytest.mark.skipif(not current_platform.is_tpu(), reason="This test needs a TPU")
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("device", [torch_xla.core.xla_model.xla_device()])
def test_mha_attn_forward(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    device: str,
):
    set_random_seed(0)
    # These are expected to be f32
    q = torch.randn(batch_size, seq_len, num_heads * head_size, device=device)
    k = torch.randn(batch_size, seq_len, num_kv_heads * head_size, device=device)
    v = torch.randn(batch_size, seq_len, num_kv_heads * head_size, device=device)
    scale = 1.0 / head_size**0.5
    attn = MMEncoderAttention(
        num_heads, head_size, scale=scale, num_kv_heads=num_kv_heads
    )
    output = attn(q, k, v)

    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads

    q = q.reshape(batch_size, seq_len, num_heads, head_size)
    k = k.reshape(batch_size, seq_len, num_kv_heads, head_size)
    v = v.reshape(batch_size, seq_len, num_kv_heads, head_size)
    if num_queries_per_kv > 1:
        k = torch.repeat_interleave(k, num_queries_per_kv, dim=2)
        v = torch.repeat_interleave(v, num_queries_per_kv, dim=2)

    ref_output = ref_attention(
        q,
        k,
        v,
        scale=scale,
    ).reshape(batch_size, seq_len, num_heads * head_size)
    # torch_xla flash_attn kernel is less accurate but much faster
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
