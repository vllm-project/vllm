# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the UltraQuant Triton unified-attention fallback."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import reference_ultraquant_attention
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import (
    ultraquant_unified_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _build_paged_cache(key, value, block_size):
    batch_size, seq_len, num_kv_heads, head_size = key.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    kv_cache = torch.zeros(
        batch_size * blocks_per_seq,
        block_size,
        num_kv_heads,
        slot_size(head_size),
        dtype=torch.uint8,
        device=key.device,
    )
    block_table = torch.arange(
        batch_size * blocks_per_seq,
        dtype=torch.int32,
        device=key.device,
    ).view(batch_size, blocks_per_seq)
    slot_mapping = torch.arange(
        batch_size * blocks_per_seq * block_size,
        dtype=torch.int64,
        device=key.device,
    ).view(batch_size, -1)[:, :seq_len]
    ultraquant_store(
        key.reshape(-1, num_kv_heads, head_size),
        value.reshape(-1, num_kv_heads, head_size),
        kv_cache,
        slot_mapping.reshape(-1),
    )
    return kv_cache, block_table


def _attention_quality(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    diff = (a - b).abs()
    return {
        "cos_sim": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "norm_ratio": (a.norm() / b.norm()).item(),
    }


def _unified_decode(query, kv_cache, block_table, seq_lens, scale, sinks=None):
    batch_size = query.shape[0]
    return ultraquant_unified_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=torch.arange(
            batch_size + 1, dtype=torch.int32, device=query.device
        ),
        scale=scale,
        max_query_len=1,
        max_seq_len=int(seq_lens.max().item()),
        sinks=sinks,
    )


@pytest.mark.parametrize("query_group_size", [6, 8])
def test_unified_decode_d256_matches_reference(query_group_size):
    """The D=256 fallback covers both target Qwen GQA layouts."""
    device = torch.device("cuda")
    torch.manual_seed(0xD256 + query_group_size)

    B, Hk, N, D = 2, 1, 256, 256
    Hq = query_group_size * Hk
    block_size = 32
    dtype = torch.bfloat16
    scale = D**-0.5

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    actual = _unified_decode(query, kv_cache, block_table, seq_lens, scale)
    expected = reference_ultraquant_attention(
        query=query,
        key=key,
        value=value,
        scale=scale,
    )

    quality = _attention_quality(actual, expected)
    assert quality["cos_sim"] > 0.98, quality
    assert quality["mean_abs"] < 0.08, quality
    assert quality["max_abs"] < 0.8, quality


def test_unified_decode_with_sinks():
    """Sinks are served by the Triton fallback, not the FlyDSL kernel."""
    device = torch.device("cuda")
    torch.manual_seed(0x8517)

    B, Hq, Hk, N, D = 1, 8, 1, 256, 256
    block_size = 32
    dtype = torch.bfloat16
    scale = D**-0.5

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    sinks = torch.randn(Hq, dtype=torch.float32, device=device) * 2.0

    actual = _unified_decode(query, kv_cache, block_table, seq_lens, scale, sinks)
    expected = reference_ultraquant_attention(
        query=query, key=key, value=value, scale=scale, sinks=sinks
    )
    quality = _attention_quality(actual, expected)
    assert quality["cos_sim"] > 0.98, quality
    assert quality["mean_abs"] < 0.08, quality
    assert quality["max_abs"] < 0.8, quality

    sinks_low = torch.full((Hq,), -50.0, dtype=torch.float32, device=device)
    out_low = _unified_decode(query, kv_cache, block_table, seq_lens, scale, sinks_low)
    out_no_sink = _unified_decode(query, kv_cache, block_table, seq_lens, scale)
    quality_low = _attention_quality(out_low, out_no_sink)
    assert quality_low["cos_sim"] > 0.999, quality_low
