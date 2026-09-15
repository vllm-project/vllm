# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the production UltraQuant D=256 FlyDSL decode."""

import pytest
import torch

from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_flydsl_hd256_available,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import (
    hadamard_matrix,
    reference_ultraquant_attention,
)
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

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


@pytest.mark.parametrize(
    ("batch_size", "query_group_size", "block_size", "contiguous_query"),
    [
        (1, 8, 32, True),
        (8, 8, 64, True),
        (9, 8, 256, True),
        (1, 16, 32, True),
        (1, 6, 16, True),
        (1, 6, 32, True),
        (2, 6, 64, True),
        (1, 6, 128, True),
        (1, 6, 256, True),
        (1, 6, 32, False),
    ],
)
def test_flydsl_decode_matches_reference(
    batch_size,
    query_group_size,
    block_size,
    contiguous_query,
):
    if not is_flydsl_hd256_available(query_group_size):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    torch.manual_seed(0x57A1D + batch_size + block_size + query_group_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_kv_heads = 1
    head_size = 256
    seq_len = 2048
    scale = head_size**-0.5

    key = torch.randn(
        batch_size,
        seq_len,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    if contiguous_query:
        qkv = torch.randn(
            batch_size,
            query_group_size + 2 * num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
        )
        query = qkv[:, :query_group_size]
    else:
        query_storage = torch.randn(
            batch_size,
            query_group_size,
            head_size * 2,
            dtype=dtype,
            device=device,
        )
        query = query_storage[..., ::2]
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    hadamard = hadamard_matrix(head_size, device)

    actual = flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        PiT=hadamard,
        max_seq_len=seq_len,
    )
    expected = reference_ultraquant_attention(
        query=query,
        key=key,
        value=value,
        scale=scale,
    )

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    error = (actual_f32 - expected_f32).abs()
    assert cosine > 0.995
    assert error.mean() < 0.05
    assert error.max() < 0.8
