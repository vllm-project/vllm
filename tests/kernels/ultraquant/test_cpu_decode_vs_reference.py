# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AVX-512 UltraQuant decode kernel on CPU.

The kernel is checked against ``reference_ultraquant_attention`` over a cache
the CPU store wrote, so store and decode are held to one shared definition of
the format. K/V are fp32 here, which is the store's bit-exact path, and Q is
rotated in fp32, so what is left is the vectorised ``fast_exp`` the online
softmax shares with ``cpu_attn.cpp``. A separate case covers the bf16
rotations the backend actually runs.
"""

from __future__ import annotations

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.ultraquant.format import DEFAULT_CONSTANT_C, slot_size
from vllm.v1.attention.ops.ultraquant.reference import (
    hadamard_matrix,
    reference_ultraquant_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cpu._is_avx512_supported(), reason="requires AVX-512"
)

# The softmax's vectorised exp is an approximation; this is its whole budget.
_FAST_EXP_TOL = 5e-3


def _build_paged_cache(key, value, block_size):
    """Store [B, N, Hk, D] K/V into a fresh CPU paged cache."""
    batch, seq_len, num_kv_heads, head_dim = key.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    kv_cache = torch.zeros(
        batch * blocks_per_seq,
        num_kv_heads,
        block_size,
        slot_size(head_dim),
        dtype=torch.uint8,
    )
    block_table = torch.arange(batch * blocks_per_seq, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    slot_mapping = torch.arange(
        batch * blocks_per_seq * block_size, dtype=torch.int64
    ).view(batch, -1)[:, :seq_len]
    ops.ultraquant_store(
        key.reshape(-1, num_kv_heads, head_dim),
        value.reshape(-1, num_kv_heads, head_dim),
        kv_cache,
        slot_mapping.reshape(-1),
        hadamard_matrix(head_dim, key.device, torch.float32),
        DEFAULT_CONSTANT_C,
    )
    return kv_cache, block_table


def _decode(query, kv_cache, block_table, seq_lens, scale, sinks=None, rot_dtype=None):
    """Rotate and FP8-round Q as the backend does, then run the op.

    ``rot_dtype`` picks the rotation's precision: bf16 is what the backend
    uses, fp32 isolates the kernel from that choice.
    """
    head_dim = query.shape[-1]
    rot_dtype = rot_dtype or torch.float32
    rot_t = hadamard_matrix(head_dim, query.device, rot_dtype).T.contiguous()
    q = (query.to(rot_dtype) @ rot_t).to(torch.float8_e4m3fn).to(torch.bfloat16)
    out = torch.empty(query.shape, dtype=torch.float32)
    ops.ultraquant_decode_attention(
        q, kv_cache, block_table, seq_lens, out, scale, sinks
    )
    return out


def _rel_err(actual, expected):
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("num_kv_heads,query_group_size", [(1, 1), (8, 3), (2, 8)])
@pytest.mark.parametrize("seq_len", [1, 31, 128, 999])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_decode_matches_reference(num_kv_heads, query_group_size, seq_len, head_dim):
    set_random_seed(0xDEC0DE)
    batch, block_size = 2, 32
    num_q_heads = num_kv_heads * query_group_size
    scale = head_dim**-0.5

    query = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)

    actual = _decode(query, kv_cache, block_table, seq_lens, scale)
    expected = reference_ultraquant_attention(
        query=query, key=key, value=value, scale=scale
    )
    assert _rel_err(actual, expected) < _FAST_EXP_TOL


def test_decode_handles_ragged_seq_lens():
    """Sequences that stop mid-block must not read the block's unwritten tail."""
    set_random_seed(0xAA)
    batch, num_kv_heads, num_q_heads, head_dim, block_size = 3, 2, 6, 128, 32
    lens = [1, 33, 100]
    scale = head_dim**-0.5

    key = torch.randn(batch, max(lens), num_kv_heads, head_dim)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    query = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16)

    actual = _decode(
        query, kv_cache, block_table, torch.tensor(lens, dtype=torch.int32), scale
    )
    for b, n in enumerate(lens):
        expected = reference_ultraquant_attention(
            query=query[b : b + 1],
            key=key[b : b + 1, :n],
            value=value[b : b + 1, :n],
            scale=scale,
        )
        assert _rel_err(actual[b : b + 1], expected) < _FAST_EXP_TOL, (
            f"seq {b} (len {n})"
        )


def test_decode_applies_sinks():
    set_random_seed(0x5151)
    batch, num_kv_heads, num_q_heads, head_dim = 2, 2, 4, 128
    seq_len, block_size = 64, 32
    scale = head_dim**-0.5

    query = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    sinks = torch.randn(num_q_heads, dtype=torch.float32)

    actual = _decode(query, kv_cache, block_table, seq_lens, scale, sinks)
    expected = reference_ultraquant_attention(
        query=query, key=key, value=value, scale=scale, sinks=sinks
    )
    assert _rel_err(actual, expected) < _FAST_EXP_TOL


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_decode_in_backend_precision(head_dim):
    """End to end on the precision the backend ships: both the store's K
    rotation and the query's run in bf16, on top of the FP4 cache itself."""
    set_random_seed(0xB16)
    batch, num_kv_heads, num_q_heads, seq_len, block_size = 2, 2, 6, 512, 32
    scale = head_dim**-0.5

    query = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)

    actual = _decode(
        query, kv_cache, block_table, seq_lens, scale, rot_dtype=torch.bfloat16
    )
    expected = reference_ultraquant_attention(
        query=query, key=key, value=value, scale=scale
    )
    assert _rel_err(actual, expected) < 5e-2
