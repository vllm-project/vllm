# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the HIP MFMA sparse-MLA decode kernels (gfx950).

Tests cover:
  - single-WG decode (split_k == 1): main-only, main+extra, with/without attn_sink
  - split-K decode: forced via SPARSE_MLA_HIP_SPLIT_K env override
  - various batch / head / sequence-length combinations
"""

import os

import pytest
import torch

from vllm.platforms import current_platform


def _is_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX950

        return _ON_GFX950
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="Requires ROCm gfx950 hardware"
)

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM


# ---------------------------------------------------------------------------
# Helpers (shared with test_rocm_triton_attn_dsv4.py)
# ---------------------------------------------------------------------------


def _pack_fp8_ds_mla_cache(kv: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pack bf16 KV rows into the fp8_ds_mla uint8 cache layout."""
    assert kv.shape[-1] == HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584),
        dtype=torch.uint8,
        device=kv.device,
    )
    cache_flat = cache.view(torch.uint8).flatten()
    kv_nope_fp8 = (
        kv[:, :NOPE_HEAD_DIM].to(current_platform.fp8_dtype()).view(torch.uint8)
    )
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)

    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[
            token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
        ].copy_(kv_rope_u8[slot])
        cache_flat[scale_base : scale_base + 7].fill_(127)
    return cache


def _read_fp8_ds_mla_cache(
    cache: torch.Tensor, slot: int, block_size: int
) -> torch.Tensor:
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576

    nope_u8 = cache_flat[token_base : token_base + NOPE_HEAD_DIM]
    nope = nope_u8.view(current_platform.fp8_dtype()).to(torch.float32)
    rope_u8 = cache_flat[
        token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
    ]
    rope = rope_u8.view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope])


def _ref_sparse_decode_ragged(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_rows: list[list[int]],
    scale: float,
    attn_sink: torch.Tensor | None,
    block_size: int,
    extra_cache: torch.Tensor | None = None,
    extra_rows: list[list[int]] | None = None,
) -> torch.Tensor:
    """Pure-Python reference for ragged sparse decode attention."""
    q_f32 = q.float()
    out = torch.empty_like(q_f32)

    for query_idx in range(q.shape[0]):
        row_kv = [
            _read_fp8_ds_mla_cache(main_cache, int(slot), block_size)
            for slot in main_rows[query_idx]
        ]
        if extra_cache is not None and extra_rows is not None:
            row_kv.extend(
                _read_fp8_ds_mla_cache(extra_cache, int(slot), block_size)
                for slot in extra_rows[query_idx]
            )

        kv = torch.stack(row_kv).to(q.device)
        for head_idx in range(q.shape[1]):
            scores = torch.mv(kv, q_f32[query_idx, head_idx]) * scale
            if attn_sink is not None:
                scores_with_sink = torch.cat(
                    [scores, attn_sink[head_idx].float().reshape(1)]
                )
                probs = torch.softmax(scores_with_sink, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[query_idx, head_idx] = torch.sum(probs[:, None] * kv, dim=0)
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _call_hip_decode(
    q,
    main_cache,
    main_indices,
    main_indptr,
    scale,
    attn_sink,
    extra_cache=None,
    extra_indices=None,
    extra_indptr=None,
    max_main_len=None,
    max_extra_len=None,
):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _decode_sparse_mla_hip,
    )

    if max_main_len is None:
        max_main_len = int(main_indices.numel())
    if max_extra_len is None:
        max_extra_len = int(extra_indices.numel()) if extra_indices is not None else 0

    return _decode_sparse_mla_hip(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        max_main_len=max_main_len,
        max_extra_len=max_extra_len,
    )


@torch.inference_mode()
def test_hip_decode_main_only_no_sink() -> None:
    """Single-WG decode with main cache only, no attn_sink."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    block_size = 4
    num_queries, num_heads = 2, 3
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _call_hip_decode(
        q,
        main_cache,
        main_indices,
        main_indptr,
        scale,
        attn_sink=None,
    )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        [[0, 2], [4, 1]],
        scale,
        attn_sink=None,
        block_size=block_size,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_hip_decode_main_only_with_sink() -> None:
    """Single-WG decode with main cache only, with attn_sink."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    block_size = 4
    num_queries, num_heads = 2, 3
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _call_hip_decode(
        q,
        main_cache,
        main_indices,
        main_indptr,
        scale,
        attn_sink=attn_sink,
    )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        [[0, 2], [4, 1]],
        scale,
        attn_sink=attn_sink,
        block_size=block_size,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_hip_decode_main_extra_with_sink() -> None:
    """Single-WG decode with main + extra cache and attn_sink."""
    device = torch.device("cuda")
    torch.manual_seed(1)
    block_size = 4
    num_queries, num_heads = 2, 3
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    extra_kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size)
    main_indices = torch.tensor([0, 2, 4, 1], dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    extra_indices = torch.tensor([1, 3, 0], dtype=torch.int32, device=device)
    extra_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    attn_sink = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32, device=device)
    scale = HEAD_DIM**-0.5

    actual = _call_hip_decode(
        q,
        main_cache,
        main_indices,
        main_indptr,
        scale,
        attn_sink=attn_sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        [[0, 2], [4, 1]],
        scale,
        attn_sink=attn_sink,
        block_size=block_size,
        extra_cache=extra_cache,
        extra_rows=[[1], [3, 0]],
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_hip_decode_split_k() -> None:
    """Force split-K path (split_k=2) and verify correctness."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    block_size = 4
    num_queries, num_heads = 1, 16
    num_tokens = 128
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = (
        torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    main_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    main_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    scale = HEAD_DIM**-0.5

    old_val = os.environ.get("SPARSE_MLA_HIP_SPLIT_K")
    try:
        os.environ["SPARSE_MLA_HIP_SPLIT_K"] = "2"
        from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

        orig = mod._SPLIT_K_OVERRIDE
        mod._SPLIT_K_OVERRIDE = "2"

        actual = _call_hip_decode(
            q,
            main_cache,
            main_indices,
            main_indptr,
            scale,
            attn_sink=attn_sink,
            max_main_len=num_tokens,
        )
    finally:
        mod._SPLIT_K_OVERRIDE = orig
        if old_val is None:
            os.environ.pop("SPARSE_MLA_HIP_SPLIT_K", None)
        else:
            os.environ["SPARSE_MLA_HIP_SPLIT_K"] = old_val

    main_rows = [list(range(num_tokens))]
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        main_rows,
        scale,
        attn_sink=attn_sink,
        block_size=block_size,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("num_queries", [1, 4])
@pytest.mark.parametrize("num_heads", [3, 16, 32])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_hip_decode_shapes(num_queries, num_heads, seq_len) -> None:
    """Parametrized test over different batch/head/seqlen combos."""
    device = torch.device("cuda")
    torch.manual_seed(num_queries * 1000 + num_heads * 10 + seq_len)
    block_size = 4
    q = (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    main_kv = (
        torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size)
    scale = HEAD_DIM**-0.5

    tokens_per_query = seq_len // num_queries
    main_rows = []
    indices_list = []
    indptr = [0]
    for qi in range(num_queries):
        start = qi * tokens_per_query
        end = start + tokens_per_query
        row_slots = list(range(start, end))
        main_rows.append(row_slots)
        indices_list.extend(row_slots)
        indptr.append(indptr[-1] + len(row_slots))

    main_indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
    main_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)

    actual = _call_hip_decode(
        q,
        main_cache,
        main_indices,
        main_indptr,
        scale,
        attn_sink=None,
        max_main_len=tokens_per_query,
    )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        main_rows,
        scale,
        attn_sink=None,
        block_size=block_size,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
