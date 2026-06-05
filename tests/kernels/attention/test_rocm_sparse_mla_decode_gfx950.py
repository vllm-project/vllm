# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the HIP MFMA sparse-MLA decode kernels (gfx950).

All tests use the DeepSeek-V4-Pro production attention dims (see HF config
https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) and
are parametrized over the runtime axes the kernel must support:
  * batch (``num_queries``): single decode and small MTP-style batches.
  * sequence length per query: the SWA window (``sliding_window=128``) and
    the topk budget (``index_topk=1024``).
  * split-K: both the single-WG path and the split-reduce path are covered
    by forcing the internal split-K override.
"""

import contextlib

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

# Per-token KV head dims for DeepSeek-V4-Pro MLA. The HIP kernel is
# hard-coded for these values (576-byte token payload + 8-byte scales):
#   nope_head_dim = head_dim(512) - qk_rope_head_dim(64) = 448
#   rope_head_dim = qk_rope_head_dim                     = 64
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM

# DeepSeek-V4-Pro production attention params (HF config.json):
#   * num_attention_heads = 128  ->  16 heads per rank at TP=8.
#   * ROCM_AITER_MLA_SPARSE backend supports kernel_block_size in {1, 64};
#     paged deployments use 64.
#   * sliding_window = 128       ->  max main/SWA tokens per query.
#   * index_topk    = 1024       ->  max extra/topk tokens per query.
DSV4_NUM_HEADS = 16
DSV4_BLOCK_SIZE = 64
DSV4_SWA_WINDOW = 128
DSV4_TOPK = 1024


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


@contextlib.contextmanager
def _force_split_k(value: int | None):
    """Temporarily override the internal split-K picker.

    The kernel module reads ``SPARSE_MLA_HIP_SPLIT_K`` once at import time
    into ``mod._SPLIT_K_OVERRIDE``; tests mutate that module attribute
    directly so they can exercise both the single-WG (split_k=1) and the
    split-reduce paths regardless of the auto-tuned heuristic.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    orig = mod._SPLIT_K_OVERRIDE
    if value is not None:
        mod._SPLIT_K_OVERRIDE = str(int(value))
    try:
        yield
    finally:
        mod._SPLIT_K_OVERRIDE = orig


def _make_q(num_queries: int, num_heads: int, device: torch.device) -> torch.Tensor:
    return (
        torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.125
    )


def _build_contiguous_ragged(
    num_queries: int,
    tokens_per_query: int,
    device: torch.device,
    start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Build (indices, indptr, rows) where each query owns
    ``tokens_per_query`` contiguous slots starting at ``start``."""
    rows = [
        list(range(start + qi * tokens_per_query, start + (qi + 1) * tokens_per_query))
        for qi in range(num_queries)
    ]
    indices = torch.tensor(
        [s for r in rows for s in r], dtype=torch.int32, device=device
    )
    indptr = torch.tensor(
        [qi * tokens_per_query for qi in range(num_queries + 1)],
        dtype=torch.int32,
        device=device,
    )
    return indices, indptr, rows


# Parametrize against the realistic DSv4 deployment axes:
#   * ``num_queries`` covers single-token decode and small MTP batches.
#   * ``split_k`` covers both the single-WG path and the split-reduce path.
#   * ``with_sink`` toggles the attention-sink branch.
# All tests pin ``num_heads = DSV4_NUM_HEADS`` (=16, DSv4 at TP=8) and
# ``block_size = DSV4_BLOCK_SIZE`` (=64, the paged-MLA cache block size).


@torch.inference_mode()
@pytest.mark.parametrize("split_k", [1, 4])
@pytest.mark.parametrize("num_queries", [1, 4])
@pytest.mark.parametrize("with_sink", [False, True])
def test_hip_decode_main_only(num_queries, with_sink, split_k) -> None:
    """SWA-only decode (main cache) at DSv4 dims, with/without sink."""
    device = torch.device("cuda")
    torch.manual_seed(42 + num_queries * 10 + int(with_sink) * 3 + split_k)
    tokens_per_query = DSV4_SWA_WINDOW

    q = _make_q(num_queries, DSV4_NUM_HEADS, device)
    main_kv = (
        torch.randn(
            num_queries * tokens_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, DSV4_BLOCK_SIZE)
    main_indices, main_indptr, main_rows = _build_contiguous_ragged(
        num_queries, tokens_per_query, device
    )
    attn_sink = (
        torch.randn(DSV4_NUM_HEADS, dtype=torch.float32, device=device) * 0.1
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    with _force_split_k(split_k):
        actual = _call_hip_decode(
            q,
            main_cache,
            main_indices,
            main_indptr,
            scale,
            attn_sink=attn_sink,
            max_main_len=tokens_per_query,
        )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        main_rows,
        scale,
        attn_sink=attn_sink,
        block_size=DSV4_BLOCK_SIZE,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("split_k", [1, 4])
@pytest.mark.parametrize("num_queries", [1, 4])
@pytest.mark.parametrize("with_sink", [False, True])
def test_hip_decode_main_extra(num_queries, with_sink, split_k) -> None:
    """SWA + topk decode (main + extra caches) at DSv4 dims."""
    device = torch.device("cuda")
    torch.manual_seed(7 + num_queries * 11 + int(with_sink) * 5 + split_k)
    # ``main`` carries SWA tokens; ``extra`` carries topk tokens. Use a
    # modest extra length (rather than the full DSV4_TOPK=1024) to keep
    # the Python reference fast while still exercising both code paths.
    main_per_query = DSV4_SWA_WINDOW
    extra_per_query = DSV4_BLOCK_SIZE * 2  # 128 topk tokens

    q = _make_q(num_queries, DSV4_NUM_HEADS, device)
    main_kv = (
        torch.randn(
            num_queries * main_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    extra_kv = (
        torch.randn(
            num_queries * extra_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, DSV4_BLOCK_SIZE)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, DSV4_BLOCK_SIZE)
    main_indices, main_indptr, main_rows = _build_contiguous_ragged(
        num_queries, main_per_query, device
    )
    extra_indices, extra_indptr, extra_rows = _build_contiguous_ragged(
        num_queries, extra_per_query, device
    )
    attn_sink = (
        torch.randn(DSV4_NUM_HEADS, dtype=torch.float32, device=device) * 0.1
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    with _force_split_k(split_k):
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
            max_main_len=main_per_query,
            max_extra_len=extra_per_query,
        )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        main_rows,
        scale,
        attn_sink=attn_sink,
        block_size=DSV4_BLOCK_SIZE,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("split_k", [2, 4, 8])
@pytest.mark.parametrize("num_queries", [1, 4])
def test_hip_decode_split_k(num_queries, split_k) -> None:
    """Force the split-K reduce path with the DSv4 topk-sized seqlen."""
    device = torch.device("cuda")
    torch.manual_seed(2026 + num_queries * 13 + split_k)
    tokens_per_query = DSV4_TOPK  # full DSv4 topk budget

    q = _make_q(num_queries, DSV4_NUM_HEADS, device)
    main_kv = (
        torch.randn(
            num_queries * tokens_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, DSV4_BLOCK_SIZE)
    main_indices, main_indptr, main_rows = _build_contiguous_ragged(
        num_queries, tokens_per_query, device
    )
    attn_sink = torch.randn(DSV4_NUM_HEADS, dtype=torch.float32, device=device) * 0.1
    scale = HEAD_DIM**-0.5

    with _force_split_k(split_k):
        actual = _call_hip_decode(
            q,
            main_cache,
            main_indices,
            main_indptr,
            scale,
            attn_sink=attn_sink,
            max_main_len=tokens_per_query,
        )
    expected = _ref_sparse_decode_ragged(
        q,
        main_cache,
        main_rows,
        scale,
        attn_sink=attn_sink,
        block_size=DSV4_BLOCK_SIZE,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("split_k", [1, 4])
@pytest.mark.parametrize("num_queries", [1, 4])
@pytest.mark.parametrize(
    "tokens_per_query",
    [DSV4_BLOCK_SIZE, DSV4_SWA_WINDOW, DSV4_TOPK],
    ids=["one_block", "swa_window", "topk"],
)
def test_hip_decode_seqlen(num_queries, tokens_per_query, split_k) -> None:
    """Sweep DSv4 sequence lengths: one block, SWA window, full topk."""
    device = torch.device("cuda")
    torch.manual_seed(num_queries * 1009 + tokens_per_query + split_k)

    q = _make_q(num_queries, DSV4_NUM_HEADS, device)
    main_kv = (
        torch.randn(
            num_queries * tokens_per_query,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = _pack_fp8_ds_mla_cache(main_kv, DSV4_BLOCK_SIZE)
    main_indices, main_indptr, main_rows = _build_contiguous_ragged(
        num_queries, tokens_per_query, device
    )
    scale = HEAD_DIM**-0.5

    with _force_split_k(split_k):
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
        block_size=DSV4_BLOCK_SIZE,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
