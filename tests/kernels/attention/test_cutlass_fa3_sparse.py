# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vendored CUTLASS FA3 MLA sparse attention kernel.

These tests verify the kernel wrapper works correctly in isolation,
following the pattern of test_flashmla_sparse.py smoke tests.

Tests cover:
  1. Extension availability
  2. Basic MLA decode smoke test
  3. Batched decode
  4. Correctness vs SDPA reference
  5. Invalid indices handling
  6. Variable sequence lengths
  7. num_splits variants
  8. softcap parameter
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.cutlass_fa3 import (
    flash_attn_with_kvcache,
    is_cutlass_fa3_available,
)

# Skip all tests if FA3 is not available (non-SM90 or CUDA < 12.4)
pytestmark = pytest.mark.skipif(
    not is_cutlass_fa3_available(),
    reason="CUTLASS FA3 not available (requires CUDA >= 12.4, SM90)",
)

# Common MLA dimensions for DeepSeek-V3.2
NUM_HEADS = 16
HEADDIM_QK = 64  # RoPE component
HEADDIM_V = 512  # kv_lora_rank (NoPE component)
HEAD_SIZE = HEADDIM_V + HEADDIM_QK  # 576
SOFTMAX_SCALE = 192 ** (-0.5)  # qk_head_dim = 192


def _make_kv_cache(kv_pool_size: int, dtype=torch.bfloat16, device="cuda"):
    """Create mock KV cache with page_size=1 format.

    FA3 expects paged KV as 4D: [num_pages, page_size=1, num_kv_heads=1, dim]
    """
    k_rope_cache = torch.randn(
        kv_pool_size, 1, 1, HEADDIM_QK, dtype=dtype, device=device
    )
    v_cache = torch.randn(kv_pool_size, 1, 1, HEADDIM_V, dtype=dtype, device=device)
    return k_rope_cache, v_cache


def _make_page_table(T, topk, kv_pool_size, device="cuda"):
    """Create page table with unique random indices per token."""
    page_table = torch.stack(
        [torch.randperm(kv_pool_size, device=device)[:topk] for _ in range(T)]
    ).to(torch.int32)
    return page_table


def _make_cu_seqlens(cache_seqlens, device="cuda"):
    """Create cumulative sequence length tensors for FA3."""
    T = cache_seqlens.shape[0]
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(cache_seqlens, dim=0),
        ]
    )
    return cu_seqlens_q, cu_seqlens_k


# ─── TEST 1.1: Availability ──────────────────────────────────────────


def test_cutlass_fa3_availability():
    """Verify the _cutlass_fa3_C extension loads successfully."""
    assert is_cutlass_fa3_available()
    assert hasattr(torch.ops, "_cutlass_fa3_C")
    assert hasattr(torch.ops._cutlass_fa3_C, "fwd")


# ─── TEST 1.2: Basic Smoke Test ──────────────────────────────────────


def test_fa3_mla_decode_smoke():
    """Basic smoke test: 1 token, 1 request, small topk."""
    device = "cuda"
    T = 1
    topk = 128
    kv_pool_size = 256

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.tensor([topk], dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert out.dtype == torch.bfloat16
    assert not out.isnan().any(), "Output contains NaN values"
    assert not out.isinf().any(), "Output contains Inf values"


# ─── TEST 1.3: Batched Decode ────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_fa3_mla_decode_batched(batch_size):
    """Verify batched decode with multiple tokens."""
    device = "cuda"
    T = batch_size
    topk = 256
    kv_pool_size = 4096

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.4: Correctness vs SDPA Reference ─────────────────────────


def test_fa3_mla_correctness_vs_reference():
    """Verify numerical correctness against PyTorch SDPA reference.

    For each batch element, we manually compute attention using the same
    Q, K, V data and compare against FA3 output.
    """
    device = "cuda"
    T = 4
    topk = 64  # small for reference tractability
    kv_pool_size = 256

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    # FA3 output
    fa3_out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=1,  # deterministic
    )
    # fa3_out is already [T, N, 512] in varlen mode

    # Reference: per-token SDPA
    for b in range(T):
        indices = page_table[b]  # [topk]
        # Gather KV from cache (4D: [topk,1,1,dim] -> squeeze to [topk,dim])
        k_rope_gathered = k_rope_cache[indices].squeeze(1).squeeze(1)  # [topk, 64]
        v_gathered = v_cache[indices].squeeze(1).squeeze(1)  # [topk, 512]

        # Full Q: [q_nope | q_rope] concatenated
        q_r = q_rope[b]  # [N, 64]
        q_n = q_nope[b]  # [N, 512]

        # Full K: [k_nope_from_v | k_rope] — for MLA, K_nope = V (latent)
        k_full = torch.cat([v_gathered, k_rope_gathered], dim=-1)  # [topk, 576]

        # Expand for MQA: [topk, 1, dim] -> [topk, N, dim]
        k_expanded = k_full.unsqueeze(1).expand(-1, NUM_HEADS, -1)  # [topk, N, 576]
        v_expanded = v_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)  # [topk, N, 512]

        # Q full: [q_nope | q_rope]
        q_full = torch.cat([q_n, q_r], dim=-1)  # [N, 576]

        # SDPA expects: Q[batch, heads, seq_q, dim], K[batch, heads, seq_k, dim]
        # Q: [1, N, 1, 576], K: [1, N, topk, 576], V: [1, N, topk, 512]
        ref_out = F.scaled_dot_product_attention(
            q_full.unsqueeze(0).unsqueeze(2),  # [1, N, 1, 576]
            k_expanded.transpose(0, 1).unsqueeze(0),  # [1, N, topk, 576]
            v_expanded.transpose(0, 1).unsqueeze(0),  # [1, N, topk, 512]
            scale=SOFTMAX_SCALE,
        )  # -> [1, N, 1, 512]
        ref_out = ref_out.squeeze(0).squeeze(1)  # [N, 512]

        # Compare (BF16 tolerances)
        torch.testing.assert_close(
            fa3_out[b].float(), ref_out.float(), rtol=0.02, atol=0.02
        )


# ─── TEST 1.5: Invalid Indices ───────────────────────────────────────


def test_fa3_mla_with_invalid_indices():
    """Verify handling of -1 (padding) entries in page_table."""
    device = "cuda"
    T = 1
    topk_valid = 64
    topk_total = 256
    kv_pool_size = 512

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    # Mix of valid and -1 entries
    page_table = torch.full((T, topk_total), -1, dtype=torch.int32, device=device)
    page_table[0, :topk_valid] = torch.randperm(kv_pool_size, device=device)[
        :topk_valid
    ].to(torch.int32)

    cache_seqlens = torch.tensor([topk_valid], dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()
    assert not out.isinf().any()


# ─── TEST 1.6: Variable Sequence Lengths ─────────────────────────────


@pytest.mark.parametrize(
    "seq_lens",
    [
        [100, 200, 500, 1000],
        [1, 1, 1, 1],
        [2048, 2048, 2048, 2048],
        [10, 2048, 50, 1500],
    ],
)
def test_fa3_mla_variable_seqlens(seq_lens):
    """Verify with variable cache_seqlens per batch element."""
    device = "cuda"
    T = len(seq_lens)
    topk = 2048
    kv_pool_size = 8192

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    page_table = torch.full((T, topk), 0, dtype=torch.int32, device=device)
    actual_seqlens = []
    for i, sl in enumerate(seq_lens):
        actual_topk = min(sl, topk)
        actual_seqlens.append(actual_topk)
        page_table[i, :actual_topk] = torch.randperm(kv_pool_size, device=device)[
            :actual_topk
        ].to(torch.int32)

    cache_seqlens = torch.tensor(actual_seqlens, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.7: num_splits Variants ───────────────────────────────────


@pytest.mark.parametrize("num_splits", [0, 1, 2, 4])
def test_fa3_mla_num_splits(num_splits):
    """Verify different num_splits values produce valid results."""
    device = "cuda"
    T = 4
    topk = 256
    kv_pool_size = 4096

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=num_splits,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.8: Softcap Parameter ─────────────────────────────────────


@pytest.mark.parametrize("softcap", [0.0, 30.0, 50.0])
def test_fa3_mla_softcap(softcap):
    """Verify logits_soft_cap parameter works."""
    device = "cuda"
    T = 2
    topk = 128
    kv_pool_size = 512

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        softcap=softcap,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any(), "Softcap test output contains NaN"


# ─── TEST 1.9: Prefill Short Sequence (REGRESSION for Issue #1) ─────


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 8])
def test_fa3_mla_prefill_short_sequence(seq_len):
    """REGRESSION TEST for Issue #1: illegal memory access on short prefill.

    Root cause: cache_seqlens was set to min(seq_len, topk) for ALL tokens,
    but for prefill tokens at the start of a sequence, the indexer produces
    fewer valid topk entries than cache_seqlens (due to causal masking).
    FA3 then reads -1 (invalid) entries from page_table -> crash.

    Fix: Use valid_counts from triton_convert_req_index_to_global_index
    as cache_seqlens, and clamp page_table -1 entries to 0.

    This test directly reproduces the bug scenario:
    - T tokens, each with different numbers of valid topk entries
    - Token 0 has 1 valid entry, token 1 has 2, etc.
    - page_table has 0-padding beyond valid entries (simulating clamp)
    - cache_seqlens = actual valid count per token (not min(seq_len, topk))
    """
    device = "cuda"
    T = seq_len
    topk = 2048
    kv_pool_size = 4096

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    # Simulate prefill: token i has (i+1) valid topk entries
    page_table = torch.zeros((T, topk), dtype=torch.int32, device=device)
    valid_counts = []
    for i in range(T):
        num_valid = i + 1  # causal: token i sees positions 0..i
        valid_counts.append(num_valid)
        page_table[i, :num_valid] = torch.randperm(kv_pool_size, device=device)[
            :num_valid
        ].to(torch.int32)

    cache_seqlens = torch.tensor(valid_counts, dtype=torch.int32, device=device)
    cu_seqlens_q, _ = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any(), f"NaN in output for seq_len={seq_len}"
    assert not out.isinf().any(), f"Inf in output for seq_len={seq_len}"


# ─── TEST 1.10: Page Table with Clamped -1 Entries ──────────────────


def test_fa3_mla_page_table_minus1_clamped():
    """Verify that page_table with -1 entries clamped to 0 doesn't crash.

    Tests the fix pattern: after converting topk indices to global cache
    slot IDs, -1 entries are replaced with 0 (a valid page index), and
    cache_seqlens is set to the actual valid count.
    """
    device = "cuda"
    T = 4
    topk = 256
    kv_pool_size = 1024

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    # Create page_table with -1 entries then clamp to 0
    page_table = torch.full((T, topk), -1, dtype=torch.int32, device=device)
    valid_per_token = [10, 50, 100, 200]
    for i in range(T):
        nv = valid_per_token[i]
        page_table[i, :nv] = torch.randperm(kv_pool_size, device=device)[:nv].to(
            torch.int32
        )

    page_table = page_table.clamp(min=0)
    cache_seqlens = torch.tensor(valid_per_token, dtype=torch.int32, device=device)
    cu_seqlens_q, _ = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()
    assert not out.isinf().any()


# ─── TEST 1.11: Single Token (Minimum Batch) ────────────────────────


def test_fa3_mla_single_token():
    """Edge case: single token with cache_seqlens=1."""
    device = "cuda"
    T = 1
    topk = 1
    kv_pool_size = 64

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    page_table = torch.zeros((T, topk), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([1], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.12: cu_seqlens_k_new=None ───────────────────────────────


def test_fa3_mla_cu_seqlens_k_none():
    """Verify FA3 works correctly when cu_seqlens_k_new=None.

    The fix passes None for cu_seqlens_k_new since it is unused when
    k_new is None. This test verifies the kernel accepts None.
    """
    device = "cuda"
    T = 4
    topk = 128
    kv_pool_size = 512

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=None,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.13: Valid Counts Match Cache Seqlens ─────────────────────


def test_fa3_valid_counts_correctness():
    """Verify that using valid_counts as cache_seqlens produces correct results.

    Tests the complete fix flow:
    1. Create page_table with varying valid entries per token
    2. Use valid_counts (= number of non-(-1) entries) as cache_seqlens
    3. Clamp page_table -1 to 0
    4. Run FA3 and verify correctness vs reference
    """
    device = "cuda"
    T = 4
    topk = 128
    kv_pool_size = 256

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    valid_per_token = [1, 8, 32, 64]
    page_table_raw = torch.full((T, topk), -1, dtype=torch.int32, device=device)
    for i in range(T):
        nv = valid_per_token[i]
        page_table_raw[i, :nv] = torch.randperm(kv_pool_size, device=device)[:nv].to(
            torch.int32
        )

    page_table = page_table_raw.clamp(min=0)
    cache_seqlens = torch.tensor(valid_per_token, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=1,
    )

    for b in range(T):
        nv = valid_per_token[b]
        indices_raw = page_table_raw[b, :nv]
        k_gathered = k_rope_cache[indices_raw].squeeze(1).squeeze(1)
        v_gathered = v_cache[indices_raw].squeeze(1).squeeze(1)

        q_r = q_rope[b]
        q_n = q_nope[b]
        k_full = torch.cat([v_gathered, k_gathered], dim=-1)

        k_exp = k_full.unsqueeze(1).expand(-1, NUM_HEADS, -1)
        v_exp = v_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)
        q_full = torch.cat([q_n, q_r], dim=-1)

        ref_out = (
            F.scaled_dot_product_attention(
                q_full.unsqueeze(0).unsqueeze(2),
                k_exp.transpose(0, 1).unsqueeze(0),
                v_exp.transpose(0, 1).unsqueeze(0),
                scale=SOFTMAX_SCALE,
            )
            .squeeze(0)
            .squeeze(1)
        )

        torch.testing.assert_close(
            out[b].float(),
            ref_out.float(),
            rtol=0.02,
            atol=0.02,
            msg=f"Token {b} (valid={nv}) mismatch",
        )


# ─── TEST 1.14: Max topk=2048 ───────────────────────────────────────


def test_fa3_mla_max_topk():
    """Verify with maximum topk=2048 entries."""
    device = "cuda"
    T = 2
    topk = 2048
    kv_pool_size = 4096

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any()


# ─── TEST 1.15: Mixed Batch (Prefill + Decode) ──────────────────────


def test_fa3_mla_mixed_batch_prefill_decode():
    """Mixed batch: 3 prefill tokens with causal masking + 1 decode token.

    This tests the scenario where a batch contains tokens from different
    requests with DIFFERENT numbers of valid topk entries:
    - Tokens 0-2: prefill with ascending valid counts [1, 2, 3]
    - Token 3: decode with full topk valid entries

    The fix must handle each token's valid_count independently.
    This test was added in V2-FIXED to close the mixed-batch test gap.
    """
    device = "cuda"
    T = 4
    topk = 256
    kv_pool_size = 4096

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)

    # Mixed batch: 3 prefill tokens (ascending valid) + 1 decode (full topk)
    valid_per_token = [1, 2, 3, topk]
    page_table = torch.zeros((T, topk), dtype=torch.int32, device=device)
    for i in range(T):
        nv = valid_per_token[i]
        page_table[i, :nv] = torch.randperm(kv_pool_size, device=device)[:nv].to(
            torch.int32
        )

    cache_seqlens = torch.tensor(valid_per_token, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=0,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any(), "Mixed batch output contains NaN"
    assert not out.isinf().any(), "Mixed batch output contains Inf"


# ─── TEST 1.16: End-to-end KV Cache Write → FA3 Read ────────────────


def test_fa3_mla_kv_cache_write_then_read():
    """End-to-end test: write KV to cache via concat_and_cache_mla, then
    read via FA3 with page_table and valid_counts.

    This tests the complete flow from KV cache population through FA3
    kernel invocation, verifying that the written data is correctly read
    back by FA3 when using the valid_counts fix with varying valid entries.
    Added in V3 to close the end-to-end gap.
    """
    device = "cuda"
    T = 4  # 4 tokens (simulating causal prefill)
    num_blocks = 4
    block_size = 64
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_size = kv_lora_rank + qk_rope_head_dim  # 576

    # 1) Create BF16 KV cache and write known values
    cache = torch.zeros(
        num_blocks,
        block_size,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_c_normed = torch.randn(T, kv_lora_rank, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(T, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    # Write to slots 0,1,2,3
    slot_mapping = torch.arange(T, dtype=torch.int64, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    from vllm import _custom_ops as ops

    ops.concat_and_cache_mla(
        kv_c_normed,
        k_pe.squeeze(1),
        cache,
        slot_mapping,
        kv_cache_dtype="auto",
        scale=k_scale,
    )

    # 2) Reshape cache for FA3 (page_size=1 format)
    S = num_blocks * block_size
    kv_flat = cache.reshape(S, head_size)
    c_kv = kv_flat[:, :kv_lora_rank].reshape(S, 1, 1, kv_lora_rank)
    k_rope = kv_flat[:, kv_lora_rank:].reshape(S, 1, 1, qk_rope_head_dim)

    # 3) Create Q and page_table simulating causal prefill
    q_rope = torch.randn(
        T, NUM_HEADS, qk_rope_head_dim, dtype=torch.bfloat16, device=device
    )
    q_nope = torch.randn(
        T, NUM_HEADS, kv_lora_rank, dtype=torch.bfloat16, device=device
    )

    # Causal page_table: token i sees slots 0..i, rest padded with 0
    topk = 128
    page_table = torch.zeros((T, topk), dtype=torch.int32, device=device)
    valid_per_token = []
    for i in range(T):
        nv = i + 1  # token i can attend to i+1 positions
        valid_per_token.append(nv)
        page_table[i, :nv] = torch.arange(nv, dtype=torch.int32, device=device)

    cache_seqlens = torch.tensor(valid_per_token, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    # 4) Run FA3 kernel
    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope,
        v_cache=c_kv,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=1,  # deterministic
    )

    # 5) Verify output shape, no NaN/Inf
    assert out.shape == (T, NUM_HEADS, kv_lora_rank)
    assert not out.isnan().any(), "E2E KV write+read output contains NaN"
    assert not out.isinf().any(), "E2E KV write+read output contains Inf"

    # 6) Verify vs reference: each token's output should match SDPA
    #    using the SAME KV data that was written to cache
    for b in range(T):
        nv = valid_per_token[b]
        # Gather the actual written KV from cache slots 0..nv-1
        k_gathered = k_rope[:nv].squeeze(1).squeeze(1)  # [nv, 64]
        v_gathered = c_kv[:nv].squeeze(1).squeeze(1)  # [nv, 512]

        q_r = q_rope[b]  # [N, 64]
        q_n = q_nope[b]  # [N, 512]
        k_full = torch.cat([v_gathered, k_gathered], dim=-1)  # [nv, 576]

        k_exp = k_full.unsqueeze(1).expand(-1, NUM_HEADS, -1)
        v_exp = v_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)
        q_full = torch.cat([q_n, q_r], dim=-1)

        ref_out = (
            F.scaled_dot_product_attention(
                q_full.unsqueeze(0).unsqueeze(2),
                k_exp.transpose(0, 1).unsqueeze(0),
                v_exp.transpose(0, 1).unsqueeze(0),
                scale=SOFTMAX_SCALE,
            )
            .squeeze(0)
            .squeeze(1)
        )

        torch.testing.assert_close(
            out[b].float(),
            ref_out.float(),
            rtol=0.02,
            atol=0.02,
            msg=f"E2E token {b} (valid={nv}): FA3 vs SDPA mismatch",
        )


# ─── TEST 1.17: Batch Size Gating — FA3 vs FlashMLA Routing ──────────


def test_batch_size_gating_constant():
    """Verify that MAX_BATCH_SIZE_FOR_FA3 is defined and is 16."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        MAX_BATCH_SIZE_FOR_FA3,
    )

    assert MAX_BATCH_SIZE_FOR_FA3 == 16, (
        f"MAX_BATCH_SIZE_FOR_FA3 should be 16, got {MAX_BATCH_SIZE_FOR_FA3}"
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_fa3_small_batch_correctness(batch_size):
    """Verify FA3 produces correct results for small batch sizes (<=16).

    These batch sizes should use the FA3 kernel path, which is faster
    for small batches due to lower kernel launch overhead.
    """
    device = "cuda"
    T = batch_size
    topk = 128
    kv_pool_size = 1024

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)
    k_rope_cache, v_cache = _make_kv_cache(kv_pool_size, device=device)
    page_table = _make_page_table(T, topk, kv_pool_size, device=device)
    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q, cu_seqlens_k = _make_cu_seqlens(cache_seqlens, device=device)

    out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=1,
    )

    assert out.shape == (T, NUM_HEADS, HEADDIM_V)
    assert not out.isnan().any(), f"NaN in FA3 output for batch_size={batch_size}"
    assert not out.isinf().any(), f"Inf in FA3 output for batch_size={batch_size}"

    # Verify correctness against SDPA reference for first token
    indices = page_table[0]
    k_gathered = k_rope_cache[indices].squeeze(1).squeeze(1)
    v_gathered = v_cache[indices].squeeze(1).squeeze(1)
    q_full = torch.cat([q_nope[0], q_rope[0]], dim=-1)
    k_full = torch.cat([v_gathered, k_gathered], dim=-1)
    k_exp = k_full.unsqueeze(1).expand(-1, NUM_HEADS, -1)
    v_exp = v_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)
    ref_out = (
        F.scaled_dot_product_attention(
            q_full.unsqueeze(0).unsqueeze(2),
            k_exp.transpose(0, 1).unsqueeze(0),
            v_exp.transpose(0, 1).unsqueeze(0),
            scale=SOFTMAX_SCALE,
        )
        .squeeze(0)
        .squeeze(1)
    )
    torch.testing.assert_close(
        out[0].float(),
        ref_out.float(),
        rtol=0.02,
        atol=0.02,
        msg=f"FA3 small batch (bs={batch_size}) token 0 mismatch",
    )


# ─── TEST 1.18: FlashMLA Fallback Available ──────────────────────────


def test_flashmla_sparse_fallback_available():
    """Verify that the FlashMLA BF16 sparse fallback is available on SM90."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        _flashmla_sparse_available,
    )

    # On SM90 with FlashMLA compiled, the fallback should be available
    # This test may skip if FlashMLA is not compiled (non-standard build)
    if not _flashmla_sparse_available:
        pytest.skip("FlashMLA sparse not available for fallback testing")
    assert _flashmla_sparse_available


# ─── TEST 1.19: FlashMLA BF16 Fallback Correctness ──────────────────


@pytest.mark.parametrize("batch_size", [17, 32, 64])
def test_flashmla_bf16_fallback_correctness(batch_size):
    """End-to-end correctness test for FlashMLA BF16 sparse fallback.

    Verifies that the FlashMLA BF16 sparse prefill kernel (used as fallback
    for batch sizes > MAX_BATCH_SIZE_FOR_FA3) produces correct results by
    comparing against per-token SDPA reference.

    This tests the complete fallback path:
    1. Q concatenation [ql_nope | q_pe] -> [T, N, 576]
    2. Head padding N -> 64
    3. KV reshape to [S, 1, 576]
    4. Index reshape to [T, 1, topk]
    5. flash_mla_sparse_fwd with topk_length
    6. Output unpadding

    NOTE: The FlashMLA sparse_prefill kernel requires topk % 128 == 0
    (assertion: topk % (2*B_TOPK) == 0 where B_TOPK=64). The real system
    uses topk=2048 which satisfies this. We use topk=256 for tractability.
    """
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        _FLASHMLA_SM90_HEAD_PADDING,
        _flashmla_sparse_available,
    )

    if not _flashmla_sparse_available:
        pytest.skip("FlashMLA sparse not available for fallback testing")

    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    device = "cuda"
    T = batch_size
    # topk must be divisible by 128 (kernel constraint: topk % (2*B_TOPK) == 0)
    topk = 256
    kv_pool_size = 512

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)

    # Create combined KV cache: [kv_pool_size, 1, 576]
    # Layout: [kv_c_normed(512) | k_pe(64)]
    kv_combined = torch.randn(
        kv_pool_size, 1, HEADDIM_V + HEADDIM_QK, dtype=torch.bfloat16, device=device
    )

    # Create page table with unique random indices per token
    page_table = torch.stack(
        [torch.randperm(kv_pool_size, device=device)[:topk] for _ in range(T)]
    ).to(torch.int32)

    # Use valid_counts = topk for all tokens to test full topk path.
    # In the real system, valid_counts may be < topk for prefill tokens.
    valid_counts = torch.full((T,), topk, dtype=torch.int32, device=device)

    # 1) Concatenate Q: [ql_nope(512) | q_pe(64)] -> [T, N, 576]
    q_concat = torch.cat([q_nope, q_rope], dim=-1)  # [T, N, 576]

    # 2) Pad heads to 64
    padded_heads = _FLASHMLA_SM90_HEAD_PADDING
    if padded_heads > NUM_HEADS:
        q_padded = q_concat.new_zeros((T, padded_heads, q_concat.shape[-1]))
        q_padded[:, :NUM_HEADS, :] = q_concat
        q_concat = q_padded

    # 3) Reshape indices for MQA: (T, topk) -> (T, 1, topk)
    indices = page_table.unsqueeze(1)

    # 4) Call FlashMLA BF16 sparse prefill kernel
    output = flash_mla_sparse_fwd(
        q_concat,  # [T, padded_heads, 576]
        kv_combined,  # [kv_pool_size, 1, 576]
        indices,  # [T, 1, topk]
        SOFTMAX_SCALE,  # 192**-0.5
        d_v=HEADDIM_V,  # 512
        topk_length=valid_counts,  # [T]
    )[0]

    # 5) Unpad heads
    output = output[:, :NUM_HEADS, :]  # [T, N, 512]

    assert output.shape == (T, NUM_HEADS, HEADDIM_V), (
        f"Output shape mismatch: {output.shape} vs expected ({T}, {NUM_HEADS}, {HEADDIM_V})"
    )
    assert not output.isnan().any(), (
        f"NaN in FlashMLA fallback output for bs={batch_size}"
    )
    assert not output.isinf().any(), (
        f"Inf in FlashMLA fallback output for bs={batch_size}"
    )

    # 6) Verify correctness against SDPA reference for first 4 tokens
    for b in range(min(4, T)):
        idx = page_table[b]  # [topk]
        # Gather KV from combined cache: [topk, 1, 576] -> split
        kv_gathered = kv_combined[idx].squeeze(1)  # [topk, 576]
        v_gathered = kv_gathered[:, :HEADDIM_V]  # [topk, 512] (kv_c_normed)
        k_gathered = kv_gathered  # [topk, 576] (full key)

        # Full Q for this token
        q_full = torch.cat([q_nope[b], q_rope[b]], dim=-1)  # [N, 576]

        # Expand for MQA: (topk, 1, dim) -> (topk, N, dim)
        k_expanded = k_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)  # [topk, N, 576]
        v_expanded = v_gathered.unsqueeze(1).expand(-1, NUM_HEADS, -1)  # [topk, N, 512]

        ref_out = (
            F.scaled_dot_product_attention(
                q_full.unsqueeze(0).unsqueeze(2),  # [1, N, 1, 576]
                k_expanded.transpose(0, 1).unsqueeze(0),  # [1, N, topk, 576]
                v_expanded.transpose(0, 1).unsqueeze(0),  # [1, N, topk, 512]
                scale=SOFTMAX_SCALE,
            )
            .squeeze(0)
            .squeeze(1)
        )  # [N, 512]

        torch.testing.assert_close(
            output[b].float(),
            ref_out.float(),
            rtol=0.02,
            atol=0.02,
            msg=f"FlashMLA fallback (bs={batch_size}) token {b} mismatch",
        )


# ─── TEST 1.20: FA3 vs FlashMLA Cross-Kernel Consistency ─────────────


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_fa3_vs_flashmla_cross_kernel_consistency(batch_size):
    """Verify that FA3 and FlashMLA BF16 fallback produce consistent results.

    This is the ultimate correctness test: given identical inputs, both
    the CUTLASS FA3 kernel and the FlashMLA BF16 sparse prefill kernel
    should produce the same output (within numerical tolerance).

    Uses small batch sizes where both kernels can run, with topk=256
    (must be divisible by 128 for FlashMLA's kernel constraint).

    Includes T=16 (the MAX_BATCH_SIZE_FOR_FA3 boundary) to verify both
    kernels agree at the exact gating threshold.
    """
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        _FLASHMLA_SM90_HEAD_PADDING,
        _flashmla_sparse_available,
    )

    if not _flashmla_sparse_available:
        pytest.skip("FlashMLA sparse not available for cross-kernel testing")

    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    device = "cuda"
    T = batch_size
    # topk must be divisible by 128 for FlashMLA kernel constraint
    topk = 256
    kv_pool_size = 1024

    q_rope = torch.randn(T, NUM_HEADS, HEADDIM_QK, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(T, NUM_HEADS, HEADDIM_V, dtype=torch.bfloat16, device=device)

    # Create combined KV cache in BF16: [kv_pool_size, 576]
    # This flat format is what both kernels see after reshaping
    kv_flat = torch.randn(
        kv_pool_size, HEADDIM_V + HEADDIM_QK, dtype=torch.bfloat16, device=device
    )

    # Create page table with unique random indices per token
    page_table = torch.stack(
        [torch.randperm(kv_pool_size, device=device)[:topk] for _ in range(T)]
    ).to(torch.int32)

    cache_seqlens = torch.full((T,), topk, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, T + 1, dtype=torch.int32, device=device)

    # === FA3 PATH ===
    # FA3 expects separate RoPE (k_cache) and NoPE (v_cache) in paged format
    c_kv = kv_flat[:, :HEADDIM_V].reshape(kv_pool_size, 1, 1, HEADDIM_V)
    k_rope = kv_flat[:, HEADDIM_V:].reshape(kv_pool_size, 1, 1, HEADDIM_QK)

    fa3_out = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope,
        v_cache=c_kv,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        num_splits=1,  # deterministic for comparison
    )

    # === FlashMLA BF16 PATH ===
    # FlashMLA expects concatenated Q and KV in [S, 1, 576] format
    q_concat = torch.cat([q_nope, q_rope], dim=-1)  # [T, N, 576]

    padded_heads = _FLASHMLA_SM90_HEAD_PADDING
    if padded_heads > NUM_HEADS:
        q_padded = q_concat.new_zeros((T, padded_heads, q_concat.shape[-1]))
        q_padded[:, :NUM_HEADS, :] = q_concat
        q_concat = q_padded

    kv_for_flashmla = kv_flat.reshape(kv_pool_size, 1, HEADDIM_V + HEADDIM_QK)
    indices = page_table.unsqueeze(1)  # [T, 1, topk]

    flashmla_out = flash_mla_sparse_fwd(
        q_concat,
        kv_for_flashmla,
        indices,
        SOFTMAX_SCALE,
        d_v=HEADDIM_V,
        topk_length=cache_seqlens,
    )[0]
    flashmla_out = flashmla_out[:, :NUM_HEADS, :]

    # === COMPARISON ===
    assert fa3_out.shape == flashmla_out.shape == (T, NUM_HEADS, HEADDIM_V), (
        f"Shape mismatch: FA3={fa3_out.shape}, FlashMLA={flashmla_out.shape}"
    )
    assert not fa3_out.isnan().any(), "FA3 output contains NaN"
    assert not flashmla_out.isnan().any(), "FlashMLA output contains NaN"

    # Cross-kernel comparison with slightly relaxed tolerance
    # (different kernels may have different accumulation order)
    torch.testing.assert_close(
        fa3_out.float(),
        flashmla_out.float(),
        rtol=0.03,
        atol=0.03,
        msg=f"FA3 vs FlashMLA mismatch at bs={batch_size}",
    )
