# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Triton block-sparse attention kernels.

Tests the three kernels in isolation without requiring a model or vLLM.
Validates correctness against a naive PyTorch reference implementation.
"""

# Standard
import math

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.attention.metadata import _block_mask_to_csr
from lmcache.v1.compute.attention.triton_kernels import (
    block_sparse_attention,
    causal_prefill_attention,
    merge_attention_outputs,
)

# Skip entire module when no GPU available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_attention(q, k, v, sm_scale, causal=False):
    """Naive PyTorch attention (no sparsity)."""
    # q: [M, H, D], k: [N, H, D], v: [N, H, D]
    M, H, D = q.shape
    N = k.shape[0]
    # [H, M, N]
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    if causal:
        # kv_pos <= q_pos
        q_pos = torch.arange(M, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(N, device=q.device).unsqueeze(0)
        mask = kv_pos <= q_pos
        scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,nhd->mhd", attn, v.float())
    return out.to(q.dtype)


def _ref_sparse_attention(q, k, v, block_indices, block_indptr, sm_scale, block_size):
    """Naive PyTorch sparse attention for reference."""
    M, H, D = q.shape
    N = k.shape[0]
    KV_H = k.shape[1]
    gqa_ratio = H // KV_H

    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((M, H), float("-inf"), dtype=torch.float32, device=q.device)

    num_q_blocks = block_indptr.shape[0] - 1
    for qb in range(num_q_blocks):
        q_start = qb * block_size
        q_end = min(q_start + block_size, M)
        kv_start_idx = block_indptr[qb].item()
        kv_end_idx = block_indptr[qb + 1].item()
        if kv_end_idx <= kv_start_idx:
            continue
        for h in range(H):
            kv_h = h // gqa_ratio
            q_block = q[q_start:q_end, h, :].float()  # [bm, D]
            all_scores = []
            all_v = []
            for idx in range(kv_start_idx, kv_end_idx):
                kb = block_indices[idx].item()
                kv_s = kb * block_size
                kv_e = min(kv_s + block_size, N)
                k_block = k[kv_s:kv_e, kv_h, :].float()  # [bn, D]
                v_block = v[kv_s:kv_e, kv_h, :].float()
                s = (q_block @ k_block.T) * sm_scale  # [bm, bn]
                all_scores.append(s)
                all_v.append(v_block)
            # concat and softmax
            scores = torch.cat(all_scores, dim=1)  # [bm, total_kv]
            vals = torch.cat(all_v, dim=0)  # [total_kv, D]
            attn = torch.softmax(scores, dim=-1)
            o = attn @ vals  # [bm, D]
            out[q_start:q_end, h, :] = o
            lse[q_start:q_end, h] = scores.max(dim=1).values + torch.log(
                torch.exp(scores - scores.max(dim=1, keepdim=True).values).sum(dim=1)
            )
    return out.to(q.dtype), lse


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCausalPrefillAttention:
    """Tests for _causal_prefill_attention_kernel."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seq_len", [64, 128, 200])
    def test_matches_reference(self, dtype, seq_len):
        H, D = 8, 64
        q = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
        k = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
        v = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
        sm_scale = 1.0 / math.sqrt(D)

        out = causal_prefill_attention(q, k, v, sm_scale, block_size=64)
        ref = _ref_attention(q, k, v, sm_scale, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gqa(self, dtype):
        """Test grouped-query attention (H > KV_H)."""
        seq_len, H, KV_H, D = 64, 8, 2, 64
        q = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
        k = torch.randn(seq_len, KV_H, D, device="cuda", dtype=dtype)
        v = torch.randn(seq_len, KV_H, D, device="cuda", dtype=dtype)
        sm_scale = 1.0 / math.sqrt(D)

        out = causal_prefill_attention(q, k, v, sm_scale, block_size=64)
        assert out.shape == (seq_len, H, D)
        assert not torch.isnan(out).any()

    def test_output_dtype_matches_input(self):
        seq_len, H, D = 64, 4, 64
        for dtype in [torch.float16, torch.bfloat16]:
            q = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
            k = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
            v = torch.randn(seq_len, H, D, device="cuda", dtype=dtype)
            out = causal_prefill_attention(q, k, v, 1.0 / math.sqrt(D))
            assert out.dtype == dtype


class TestBlockSparseAttention:
    """Tests for block_sparse_attention kernel."""

    def _make_full_mask_csr(self, num_q_blocks, num_kv_blocks, device):
        """CSR for a fully dense mask (all blocks attended)."""
        mask = torch.ones(num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
        return _block_mask_to_csr(mask, device)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_full_mask_matches_dense(self, dtype):
        """With all blocks attended, should match dense attention."""
        M, N, H, D = 64, 128, 4, 64
        BS = 32
        q = torch.randn(M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(N, H, D, device="cuda", dtype=dtype)
        v = torch.randn(N, H, D, device="cuda", dtype=dtype)
        sm_scale = 1.0 / math.sqrt(D)

        num_q_blocks = (M + BS - 1) // BS
        num_kv_blocks = (N + BS - 1) // BS
        indices, indptr = self._make_full_mask_csr(
            num_q_blocks, num_kv_blocks, q.device
        )

        out, lse = block_sparse_attention(
            q, k, v, indices, indptr, sm_scale, block_size=BS
        )
        ref = _ref_attention(q, k, v, sm_scale, causal=False)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        assert lse.shape == (M, H)
        assert not torch.isnan(lse).any()

    def test_empty_mask_returns_zeros(self):
        """All-empty CSR → output zeros, LSE -inf."""
        M, N, H, D = 64, 128, 4, 64
        BS = 32
        q = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(N, H, D, device="cuda", dtype=torch.float16)

        num_q_blocks = (M + BS - 1) // BS
        indptr = torch.zeros(num_q_blocks + 1, dtype=torch.int32, device="cuda")
        indices = torch.zeros(0, dtype=torch.int32, device="cuda")

        out, lse = block_sparse_attention(
            q, k, v, indices, indptr, 1.0 / math.sqrt(D), block_size=BS
        )
        assert torch.all(out == 0)
        assert torch.all(lse == float("-inf"))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gqa_sparse(self, dtype):
        """GQA with sparse mask."""
        M, N, H, KV_H, D = 64, 128, 8, 2, 64
        BS = 32
        q = torch.randn(M, H, D, device="cuda", dtype=dtype)
        k = torch.randn(N, KV_H, D, device="cuda", dtype=dtype)
        v = torch.randn(N, KV_H, D, device="cuda", dtype=dtype)

        num_q_blocks = (M + BS - 1) // BS
        num_kv_blocks = (N + BS - 1) // BS
        indices, indptr = self._make_full_mask_csr(
            num_q_blocks, num_kv_blocks, q.device
        )

        out, lse = block_sparse_attention(
            q, k, v, indices, indptr, 1.0 / math.sqrt(D), block_size=BS
        )
        assert out.shape == (M, H, D)
        assert not torch.isnan(out).any()

    def test_partial_trailing_block(self):
        """Non-aligned seq_len (trailing partial block)."""
        M, N, H, D = 100, 128, 4, 64  # 100 not divisible by 32
        BS = 32
        q = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(N, H, D, device="cuda", dtype=torch.float16)

        num_q_blocks = (M + BS - 1) // BS  # 4 blocks, last has 4 rows
        num_kv_blocks = (N + BS - 1) // BS
        indices, indptr = self._make_full_mask_csr(
            num_q_blocks, num_kv_blocks, q.device
        )

        out, lse = block_sparse_attention(
            q, k, v, indices, indptr, 1.0 / math.sqrt(D), block_size=BS
        )
        assert out.shape == (M, H, D)
        assert not torch.isnan(out).any()


class TestMergeAttentionOutputs:
    """Tests for _merge_attention_kernel."""

    def test_one_path_inf(self):
        """When one LSE is -inf, output should be entirely from the other."""
        M, H, D = 64, 4, 64
        o1 = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        o2 = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        lse1 = torch.full((M, H), float("-inf"), dtype=torch.float32, device="cuda")
        lse2 = torch.ones(M, H, dtype=torch.float32, device="cuda")

        merged = merge_attention_outputs(o1, lse1, o2, lse2)
        torch.testing.assert_close(merged, o2, atol=1e-3, rtol=1e-3)

    def test_both_inf_no_nan(self):
        """When both LSEs are -inf, output should be zero (not NaN)."""
        M, H, D = 64, 4, 64
        o1 = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        o2 = torch.randn(M, H, D, device="cuda", dtype=torch.float16)
        lse1 = torch.full((M, H), float("-inf"), dtype=torch.float32, device="cuda")
        lse2 = torch.full((M, H), float("-inf"), dtype=torch.float32, device="cuda")

        merged = merge_attention_outputs(o1, lse1, o2, lse2)
        assert not torch.isnan(merged).any()

    def test_equal_lse_averages(self):
        """When LSE values are equal, output should be average of both."""
        M, H, D = 32, 2, 64
        o1 = torch.ones(M, H, D, device="cuda", dtype=torch.float16)
        o2 = torch.ones(M, H, D, device="cuda", dtype=torch.float16) * 3
        lse = torch.ones(M, H, dtype=torch.float32, device="cuda")

        merged = merge_attention_outputs(o1, lse, o2, lse)
        expected = torch.ones_like(o1) * 2  # (1 + 3) / 2
        torch.testing.assert_close(merged, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_output_dtype(self, dtype):
        M, H, D = 32, 4, 64
        o1 = torch.randn(M, H, D, device="cuda", dtype=dtype)
        o2 = torch.randn(M, H, D, device="cuda", dtype=dtype)
        lse1 = torch.randn(M, H, dtype=torch.float32, device="cuda")
        lse2 = torch.randn(M, H, dtype=torch.float32, device="cuda")
        merged = merge_attention_outputs(o1, lse1, o2, lse2)
        assert merged.dtype == dtype


class TestBlockMaskToCsr:
    """Tests for _block_mask_to_csr helper."""

    def test_full_mask(self):
        mask = torch.ones(3, 4, dtype=torch.bool, device="cuda")
        indices, indptr = _block_mask_to_csr(mask, mask.device)
        assert indptr.tolist() == [0, 4, 8, 12]
        assert indices.tolist() == [0, 1, 2, 3] * 3

    def test_empty_mask(self):
        mask = torch.zeros(3, 4, dtype=torch.bool, device="cuda")
        indices, indptr = _block_mask_to_csr(mask, mask.device)
        assert indptr.tolist() == [0, 0, 0, 0]
        assert len(indices) == 0

    def test_diagonal_mask(self):
        mask = torch.eye(3, dtype=torch.bool, device="cuda")
        indices, indptr = _block_mask_to_csr(mask, mask.device)
        assert indptr.tolist() == [0, 1, 2, 3]
        assert indices.tolist() == [0, 1, 2]
