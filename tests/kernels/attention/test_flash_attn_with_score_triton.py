# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Flash Attention with score computation using Triton kernels.

This module tests the token importance computation utilities implemented
with Triton kernels.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton not available"),
]


def reference_varlen_key_importance(q, k, cu_seqlens, softmax_lse, scale=None):
    """
    Compute reference key importance for varlen (packed) sequences.

    Args:
        q: [total_tokens, heads, head_dim] - Packed sequence
        k: [total_tokens, heads, head_dim] - Packed sequence
        cu_seqlens: [batch + 1] - Cumulative sequence lengths
        softmax_lse: [heads, total_tokens] - Packed format (matches
            flash_attn_varlen_func output)
        scale: softmax scaling factor

    Returns:
        key_importance: [total_tokens, heads]
    """
    total_tokens, heads, head_dim = q.shape
    batch_size = cu_seqlens.numel() - 1

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Initialize output
    key_importance = torch.zeros(
        total_tokens, heads, dtype=torch.float32, device=q.device
    )

    # Process each sequence separately
    for b in range(batch_size):
        start_idx = cu_seqlens[b].item()
        end_idx = cu_seqlens[b + 1].item()
        _seq_len = end_idx - start_idx

        # Extract sequence
        q_seq = q[start_idx:end_idx]  # [seq_len, heads, head_dim]
        k_seq = k[start_idx:end_idx]  # [seq_len, heads, head_dim]
        lse_seq = softmax_lse[:, start_idx:end_idx]  # [heads, seq_len]

        # Transpose to [heads, seq_len, head_dim]
        q_seq = q_seq.transpose(0, 1).contiguous()  # [heads, seq_len, head_dim]
        k_seq = k_seq.transpose(0, 1).contiguous()  # [heads, seq_len, head_dim]

        # Compute attention scores: Q @ K.T
        # [heads, seq_q, head_dim] x [heads, seq_k, head_dim]^T -> [heads, seq_q, seq_k]
        qk = torch.einsum("hqd,hkd->hqk", q_seq.float(), k_seq.float()) * scale

        # Compute softmax probabilities using LSE
        # lse_seq: [heads, seq_q] -> [heads, seq_q, 1]
        lse_expanded = lse_seq.unsqueeze(-1)  # [heads, seq_q, 1]
        attention_probs = torch.exp(qk - lse_expanded)  # [heads, seq_q, seq_k]

        # Sum over queries: [heads, seq_q, seq_k] -> [heads, seq_k]
        importance_seq = attention_probs.sum(dim=1)  # [heads, seq_k]

        # Transpose back to [seq_k, heads] and store
        importance_seq = importance_seq.transpose(0, 1)  # [seq_k, heads]
        key_importance[start_idx:end_idx] = importance_seq

    return key_importance


class TestVarlenImportanceTriton:
    """Tests for compute_varlen_importance function."""

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_compute_varlen_importance_basic(self, batch_size):
        """Test varlen token importance computation."""
        from vllm.model_executor.layers.attention import (
            compute_flash_attn_score_triton as _score_triton,
        )

        device = "cuda"
        nheads = 4
        head_dim = 64

        # Create variable-length sequences
        seq_lens = [10, 20, 15][:batch_size]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)

        # Generate packed inputs
        q = torch.randn(
            total_tokens, nheads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            total_tokens, nheads, head_dim, device=device, dtype=torch.float16
        )

        # Create cu_seqlens
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.int32,
            device=device,
        )

        scale = 1.0 / math.sqrt(head_dim)

        # Compute softmax_lse for each sequence (packed format [heads, total_tokens])
        softmax_lse = torch.zeros(
            nheads, total_tokens, dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            start_idx = cu_seqlens[b].item()
            end_idx = cu_seqlens[b + 1].item()
            _seq_len = end_idx - start_idx

            q_seq = q[start_idx:end_idx]  # [seq_len, heads, head_dim]
            k_seq = k[start_idx:end_idx]  # [seq_len, heads, head_dim]

            # Transpose to [heads, seq_len, head_dim]
            q_seq = q_seq.transpose(0, 1).contiguous()
            k_seq = k_seq.transpose(0, 1).contiguous()

            # Compute logsumexp
            qk = torch.einsum("hqd,hkd->hqk", q_seq.float(), k_seq.float()) * scale
            lse = torch.logsumexp(qk, dim=-1)  # [heads, seq_len]

            softmax_lse[:, start_idx:end_idx] = lse

        # Compute token importance using Triton
        computed_importance = _score_triton.compute_varlen_importance(
            q, k, cu_seqlens, max_seqlen, softmax_lse, softmax_scale=scale
        )

        # Verify shape (should be [total_tokens] after mean over heads)
        assert computed_importance.shape == (total_tokens,)

        # Compute reference (before mean over heads)
        ref_importance = reference_varlen_key_importance(
            q, k, cu_seqlens, softmax_lse, scale
        )  # [total_tokens, heads]

        # Take mean over heads for comparison
        ref_importance_mean = ref_importance.mean(dim=1)  # [total_tokens]

        # Verify values
        torch.testing.assert_close(
            computed_importance, ref_importance_mean, rtol=1e-3, atol=1e-3
        )

    def test_compute_varlen_importance_single_sequence(self):
        """Test varlen importance with a single sequence."""
        from vllm.model_executor.layers.attention import (
            compute_flash_attn_score_triton as _score_triton,
        )

        device = "cuda"
        seq_len = 64
        nheads = 4
        head_dim = 64

        q = torch.randn(seq_len, nheads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(seq_len, nheads, head_dim, device=device, dtype=torch.float16)

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        scale = 1.0 / math.sqrt(head_dim)

        # Compute softmax_lse in packed format [heads, total_tokens]
        q_transposed = q.transpose(0, 1).contiguous()  # [heads, seq_len, head_dim]
        k_transposed = k.transpose(0, 1).contiguous()  # [heads, seq_len, head_dim]
        qk = (
            torch.einsum("hqd,hkd->hqk", q_transposed.float(), k_transposed.float())
            * scale
        )
        softmax_lse = torch.logsumexp(qk, dim=-1)  # [heads, seq_len]

        # Compute token importance
        computed_importance = _score_triton.compute_varlen_importance(
            q, k, cu_seqlens, seq_len, softmax_lse, softmax_scale=scale
        )

        # Verify shape
        assert computed_importance.shape == (seq_len,)

        # Compute reference
        ref_importance = reference_varlen_key_importance(
            q, k, cu_seqlens, softmax_lse, scale
        )
        ref_importance_mean = ref_importance.mean(dim=1)

        # Verify values
        torch.testing.assert_close(
            computed_importance, ref_importance_mean, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
