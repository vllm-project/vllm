# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


@triton.jit
def _compute_key_importance_varlen_kernel(
    Q,
    K,
    LSE,
    Out_Score,
    CuSeqLens,
    n_heads,
    stride_q_tok,
    stride_q_h,
    stride_q_d,
    stride_k_tok,
    stride_k_h,
    stride_k_d,
    stride_lse_h,
    stride_lse_tok,
    stride_out_tok,
    sm_scale,
    ACTUAL_HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Key importance (column-sum of attention) for variable-length sequences.

    Layout: Q/K [Total_Tokens, Heads, Dim],
            LSE [Heads, Total_Tokens] (packed, matches flash_attn_varlen_func).
    BLOCK_D is ACTUAL_HEAD_DIM rounded up to the next power of 2.
    """
    pid_n = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    start_idx = tl.load(CuSeqLens + pid_seq)
    end_idx = tl.load(CuSeqLens + pid_seq + 1)
    seq_len = end_idx - start_idx

    if pid_n * BLOCK_N >= seq_len:
        return

    # Load K block [BLOCK_D, BLOCK_N] (transposed for dot product)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = (offs_n[None, :] < seq_len) & (offs_d[:, None] < ACTUAL_HEAD_DIM)
    k_ptrs = K + (
        (start_idx + offs_n[None, :]) * stride_k_tok
        + pid_head * stride_k_h
        + offs_d[:, None] * stride_k_d
    )
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    acc_score = tl.zeros([BLOCK_N], dtype=tl.float32)

    lse_base_ptr = LSE + (pid_head * stride_lse_h + start_idx * stride_lse_tok)
    q_base_ptr = Q + (start_idx * stride_q_tok + pid_head * stride_q_h)

    # Iterate over all query blocks within this sequence
    for start_m in range(0, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < ACTUAL_HEAD_DIM)
        q_ptrs = q_base_ptr + (
            offs_m[:, None] * stride_q_tok + offs_d[None, :] * stride_q_d
        )
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        lse_ptrs = lse_base_ptr + offs_m * stride_lse_tok
        lse_block = tl.load(lse_ptrs, mask=offs_m < seq_len, other=0.0)

        # S = Q @ K^T, P = exp(S - LSE)
        qk = tl.dot(q_block, k_block)
        qk *= sm_scale
        p_block = tl.exp(qk - lse_block[:, None])
        p_block = tl.where(offs_m[:, None] < seq_len, p_block, 0.0)

        # Column sum (sum over queries)
        acc_score += tl.sum(p_block, axis=0)

    # Average over heads and write back
    avg_score = acc_score / n_heads
    out_ptrs = Out_Score + (start_idx + offs_n)
    tl.atomic_add(out_ptrs, avg_score, mask=offs_n < seq_len)


def compute_varlen_importance(
    q, k, cu_seqlens, max_seqlen, softmax_lse, softmax_scale=None
):
    """Compute per-token importance for variable-length (packed) sequences.

    Args:
        q: [Total_Tokens, Heads, Dim]
        k: [Total_Tokens, Heads, Dim]
        cu_seqlens: [Batch + 1] cumulative sequence lengths (int32)
        max_seqlen: maximum sequence length (for grid sizing)
        softmax_lse: [Heads, Total_Tokens] from flash_attn_varlen_func
        softmax_scale: 1/sqrt(head_dim) by default

    Returns:
        token_importance: [Total_Tokens] importance averaged over heads
    """
    assert q.dim() == 3, "Q should be packed [Total_Tokens, Heads, Dim]"
    total_tokens, n_heads, head_dim = q.shape
    batch_size = cu_seqlens.numel() - 1

    assert softmax_lse.dim() == 2, (
        f"LSE should be 2D [Heads, Total_Tokens], got {softmax_lse.dim()}D"
    )
    assert softmax_lse.shape[0] == n_heads, (
        f"LSE heads mismatch: {softmax_lse.shape[0]} vs {n_heads}"
    )
    assert softmax_lse.shape[1] == total_tokens, (
        f"LSE tokens mismatch: {softmax_lse.shape[1]} vs {total_tokens}"
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    out_score = torch.zeros((total_tokens,), dtype=torch.float32, device=q.device)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(max_seqlen, BLOCK_N), n_heads, batch_size)

    BLOCK_D = _next_power_of_2(head_dim)
    _compute_key_importance_varlen_kernel[grid](
        q,
        k,
        softmax_lse,
        out_score,
        cu_seqlens,
        n_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        out_score.stride(0),
        softmax_scale,
        ACTUAL_HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out_score
