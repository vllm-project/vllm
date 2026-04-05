# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.utils.math_utils import RCP_LN2
from vllm.v1.attention.ops.triton_prefill_attention import (
    context_attention_fwd,
    get_block_size,
)


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
) -> torch.Tensor:
    """Reference implementation using PyTorch SDPA."""
    # q, k, v: [total_tokens, num_heads, head_dim]
    # SDPA expects [batch, num_heads, seq_len, head_dim]

    total_tokens = q.shape[0]

    # Add batch dimension and transpose
    q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]
    k = k.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]
    v = v.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]

    # Create attention mask if needed
    attn_mask = None
    use_causal = is_causal

    # If we have sliding window or need custom masking, create explicit mask
    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0
    if (sliding_window_q > 0) or (sliding_window_k > 0):
        # Position indices
        pos_q = torch.arange(total_tokens, device=q.device).unsqueeze(1)
        pos_k = torch.arange(total_tokens, device=q.device).unsqueeze(0)

        # Start with valid mask (False = no masking)
        mask = torch.ones(
            (total_tokens, total_tokens), dtype=torch.bool, device=q.device
        )

        # Apply causal mask
        if is_causal:
            mask = mask & (pos_q >= pos_k)

        # Apply sliding window masks
        sliding_window_mask = torch.ones_like(mask)
        if sliding_window_q > 0:
            sliding_window_mask &= pos_q - pos_k <= sliding_window_q

        if sliding_window_k > 0:
            sliding_window_mask &= pos_k - pos_q <= sliding_window_k

        mask = mask & sliding_window_mask

        attn_mask = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
        use_causal = False  # Don't use is_causal when providing explicit mask

    # Use SDPA
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=use_causal, dropout_p=0.0
    )

    # Convert back to original shape: [total_tokens, num_heads, head_dim]
    output = output.transpose(1, 2).squeeze(0)

    return output


def ref_skip_softmax_threshold_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    threshold_scale_factor: float = 0.0,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
) -> torch.Tensor:
    """Reference for thresholded Skip-Softmax (BLASST-style) attention."""
    total_tokens, num_heads, head_dim = q.shape
    assert k.shape == (total_tokens, num_heads, head_dim)
    assert v.shape == (total_tokens, num_heads, head_dim)

    block = get_block_size(q.dtype)
    sm_scale = (1.0 / (head_dim**0.5)) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2
    skip_threshold = threshold_scale_factor / max(total_tokens, 1)

    out = torch.zeros_like(q)

    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0

    for h in range(num_heads):
        qh = q[:, h, :].float()
        kh = k[:, h, :].float()
        vh = v[:, h, :].float()

        for start_m in range(0, total_tokens, block):
            end_m = min(start_m + block, total_tokens)
            q_block = qh[start_m:end_m]  # [M, D]
            m_len = q_block.shape[0]

            m_i = torch.full((m_len,), float("-inf"), device=q.device)
            l_i = torch.zeros((m_len,), device=q.device)
            acc = torch.zeros((m_len, head_dim), device=q.device)

            end_n = total_tokens
            if is_causal:
                end_n = min(end_n, (start_m + 1) * block)

            for start_n in range(0, end_n, block):
                end_n_blk = min(start_n + block, end_n)
                k_block = kh[start_n:end_n_blk]  # [N, D]
                v_block = vh[start_n:end_n_blk]  # [N, D]
                n_len = k_block.shape[0]

                qk = torch.matmul(q_block, k_block.transpose(0, 1))  # [M, N]

                pos_q = (start_m + torch.arange(m_len, device=q.device))[:, None]
                pos_k = (start_n + torch.arange(n_len, device=q.device))[None, :]
                mask = torch.ones((m_len, n_len), dtype=torch.bool, device=q.device)
                if is_causal:
                    mask &= pos_q >= pos_k
                if sliding_window_q > 0:
                    mask &= pos_q - pos_k <= sliding_window_q
                if sliding_window_k > 0:
                    mask &= pos_k - pos_q <= sliding_window_k

                qk = torch.where(mask, qk * sm_scale, torch.full_like(qk, -1.0e8))
                local_max = qk.max(dim=1).values

                skip_rows = torch.zeros((m_len,), dtype=torch.bool, device=q.device)
                if skip_threshold > 0.0:
                    has_running_max = torch.isfinite(m_i)
                    rel_contrib = torch.exp2(local_max - m_i)
                    skip_rows = has_running_max & (rel_contrib < skip_threshold)
                    if torch.all(skip_rows):
                        continue

                m_ij = torch.maximum(m_i, local_max)
                p = torch.exp2(qk - m_ij[:, None])
                l_ij = p.sum(dim=1)
                alpha = torch.exp2(m_i - m_ij)

                if skip_threshold > 0.0:
                    p = torch.where(skip_rows[:, None], torch.zeros_like(p), p)
                    l_ij = torch.where(skip_rows, torch.zeros_like(l_ij), l_ij)
                    alpha = torch.where(skip_rows, torch.ones_like(alpha), alpha)
                    m_ij = torch.where(skip_rows, m_i, m_ij)

                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                acc = acc + torch.matmul(p, v_block)
                m_i = m_ij

            acc = acc / l_i[:, None]
            out[start_m:end_m, h, :] = acc.to(out.dtype)

    return out


@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    is_causal: bool,
    dtype: torch.dtype,
):
    """Test basic context attention without sliding window."""
    torch.manual_seed(42)

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (B,), device="cuda")
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Call Triton kernel
    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        max_seq_len,
        is_causal=is_causal,
        sliding_window_q=None,
        sliding_window_k=None,
    )

    # Compute reference output for each sequence in batch
    o_ref = torch.zeros_like(q)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        # Expand KV heads if using GQA
        if H_Q != H_KV:
            kv_group_num = H_Q // H_KV
            k_seq = k_seq.repeat_interleave(kv_group_num, dim=1)
            v_seq = v_seq.repeat_interleave(kv_group_num, dim=1)

        o_ref[start:end] = ref_masked_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=is_causal,
            sliding_window_q=None,
            sliding_window_k=None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("sliding_window", [(32, 32), (32, 0), (0, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention_sliding_window(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    sliding_window: tuple[int, int],
    dtype: torch.dtype,
):
    sliding_window_q, sliding_window_k = sliding_window
    """Test context attention with sliding window."""
    torch.manual_seed(42)

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (B,), device="cuda")
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Call Triton kernel
    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        max_seq_len,
        is_causal=False,
        sliding_window_q=sliding_window_q,
        sliding_window_k=sliding_window_k,
    )

    # Compute reference output for each sequence in batch
    o_ref = torch.zeros_like(q)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        # Expand KV heads if using GQA
        if H_Q != H_KV:
            kv_group_num = H_Q // H_KV
            k_seq = k_seq.repeat_interleave(kv_group_num, dim=1)
            v_seq = v_seq.repeat_interleave(kv_group_num, dim=1)

        o_ref[start:end] = ref_masked_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=False,
            sliding_window_q=sliding_window_q if sliding_window_q > 0 else None,
            sliding_window_k=sliding_window_k if sliding_window_k > 0 else None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("B", [3])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention_skip_softmax_threshold(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    is_causal: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(1234)
    threshold_scale_factor = 1000.0

    seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (B,), device="cuda")
    total_tokens = seq_lens.sum().item()

    b_start_loc = torch.zeros(B, dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        max_seq_len,
        is_causal=is_causal,
        sliding_window_q=None,
        sliding_window_k=None,
        skip_softmax_threshold_scale=threshold_scale_factor,
    )

    o_ref = torch.zeros_like(q)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        if H_Q != H_KV:
            kv_group_num = H_Q // H_KV
            k_seq = k_seq.repeat_interleave(kv_group_num, dim=1)
            v_seq = v_seq.repeat_interleave(kv_group_num, dim=1)

        o_ref[start:end] = ref_skip_softmax_threshold_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=is_causal,
            threshold_scale_factor=threshold_scale_factor,
        )

    torch.testing.assert_close(o, o_ref, rtol=3e-2, atol=3e-2)
