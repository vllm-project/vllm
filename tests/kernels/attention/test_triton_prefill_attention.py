# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.attention.ops.triton_prefill_attention import context_attention_fwd


def ref_masked_attention(
    q, k, v, is_causal=True, sliding_window_q=None, sliding_window_k=None
):
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
    if (sliding_window_q is not None and sliding_window_q > 0) or (
        sliding_window_k is not None and sliding_window_k > 0
    ):
        # Position indices
        pos_q = torch.arange(total_tokens, device=q.device).unsqueeze(1)
        pos_k = torch.arange(total_tokens, device=q.device).unsqueeze(0)

        # Start with valid mask (False = no masking)
        mask = torch.zeros(
            (total_tokens, total_tokens), dtype=torch.bool, device=q.device
        )

        # Apply causal mask
        if is_causal:
            mask = mask | (pos_k > pos_q)

        # Apply sliding window masks
        if sliding_window_q is not None and sliding_window_q > 0:
            mask = mask | (pos_k > pos_q + sliding_window_q)

        if sliding_window_k is not None and sliding_window_k > 0:
            mask = mask | (pos_k < pos_q - sliding_window_k)

        # Convert bool mask to float (SDPA expects float mask)
        # True (masked) -> -inf, False (not masked) -> 0
        attn_mask = torch.zeros(
            (total_tokens, total_tokens), dtype=q.dtype, device=q.device
        )
        attn_mask.masked_fill_(mask, float("-inf"))
        use_causal = False  # Don't use is_causal when providing explicit mask

    # Use SDPA
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=use_causal, dropout_p=0.0
    )

    # Convert back to original shape: [total_tokens, num_heads, head_dim]
    output = output.transpose(1, 2).squeeze(0)

    return output


@pytest.mark.parametrize("B", [1, 3, 5])
@pytest.mark.parametrize("max_seq_len", [128, 512, 1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention(B, max_seq_len, H_Q, H_KV, D, is_causal, dtype):
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


@pytest.mark.parametrize("B", [2, 4])
@pytest.mark.parametrize("max_seq_len", [256, 512])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("sliding_window", [(16, 16), (32, 0), (0, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_context_attention_sliding_window(
    B, max_seq_len, H_Q, H_KV, D, sliding_window, dtype
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
        is_causal=True,
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
            is_causal=True,
            sliding_window_q=sliding_window_q if sliding_window_q > 0 else None,
            sliding_window_k=sliding_window_k if sliding_window_k > 0 else None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
