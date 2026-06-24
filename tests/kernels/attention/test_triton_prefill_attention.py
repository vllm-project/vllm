# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_prefill_attention import (
    context_attention_fwd,
    get_block_size,
)

DEVICE_TYPE = current_platform.device_type


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Reference implementation using PyTorch SDPA."""
    # q, k, v: [total_tokens, num_heads, head_dim]
    # SDPA expects [batch, num_heads, seq_len, head_dim]

    orig_dtype = q.dtype
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

    if softcap > 0:
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        scores = torch.tanh(scores / softcap) * softcap

        if attn_mask is not None:
            scores += attn_mask[None, None, :, :].float()
        elif use_causal:
            pos_q = torch.arange(total_tokens, device=q.device).unsqueeze(1)
            pos_k = torch.arange(total_tokens, device=q.device).unsqueeze(0)
            causal_mask = pos_q >= pos_k
            scores = scores.masked_fill(~causal_mask[None, None, :, :], -torch.inf)

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_f).to(orig_dtype)
        return output.transpose(1, 2).squeeze(0)

    # Use SDPA
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=use_causal, dropout_p=0.0
    )

    # Convert back to original shape: [total_tokens, num_heads, head_dim]
    output = output.transpose(1, 2).squeeze(0)

    return output


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
    seq_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (B,), device=DEVICE_TYPE
    )
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
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


@pytest.mark.skipif(
    not (
        current_platform.is_cuda_alike() and current_platform.has_device_capability(80)
    ),
    reason="head-dim dependent block selection is specific to SM80+ CUDA-like GPUs",
)
@pytest.mark.parametrize(
    ("dtype", "head_dim", "expected_block"),
    [
        (torch.bfloat16, 128, 128),
        (torch.bfloat16, 256, 64),
        (torch.float16, 384, 32),
        (torch.float32, 128, 32),
    ],
)
def test_context_attention_block_size_policy(
    dtype: torch.dtype,
    head_dim: int,
    expected_block: int,
):
    assert get_block_size(dtype, head_dim) == expected_block


@pytest.mark.skipif(
    not (
        current_platform.is_cuda_alike() and current_platform.has_device_capability(80)
    ),
    reason="D=256 prefill launch policy is specific to SM80+ CUDA-like GPUs",
)
@torch.inference_mode()
def test_context_attention_head_dim_256_launches():
    torch.manual_seed(42)

    seq_len = 128
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 256
    dtype = torch.bfloat16

    b_start_loc = torch.tensor([0], dtype=torch.int32, device=DEVICE_TYPE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE_TYPE)
    q = torch.randn(seq_len, num_query_heads, head_dim, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn_like(k)
    o = torch.zeros_like(q)

    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        seq_len,
        is_causal=True,
        sliding_window_q=None,
        sliding_window_k=None,
    )

    kv_group_num = num_query_heads // num_kv_heads
    o_ref = ref_masked_attention(
        q,
        k.repeat_interleave(kv_group_num, dim=1),
        v.repeat_interleave(kv_group_num, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("sliding_window", [(None, None), (64, 0)])
@torch.inference_mode()
def test_context_attention_softcap(
    D: int,
    sliding_window: tuple[int | None, int | None],
):
    torch.manual_seed(42)

    seq_len = 257
    num_query_heads = 4
    num_kv_heads = 2
    dtype = torch.bfloat16
    softcap = 30.0
    sliding_window_q, sliding_window_k = sliding_window

    b_start_loc = torch.tensor([0], dtype=torch.int32, device=DEVICE_TYPE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE_TYPE)
    q = torch.randn(seq_len, num_query_heads, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(seq_len, num_kv_heads, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn_like(k)
    o = torch.zeros_like(q)

    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        seq_len,
        is_causal=True,
        softcap=softcap,
        sliding_window_q=sliding_window_q,
        sliding_window_k=sliding_window_k,
    )

    kv_group_num = num_query_heads // num_kv_heads
    o_ref = ref_masked_attention(
        q,
        k.repeat_interleave(kv_group_num, dim=1),
        v.repeat_interleave(kv_group_num, dim=1),
        is_causal=True,
        sliding_window_q=sliding_window_q,
        sliding_window_k=sliding_window_k,
        softcap=softcap,
    )
    torch.testing.assert_close(o, o_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [72, 128, 256, 320])
@pytest.mark.parametrize("sliding_window", [(32, 32), (32, 0), (0, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_context_attention_sliding_window(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    sliding_window: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
):
    sliding_window_q, sliding_window_k = sliding_window
    """Test context attention with sliding window."""
    torch.manual_seed(42)

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (B,), device=DEVICE_TYPE
    )
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
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
            is_causal=is_causal,
            sliding_window_q=sliding_window_q if sliding_window_q > 0 else None,
            sliding_window_k=sliding_window_k if sliding_window_k > 0 else None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)
