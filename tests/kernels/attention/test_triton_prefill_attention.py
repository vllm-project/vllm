# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops import triton_prefill_attention as prefill_attn
from vllm.v1.attention.ops.triton_prefill_attention import (
    _get_head_dim_blocks,
    _split_head_dim,
    context_attention_fwd,
)

DEVICE_TYPE = current_platform.device_type


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


@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 72, 80, 96, 128])
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


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 72, 80, 96, 128])
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


def test_split_head_dim_never_widens_the_dot():
    """Whatever the head dim, the split is never worse than one block.

    A tail pass costs an extra dot, two extra loads and an accumulator, so it
    has to buy a strictly narrower extent or not happen at all.
    """
    for Lk in range(1, 513):
        main, tail = _split_head_dim(Lk)
        npo2 = triton.next_power_of_2(Lk)

        assert main & (main - 1) == 0, Lk
        assert tail & (tail - 1) == 0, Lk
        assert main + tail >= Lk, f"Lk={Lk} is not covered"

        if tail == 0:
            assert main == npo2, Lk
        else:
            assert main + tail < npo2, f"Lk={Lk} tail pass does not narrow"


def test_split_head_dim_vit_shapes():
    """Real vision head dims, covering both tail widths the kernel can emit."""
    assert _split_head_dim(72) == (64, 16)  # SigLIP, Qwen3-VL
    assert _split_head_dim(80) == (64, 16)  # Qwen2.5-VL
    assert _split_head_dim(96) == (64, 32)


@pytest.mark.parametrize("Lk", [12, 24, 100, 112, 120])
def test_split_head_dim_declines_when_it_would_not_pay(Lk: int):
    """Head dims whose tail block would cover no fewer lanes keep one block."""
    assert _split_head_dim(Lk) == (triton.next_power_of_2(Lk), 0)


@pytest.mark.parametrize("on_gfx115x", [True, False])
def test_get_head_dim_blocks_is_gated(monkeypatch: pytest.MonkeyPatch, on_gfx115x):
    """The split is only taken where it has been measured."""
    monkeypatch.setattr(prefill_attn, "_ON_GFX115X", on_gfx115x)
    assert _get_head_dim_blocks(72) == ((64, 16) if on_gfx115x else (128, 0))
    assert _get_head_dim_blocks(128) == (128, 0)


@pytest.mark.parametrize("D", [72, 96])
def test_context_attention_split_d_forced(monkeypatch: pytest.MonkeyPatch, D: int):
    """Exercise the split-D kernel even off gfx115x, so CI always covers it."""
    monkeypatch.setattr(prefill_attn, "_ON_GFX115X", True)
    assert _get_head_dim_blocks(D)[1] > 0

    torch.manual_seed(42)
    B, S, H = 2, 256, 8
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    q, k, v = (
        torch.randn(B * S, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)
        for _ in range(3)
    )
    o = torch.zeros_like(q)
    context_attention_fwd(q, k, v, o, b_start_loc, seq_lens, S, is_causal=True)

    o_ref = torch.zeros_like(o)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()
        o_ref[start:end] = ref_masked_attention(
            q[start:end], k[start:end], v[start:end], is_causal=True
        )

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
