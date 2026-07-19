# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_prefill_attention import (
    _select_num_stages,
    context_attention_fwd,
)

DEVICE_TYPE = current_platform.device_type

# num_stages > 1 is only enabled on the exact CUDA capabilities the launch policy was
# validated on (SM80 A100-class, SM90 H100-class); everywhere else context_attention_fwd
# forces num_stages=1. Tests that mean to exercise the pipelined launch guard on this.
STAGE3_PIPELINE_SUPPORTED = current_platform.is_cuda() and (
    current_platform.is_device_capability(80)
    or current_platform.is_device_capability(90)
)


# block_n is passed explicitly (the launch block for that dtype on sm80+: fp32 -> 32,
# otherwise 128) so the expected num_stages is deterministic regardless of the platform
# running the test.
@pytest.mark.parametrize(
    "dtype,head_dim,max_input_len,block_n,expected",
    [
        # fp32 always keeps the historical single stage.
        (torch.float32, 128, 4096, 32, 1),
        # head_dim <= 64: tiny tiles, pipelining regresses -> single stage.
        (torch.bfloat16, 64, 4096, 128, 1),
        # head_dim > 128 (e.g. 256): num_stages>1 would exceed A100 shared memory.
        (torch.bfloat16, 256, 4096, 128, 1),
        # Short prefill (< 8 KV iterations): too few to fill a pipeline.
        (torch.bfloat16, 128, 128, 128, 1),
        (torch.bfloat16, 128, 512, 128, 1),
        # Low end of the band (8-15 KV iters, ~1024-2047) -> depth 2.
        (torch.bfloat16, 128, 1024, 128, 2),
        (torch.float16, 128, 1536, 128, 2),
        # Long BF16/FP16 prefill (>= 16 KV iters) -> depth 3 (the big win).
        (torch.float16, 96, 2048, 128, 3),
        (torch.bfloat16, 128, 4096, 128, 3),
        (torch.float16, 128, 8192, 128, 3),
    ],
)
def test_select_num_stages(dtype, head_dim, max_input_len, block_n, expected):
    """The launch policy is a pure function of shape/dtype; validate its bands.

    Runs on CPU (no GPU needed): guards that the conservative fallbacks (fp32, tiny /
    large head dims, short sequences -> num_stages=1) and the pipelined band
    (64 < head_dim <= 128 BF16/FP16 -> depth 2 for a moderate prefill, depth 3 for a
    long one) stay as intended.
    """
    assert (
        _select_num_stages(
            dtype=dtype,
            head_dim=head_dim,
            max_input_len=max_input_len,
            block_n=block_n,
        )
        == expected
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


@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
# D=256 exercises the large-head-dim path, where the launch policy must keep
# num_stages=1 (num_stages>1 would exceed shared memory / OutOfResources).
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("is_causal", [True, False])
# float16 and bfloat16 are the dtypes the num_stages policy actually pipelines
# (at D=128, max_seq_len=1024 -> num_stages=2), so both must exercise the real kernel
# launch, not just the pure-policy test above. The num_stages=3 path (>=16 KV iters) is
# covered by test_context_attention_stage3_launch below.
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
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


@pytest.mark.skipif(
    not STAGE3_PIPELINE_SUPPORTED,
    reason="num_stages=3 is enabled only on SM80/SM90 CUDA",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_context_attention_stage3_launch(dtype):
    """Compile and launch the num_stages=3 pipelined configuration on a GPU.

    The parametrized tests above use max_seq_len=1024, which now selects num_stages=2;
    this longer single sequence (>=16 K/V-loop iterations at BLOCK=128 on SM80/SM90)
    selects num_stages=3, so it provides regression protection that the depth-3 launch
    compiles, runs, and stays correct. B=1 with a small head count keeps it cheap.
    """
    torch.manual_seed(0)
    H_Q = H_KV = 8
    D = 128
    seq = 2048
    assert (
        _select_num_stages(
            dtype=dtype, head_dim=D, max_input_len=seq, block_n=128
        )
        == 3
    )

    q = torch.randn(seq, H_Q, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(seq, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(seq, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    o = torch.zeros_like(q)

    b_start_loc = torch.zeros(1, dtype=torch.int32, device=DEVICE_TYPE)
    b_seq_len = torch.tensor([seq], dtype=torch.int32, device=DEVICE_TYPE)

    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        seq,
        is_causal=True,
        sliding_window_q=None,
        sliding_window_k=None,
    )

    o_ref = ref_masked_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
