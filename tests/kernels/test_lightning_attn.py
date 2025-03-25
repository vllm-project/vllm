# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import (
    lightning_attention, lightning_attention2_parallel,
    linear_decode_forward_triton)
from vllm.platforms import current_platform

NUM_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16, 128]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention(
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")

    output, kv = lightning_attention(q, k, v, ed)

    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads

    output2, kv2 = lightning_attention2_parallel(q, k, v, ed)

    torch.testing.assert_close(output, output2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(kv, kv2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention_with_kv_history(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    seq_len = 32

    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")

    kv_history = torch.randn(batch_size,
                             num_heads,
                             head_size,
                             head_size,
                             dtype=torch.float32,
                             device="cuda")

    output, kv = lightning_attention(q, k, v, ed, kv_history=kv_history)

    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = torch.randn(batch_size,
                            num_heads,
                            head_size,
                            head_size,
                            dtype=dtype,
                            device="cuda")

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.arange(batch_size, device="cuda")

    output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate,
                                          slot_idx)

    assert output.shape == (batch_size, num_heads * head_size)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton_with_padding(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    batch_size = 4

    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = torch.randn(batch_size,
                            num_heads,
                            head_size,
                            head_size,
                            dtype=dtype,
                            device="cuda")

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.tensor([0, 1, -1, 2], device="cuda")

    output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate,
                                          slot_idx)

    assert output.shape == (batch_size, num_heads * head_size)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention_vs_reference(
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    """Test lightning attention against reference implementation"""
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")

    # Using lightning attention implementation
    lightning_output, _ = lightning_attention(q, k, v, ed)

    # Reference implementation: attention with exponential decay
    def reference_lightning_attention(q, k, v, ed):
        b, h, n, d = q.shape
        # Convert to float32 for better precision
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()

        # Create output tensor
        output = torch.zeros_like(q_f)

        # Compute separately for each batch and head
        for bi in range(b):
            for hi in range(h):
                decay_rate = ed[hi].item()

                # Compute attention for each query position
                for qi in range(n):
                    # Only consider causal key-value pairs (qi >= ki)
                    for ki in range(qi + 1):
                        # Calculate exponential decay
                        # based on position difference
                        position_diff = qi - ki
                        decay = torch.exp(-decay_rate * position_diff)

                        # Compute dot product of query and key
                        qk = torch.sum(q_f[bi, hi, qi] * k_f[bi, hi, ki])

                        # Apply decay and accumulate to output
                        output[bi, hi, qi] += decay * qk * v_f[bi, hi, ki]

        return output.to(q.dtype)

    reference_output = reference_lightning_attention(q, k, v, ed)

    # Compare results from both implementations
    # Using relaxed tolerances due to
    # algorithmic approximations and numerical precision differences
    torch.testing.assert_close(lightning_output,
                               reference_output,
                               rtol=1e-2,
                               atol=1e-2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton_vs_reference(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    """Test linear decode forward pass against reference implementation"""
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = torch.randn(batch_size,
                            num_heads,
                            head_size,
                            head_size,
                            dtype=dtype,
                            device="cuda")

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.arange(batch_size, device="cuda")

    # Using Triton implementation
    triton_output = linear_decode_forward_triton(q, k, v, kv_caches,
                                                 slope_rate, slot_idx)

    # Reference implementation
    def reference_linear_decode(q, k, v, kv_caches, slope_rate, slot_idx):
        B, H, _, D = q.shape
        output = torch.zeros(B, H * D, dtype=q.dtype, device=q.device)

        for b in range(B):
            slot_id = slot_idx[b].item()
            if slot_id == -1:  # Skip padding positions
                continue

            for h in range(H):
                decay = torch.exp(torch.tensor(-slope_rate[h].item(), 
                                              device=q.device, 
                                              dtype=torch.float32))

                # Get current query, key and value
                q_bh = q[b, h, 0].float()
                k_bh = k[b, h, 0].float()
                v_bh = v[b, h, 0].float()

                # Get cache
                kv_cache_old = kv_caches[b, h].float()

                # Compute new key-value outer product
                kv_outer = torch.outer(k_bh, v_bh)

                # Apply decay and update cache
                kv_new = kv_outer + decay * kv_cache_old

                # Compute output
                out_h = torch.matmul(q_bh, kv_new)

                # Update output and cache
                output[b, h * D:(h + 1) * D] = out_h.to(output.dtype)
                kv_caches[b, h] = kv_new.to(kv_caches.dtype)

        return output

    reference_output = reference_linear_decode(q, k, v, kv_caches.clone(),
                                               slope_rate, slot_idx)

    # Compare results from both implementations
    torch.testing.assert_close(triton_output,
                               reference_output,
                               rtol=1e-2,
                               atol=1e-2)
