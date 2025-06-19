# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import (
    linear_decode_forward_triton)
from vllm.platforms import current_platform

NUM_HEADS = [4, 8]
HEAD_SIZES = [64]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16]
DTYPES = [torch.float32]


def reference_lightning_attention(q, k, v, ed, block_size, kv_history):
    """Reference implementation of lightning attention core algorithm
    
    The difference from the main implementation is that this processes 
    each step sequentially, instead of using parallelized triton kernels
    """
    B, H, S, D = q.shape
    E = v.shape[-1]
    dtype = q.dtype
    output = torch.zeros((B, H, S, E), dtype=dtype, device=q.device)

    # Use clone() to ensure an independent copy
    if kv_history is None:
        kv_cache = torch.zeros((B, H, D, E), dtype=dtype, device=q.device)
    else:
        kv_cache = kv_history.clone()

    # More efficient implementation
    # Convert decay factors to matrix form
    if ed.dim() == 1:
        decay = torch.exp(-ed).view(1, -1, 1, 1)
    else:
        decay = torch.exp(-ed)

    for b in range(B):
        for step in range(S):
            # Process all heads at once for this position
            q_bs = q[b, :, step]  # [H, D]
            k_bs = k[b, :, step]  # [H, D]
            v_bs = v[b, :, step]  # [H, E]

            # Calculate KV outer products for all heads
            for h in range(H):
                # Calculate KV outer product
                kv_outer = torch.outer(k_bs[h], v_bs[h])

                # Update KV cache with decay
                # Note: Using the same order as in the Triton kernel
                kv_cache[b, h] = decay[0, h, 0, 0] * kv_cache[b, h] + kv_outer

                # Calculate attention output
                output[b, h, step] = torch.matmul(q_bs[h], kv_cache[b, h])

    # Match the shape returned by the actual implementation
    # The actual implementation returns a tensor of shape [B, H, 2, D, E]
    # where dimension 2 contains both KV and KV history
    kv_reshaped = kv_cache.unsqueeze(2)  # [B, H, 1, D, E]
    final_kv_cache = torch.cat([kv_reshaped, kv_reshaped],
                               dim=2)  # [B, H, 2, D, E]

    return output, final_kv_cache


def reference_linear_decode(q, k, v, kv_caches, slope_rate, slot_idx):
    """Reference implementation: linear attention decode function"""
    B, H, _, D = q.shape
    output = torch.zeros(B, H * D, dtype=q.dtype, device=q.device)

    # Calculate decay factors once (more efficient)
    decay = torch.exp(-slope_rate).view(-1, 1, 1)  # [H, 1, 1]

    # Process each batch
    for b in range(B):
        slot_id = slot_idx[b].item()

        # Skip padding positions
        if slot_id == -1:
            continue

        # Process all heads at once for this batch
        q_b = q[b, :, 0]  # [H, D]
        k_b = k[b, :, 0]  # [H, D]
        v_b = v[b, :, 0]  # [H, D]

        # Process each attention head
        for h in range(H):
            # Get current query, key and value
            q_bh = q_b[h]
            k_bh = k_b[h]
            v_bh = v_b[h]

            # Get cache
            kv_cache_old = kv_caches[b, h]

            # Calculate new key-value outer product
            kv_outer = torch.outer(k_bh, v_bh)

            # Apply decay and update cache
            kv_new = kv_outer + decay[h, 0, 0] * kv_cache_old

            # Calculate output
            out_h = torch.matmul(q_bh, kv_new)

            # Update output and cache
            output[b, h * D:(h + 1) * D] = out_h
            kv_caches[b, h] = kv_new

    return output


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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = base * torch.randn(batch_size,
                                   num_heads,
                                   head_size,
                                   head_size,
                                   dtype=dtype,
                                   device="cuda")

    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.zeros(num_heads, device="cuda")
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)

    slot_idx = torch.arange(batch_size, device="cuda")

    triton_output = linear_decode_forward_triton(q, k, v, kv_caches,
                                                 slope_rate, slot_idx)

    reference_output = reference_linear_decode(q, k, v, kv_caches_copy,
                                               slope_rate, slot_idx)
    torch.testing.assert_close(triton_output,
                               reference_output,
                               rtol=1e-1,
                               atol=1e-1)
    torch.testing.assert_close(kv_caches, kv_caches_copy, rtol=1e-1, atol=1e-1)

    assert triton_output.shape == (batch_size, num_heads * head_size)


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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)

    batch_size = 4
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = base * torch.randn(batch_size,
                                   num_heads,
                                   head_size,
                                   head_size,
                                   dtype=dtype,
                                   device="cuda")

    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.zeros(num_heads, device="cuda")
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)

    slot_idx = torch.tensor([0, 1, -1, 2], device="cuda")

    triton_output = linear_decode_forward_triton(q, k, v, kv_caches,
                                                 slope_rate, slot_idx)

    reference_output = reference_linear_decode(q, k, v, kv_caches_copy,
                                               slope_rate, slot_idx)

    padding_mask = (slot_idx
                    != -1).unsqueeze(1).expand(-1, num_heads * head_size)

    triton_masked = triton_output[padding_mask]
    reference_masked = reference_output[padding_mask]

    atol, rtol = 1.5e-1, 1.5e-1

    valid_indices = slot_idx != -1

    for i in range(batch_size):
        if valid_indices[i] > 0:
            torch.testing.assert_close(kv_caches[i],
                                       kv_caches_copy[i],
                                       rtol=rtol,
                                       atol=atol)

    torch.testing.assert_close(triton_masked,
                               reference_masked,
                               rtol=rtol,
                               atol=atol)

    assert triton_output.shape == (batch_size, num_heads * head_size)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention_reference(
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)

    base = 0.01
    q = base * torch.randn(
        batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = base * torch.randn(
        batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = base * torch.randn(
        batch_size, num_heads, seq_len, head_size, dtype=dtype)

    ed = torch.zeros(num_heads, device="cuda")
    for h in range(num_heads):
        ed[h] = 0.1 * (h + 1)

    kv_history = base * torch.randn(batch_size,
                                    num_heads,
                                    head_size,
                                    head_size,
                                    dtype=dtype,
                                    device="cuda")

    kv_history_clone = kv_history.clone()

    ref_output, ref_kv_cache = reference_lightning_attention(
        q, k, v, ed, 256, kv_history)

    from vllm.model_executor.layers.lightning_attn import lightning_attention
    actual_output, actual_kv_cache = lightning_attention(
        q, k, v, ed, 256, kv_history_clone)

    atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(ref_output, actual_output, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_kv_cache,
                               actual_kv_cache,
                               rtol=rtol,
                               atol=atol)

    assert ref_output.shape == (batch_size, num_heads, seq_len, head_size)
    assert ref_kv_cache.shape == actual_kv_cache.shape
