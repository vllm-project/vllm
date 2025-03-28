# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import (
    linear_decode_forward_triton)
from vllm.platforms import current_platform

NUM_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16, 128]
DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def reference_lightning_attention(q, k, v, ed, block_size, kv_history):
    """Rreference implementation: using sequential linear decoding"""
    B, H, S, D = q.shape
    output = torch.zeros_like(q)
    kv_cache = kv_history.clone() if kv_history is not None else \
        torch.zeros((B, H, D, D), dtype=torch.float32, device=q.device)

    for step in range(S):
        q_step = q[:, :, step:step + 1]
        k_step = k[:, :, step:step + 1]
        v_step = v[:, :, step:step + 1]

        q_linear = q_step.permute(0, 1, 3, 2)
        k_linear = k_step.permute(0, 1, 3, 2)
        v_linear = v_step.permute(0, 1, 3, 2)

        output_step = linear_decode_forward_triton(
            q_linear, k_linear, v_linear, kv_cache, ed,
            torch.arange(B, device=q.device))

        output_step = output_step.view(B, H, D).permute(0, 1, 3, 2)
        output[:, :, step] = output_step.squeeze(2)

    return output, kv_cache


def reference_linear_decode(q, k, v, kv_caches, slope_rate, slot_idx):
    """Reference implementation: linear attention decode function
    
    Args:
        q: Query tensor with shape [B, H, 1, D]
        k: Key tensor with shape [B, H, 1, D]
        v: Value tensor with shape [B, H, 1, D]
        kv_caches: KV cache tensors
        slope_rate: Decay rate tensor
        slot_idx: Slot indices for the batch
        
    Returns:
        output: Attention output tensor
    """
    B, H, _, D = q.shape
    # Initialize output with the correct shape directly
    output = torch.zeros(B, H * D, dtype=q.dtype, device=q.device)

    # Process each batch
    for b in range(B):
        slot_id = slot_idx[b].item()

        # Skip padding positions
        if slot_id == -1:
            continue

        # Process each attention head
        for h in range(H):
            # Get decay rate
            decay = torch.exp(
                torch.tensor(-slope_rate[h].item(),
                             device=q.device,
                             dtype=torch.float32))

            # Get current query, key and value
            q_bh = q[b, h, 0].float()
            k_bh = k[b, h, 0].float()
            v_bh = v[b, h, 0].float()

            # Get cache
            kv_cache_old = kv_caches[b, h].float()

            # Calculate new key-value outer product
            kv_outer = torch.outer(k_bh, v_bh)

            # Apply decay and update cache
            kv_new = kv_outer + decay * kv_cache_old

            # Calculate output
            out_h = torch.matmul(q_bh, kv_new)

            # Update output and cache
            output[b, h * D:(h + 1) * D] = out_h.to(output.dtype)
            kv_caches[b, h] = kv_new.to(kv_caches.dtype)

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

    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.arange(batch_size, device="cuda")

    # Triton implementation
    triton_output = linear_decode_forward_triton(q, k, v, kv_caches,
                                                 slope_rate, slot_idx)

    # Reference implementation
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
    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.tensor([0, 1, -1, 2], device="cuda")

    output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate,
                                          slot_idx)

    assert output.shape == (batch_size, num_heads * head_size)

    # Compare implementation results
    triton_output = linear_decode_forward_triton(q, k, v, kv_caches,
                                                 slope_rate, slot_idx)
    reference_output = reference_linear_decode(q, k, v, kv_caches_copy,
                                               slope_rate, slot_idx)

    torch.testing.assert_close(triton_output,
                               reference_output,
                               rtol=1e-1,
                               atol=1e-1)
    torch.testing.assert_close(kv_caches, kv_caches_copy, rtol=1e-1, atol=1e-1)
