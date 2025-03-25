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

    # Create kv_caches's copy to ensure both
    # implementations use the same initial values
    kv_caches_copy = kv_caches.clone()

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

    reference_output = reference_linear_decode(q, k, v, kv_caches_copy,
                                               slope_rate, slot_idx)

    # Increase tolerance to handle floating point precision differences
    torch.testing.assert_close(triton_output,
                               reference_output,
                               rtol=1e-1,
                               atol=1e-1)
