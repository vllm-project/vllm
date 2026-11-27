# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import linear_decode_forward_triton
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Lightning attention Triton kernels require CUDA/ROCm or XPU.",
)

NUM_HEADS = [4, 8]
HEAD_SIZES = [64]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16]
DTYPES = [torch.float32]


def reference_lightning_attention(q, k, v, ed, block_size, kv_history):
    """Reference implementation of lightning attention core algorithm.

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
    decay = torch.exp(-ed).view(1, -1, 1, 1) if ed.dim() == 1 else torch.exp(-ed)

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
    final_kv_cache = torch.cat([kv_reshaped, kv_reshaped], dim=2)  # [B, H, 2, D, E]

    return output, final_kv_cache


def reference_linear_decode(q, k, v, kv_caches, slope_rate, slot_idx):
    """Reference implementation: linear attention decode function."""
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
            output[b, h * D : (h + 1) * D] = out_h
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
    torch.set_default_device(DEVICE)
    set_random_seed(42)
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = base * torch.randn(
        batch_size, num_heads, head_size, head_size, dtype=dtype, device=DEVICE
    )

    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.zeros(num_heads, device=DEVICE)
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)

    slot_idx = torch.arange(batch_size, device=DEVICE)

    triton_output = linear_decode_forward_triton(
        q, k, v, kv_caches, slope_rate, slot_idx
    )

    reference_output = reference_linear_decode(
        q, k, v, kv_caches_copy, slope_rate, slot_idx
    )
    torch.testing.assert_close(triton_output, reference_output, rtol=1e-1, atol=1e-1)
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
    torch.set_default_device(DEVICE)
    set_random_seed(42)

    batch_size = 4
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = base * torch.randn(
        batch_size, num_heads, head_size, head_size, dtype=dtype, device=DEVICE
    )

    kv_caches_copy = kv_caches.clone()

    slope_rate = torch.zeros(num_heads, device=DEVICE)
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)

    slot_idx = torch.tensor([0, 1, -1, 2], device=DEVICE)

    triton_output = linear_decode_forward_triton(
        q, k, v, kv_caches, slope_rate, slot_idx
    )

    reference_output = reference_linear_decode(
        q, k, v, kv_caches_copy, slope_rate, slot_idx
    )

    padding_mask = (slot_idx != -1).unsqueeze(1).expand(-1, num_heads * head_size)

    triton_masked = triton_output[padding_mask]
    reference_masked = reference_output[padding_mask]

    atol, rtol = 1.5e-1, 1.5e-1

    valid_indices = slot_idx != -1

    for i in range(batch_size):
        if valid_indices[i] > 0:
            torch.testing.assert_close(
                kv_caches[i], kv_caches_copy[i], rtol=rtol, atol=atol
            )

    torch.testing.assert_close(triton_masked, reference_masked, rtol=rtol, atol=atol)

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
    torch.set_default_device(DEVICE)
    set_random_seed(42)

    base = 0.01
    q = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)

    ed = torch.zeros(num_heads, device=DEVICE)
    for h in range(num_heads):
        ed[h] = 0.1 * (h + 1)

    kv_history = base * torch.randn(
        batch_size, num_heads, head_size, head_size, dtype=dtype, device=DEVICE
    )

    kv_history_clone = kv_history.clone()

    ref_output, ref_kv_cache = reference_lightning_attention(
        q, k, v, ed, 256, kv_history
    )

    from vllm.model_executor.layers.lightning_attn import lightning_attention

    actual_output, actual_kv_cache = lightning_attention(
        q, k, v, ed, 256, kv_history_clone
    )

    atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(ref_output, actual_output, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_kv_cache, actual_kv_cache, rtol=rtol, atol=atol)

    assert ref_output.shape == (batch_size, num_heads, seq_len, head_size)
    assert ref_kv_cache.shape == actual_kv_cache.shape


@torch.inference_mode()
def test_bailing_prefill_exports_state_at_checkpoint():
    """A checkpoint holds the state of the prefill stopped at its offset.

    One offset is on the kernel's 256-token block grid and one is not. A decode
    leads the batch, and nothing else in the cache changes, including the null
    block of a row whose checkpoint block was not reserved.
    """
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointMetadata,
    )
    from vllm.model_executor.layers.mamba.linear.bailing_linear_attn import (
        linear_attention_prefill_and_mix,
    )
    from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
        MiniMaxText01LinearKernel,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    torch.set_default_device(DEVICE)
    set_random_seed(0)
    heads, dim = 4, 64
    slope = 0.1 * torch.arange(1, heads + 1, dtype=torch.float32)
    initial = 0.01 * torch.randn(8, heads, dim, dim)
    kv_cache = initial.clone()

    def run(qkv, num_decodes, prefill_lens, slots, checkpoint=None):
        kv_cache.copy_(initial)
        lens = [1] * num_decodes + prefill_lens
        metadata = SimpleNamespace(
            num_decodes=num_decodes,
            num_prefills=len(prefill_lens),
            num_decode_tokens=num_decodes,
            query_start_loc=torch.tensor([0, *lens]).cumsum(0),
            state_indices_tensor_p=torch.tensor(slots[num_decodes:]),
            checkpoint=checkpoint,
        )
        q, k, v = qkv
        out = linear_attention_prefill_and_mix(
            q,
            k,
            v,
            kv_cache,
            torch.tensor(slots),
            metadata,
            slope,
            256,
            decode_fn=lambda q, *_: q.new_zeros(num_decodes, heads * dim),
            prefix_fn=MiniMaxText01LinearKernel.jit_linear_forward_prefix,
        )
        return out[num_decodes:], kv_cache.clone()

    prefill_lens, offsets = [600, 300, 400], [512, 100, 256]
    slots, ckpt_slots = [1, 2, 3, 4], [5, 6, NULL_BLOCK_ID]
    qkv = [0.1 * torch.randn(1 + sum(prefill_lens), heads, dim) for _ in range(3)]
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32),
        state_indices=torch.tensor(ckpt_slots, dtype=torch.int32),
        offsets=offsets,
    )
    out, without_checkpoint = run(qkv, 1, prefill_lens, slots)
    out_ckpt, with_checkpoint = run(qkv, 1, prefill_lens, slots, checkpoint)

    starts = [1 + sum(prefill_lens[:i]) for i in range(len(prefill_lens))]
    for start, offset, slot, ckpt_slot in zip(starts, offsets, slots[1:], ckpt_slots):
        if ckpt_slot == NULL_BLOCK_ID:
            continue
        part = [x[start : start + offset] for x in qkv]
        _, stopped = run(part, 0, [offset], [slot])
        torch.testing.assert_close(
            with_checkpoint[ckpt_slot], stopped[slot], rtol=0, atol=0
        )
        with_checkpoint[ckpt_slot] = without_checkpoint[ckpt_slot]
    torch.testing.assert_close(with_checkpoint, without_checkpoint)

    # Splitting changes the kernel's tiling, so the outputs differ by its
    # rounding; the split must be no less accurate than the single pass.
    ref = []
    for start, n, slot in zip(starts, prefill_lens, slots[1:]):
        q, k, v = (x[start : start + n].transpose(0, 1)[None].double() for x in qkv)
        out_ref, _ = reference_lightning_attention(
            q, k, v, slope.double(), 256, initial[slot][None].double()
        )
        ref.append(out_ref[0].transpose(0, 1).reshape(n, -1))
    error = lambda o: (o.double() - torch.cat(ref)).abs().max()
    assert error(out_ckpt) <= 2 * error(out)
