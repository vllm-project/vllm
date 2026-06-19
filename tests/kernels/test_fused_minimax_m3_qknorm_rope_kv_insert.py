# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the horizontally-fused MiniMax-M3 attention pre-processing
kernel:

  fused_minimax_m3_qknorm_rope_kv_insert
    - q / k / index_q / index_k: Gemma RMSNorm + partial NeoX RoPE (in place)
    - sparse (insert) mode: scatter k/v into the paged bf16 KV cache and the
      index key into the index cache by its own slot mapping.

Reference: PyTorch Gemma RMSNorm with the same dtype materialization boundary
as the unfused path, followed by vLLM CUDA rotary_embedding-style NeoX RoPE.
"""

import pytest
import torch

import vllm._custom_ops as ops

HEAD_DIM = 128
ROTARY_DIM = 64


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused MiniMax-M3 op not built in",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rotary_dim]
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    """x: [..., 128]; weight: [128]. Returns original dtype."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    """NeoX-style RoPE on the leading rotary_dim dims; rest pass through.

    x: [num_tokens, num_heads, head_dim]
    cos_sin_cache: [max_pos, rotary_dim] (cos||sin), read as float (matches the
    kernel, which loads the bf16 cache and converts to fp32).
    """
    half = rotary_dim // 2
    cs = cos_sin_cache[positions].float()  # [num_tokens, rotary_dim]
    cos = cs[..., :half].unsqueeze(1)  # [nt, 1, half]
    sin = cs[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim].float()
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = x.clone()
    out[..., :half] = o1
    out[..., half:rotary_dim] = o2
    return out.to(x.dtype)


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps):
    """[nt, nheads, 128] -> Gemma norm + neox partial rope."""
    normed = gemma_rmsnorm(x, weight, eps)
    roped = apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM)
    return roped


# ── Test 1: dense mode (norm+rope only, no index, no insert) ─────────────────


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_dense_norm_rope(num_tokens, num_heads, num_kv_heads):
    torch.manual_seed(0)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        kv_cache_dtype="auto",
    )
    q_out, k_out, v_out = qkv.split([qsz, kvsz, kvsz], dim=-1)

    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    # V is untouched.
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)


# ── Test 2: sparse mode (full: index branch + cache inserts) ─────────────────


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_full(num_tokens, block_size, kv_cache_dtype):
    torch.manual_seed(1)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM
    # Single fused tensor packing [q | k | v | index_q | index_k].
    qkv = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device
    )
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_storage_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        HEAD_DIM,
        dtype=kv_cache_storage_dtype,
        device=device,
    )
    index_cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    # Contiguous gather targets: the kernel writes the normed/roped q and
    # index_q here (de-interleaved from the packed qkv); k/v/index_k stay in
    # place inside qkv and are scatter-inserted into the caches.
    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
    index_q = torch.empty(num_tokens, iqsz, dtype=dtype, device=device)

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        iq_w,
        ik_w,
        num_idx_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q,
        kv_cache_dtype,
    )

    # ── norm+rope parity. q/index_q land in their gather buffers; k/index_k are
    # rewritten in place inside qkv. ──
    _, k_out, v_out, _, index_k = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, iq_orig, ik_orig = qkv_orig.split(splits, dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, kvsz)
    iq_ref = norm_rope_ref(
        iq_orig.view(num_tokens, num_idx_heads, HEAD_DIM),
        iq_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, num_idx_heads * HEAD_DIM)
    ik_ref = norm_rope_ref(
        ik_orig.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps
    ).view(num_tokens, HEAD_DIM)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k, ik_ref, rtol=1e-2, atol=1e-2)

    # ── Cache inserts. ──
    # Main cache layout is [num_blocks, 2, block_size, num_kv_heads, head_dim]
    # (the K/V axis sits *before* block_size); index cache is [nb, bs, head_dim].
    k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
    v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)  # v is raw (no norm/rope)
    if kv_cache_dtype == "fp8":
        expected_kv_cache = torch.zeros_like(kv_cache)
        scale = torch.ones((), device=device)
        ops.reshape_and_cache_flash(
            k_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            v_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            expected_kv_cache[:, 0],
            expected_kv_cache[:, 1],
            slot_mapping,
            kv_cache_dtype,
            scale,
            scale,
        )
        torch.testing.assert_close(kv_cache, expected_kv_cache, rtol=0, atol=0)
    else:
        for t in range(num_tokens):
            s = slot_mapping[t].item()
            b, pos = s // block_size, s % block_size
            torch.testing.assert_close(
                kv_cache[b, 0, pos], k_ref_h[t], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(kv_cache[b, 1, pos], v_ref_h[t], rtol=0, atol=0)

    expected_index_cache = torch.zeros_like(index_cache).view(-1, HEAD_DIM)
    expected_index_cache[index_slot_mapping] = index_k
    torch.testing.assert_close(
        index_cache.view(-1, HEAD_DIM), expected_index_cache, rtol=0, atol=0
    )
