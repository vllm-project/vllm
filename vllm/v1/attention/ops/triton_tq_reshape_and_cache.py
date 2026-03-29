# SPDX-License-Identifier: Apache-2.0
"""Triton kernel for TurboQuant quantize-on-store into KV cache.

Per token, per KV head:
  1. Normalize: x_hat = x / ||x||
  2. Rotate: y = Pi @ x_hat  (GEMV 128x128)
  3. Scalar quantize: idx[j] = nearest(y[j], centroids)
  4. Reconstruct: y_hat = centroids[idx], x_mse = Pi^T @ y_hat (GEMV)
  5. Residual: r = x_hat - x_mse, gamma = ||r||
  6. QJL: signs = sign(S @ r) (GEMV), pack into bits
  7. Pack [mse_indices | qjl_signs | norm | gamma] -> cache slot
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tq_quantize_store_kernel(
    # Input K/V
    key_ptr,           # [num_tokens, num_heads, head_size] float16/bf16
    value_ptr,         # [num_tokens, num_heads, head_size] float16/bf16
    # Cache output
    key_cache_ptr,     # [num_blocks, block_size, num_heads, packed_size] uint8
    val_cache_ptr,     # [num_blocks, block_size, num_heads, packed_size] uint8
    # Slot mapping
    slot_mapping_ptr,  # [num_tokens] int64
    # TQ params
    Pi_ptr,            # [head_size, head_size] float32
    S_ptr,             # [head_size, head_size] float32
    centroids_ptr,     # [n_centroids] float32
    # Dims
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    packed_size: tl.constexpr,
    mse_bits: tl.constexpr,
    n_centroids: tl.constexpr,
    mse_bytes: tl.constexpr,
    qjl_bytes: tl.constexpr,
):
    """One program per (token, head)."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Load slot
    slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot < 0:
        return

    block_idx = slot // block_size
    block_off = slot % block_size

    # Offsets for loading key vector
    d_offs = tl.arange(0, head_size)
    key_base = token_idx * num_heads * head_size + head_idx * head_size
    k_vec = tl.load(key_ptr + key_base + d_offs).to(tl.float32)

    # === 1. Normalize ===
    vec_norm_sq = tl.sum(k_vec * k_vec)
    vec_norm = tl.sqrt(vec_norm_sq + 1e-12)
    x_hat = k_vec / vec_norm

    # === 2. Rotate: y = Pi @ x_hat ===
    y = tl.zeros([head_size], dtype=tl.float32)
    for i in tl.static_range(head_size):
        Pi_row = tl.load(Pi_ptr + i * head_size + d_offs)
        y_i = tl.sum(Pi_row * x_hat)
        y += tl.where(d_offs == i, y_i, 0.0)

    # === 3. Scalar quantize ===
    c_offs = tl.arange(0, n_centroids)
    centroids = tl.load(centroids_ptr + c_offs)

    idx = tl.zeros([head_size], dtype=tl.int32)
    y_hat = tl.zeros([head_size], dtype=tl.float32)

    for j in tl.static_range(head_size):
        y_j = tl.sum(tl.where(d_offs == j, y, 0.0))
        dists = tl.abs(y_j - centroids)
        best_k = tl.argmin(dists, axis=0)
        best_c = tl.load(centroids_ptr + best_k)
        idx += tl.where(d_offs == j, best_k, 0).to(tl.int32)
        y_hat += tl.where(d_offs == j, best_c, 0.0)

    # === 4. Unrotate: x_mse = Pi^T @ y_hat ===
    x_mse = tl.zeros([head_size], dtype=tl.float32)
    for i in tl.static_range(head_size):
        Pi_col = tl.load(Pi_ptr + d_offs * head_size + i)
        x_mse_i = tl.sum(Pi_col * y_hat)
        x_mse += tl.where(d_offs == i, x_mse_i, 0.0)

    # === 5. Residual ===
    r = x_hat - x_mse
    gamma_sq = tl.sum(r * r)
    gamma = tl.sqrt(gamma_sq + 1e-12)

    # === 6. QJL: signs = sign(S @ r) ===
    signs = tl.zeros([head_size], dtype=tl.int32)
    for i in tl.static_range(head_size):
        S_row = tl.load(S_ptr + i * head_size + d_offs)
        proj_i = tl.sum(S_row * r)
        sign_i = tl.where(proj_i >= 0, 1, 0)
        signs += tl.where(d_offs == i, sign_i, 0)

    # === 7. Pack into cache ===
    cache_base = (
        block_idx * block_size * num_heads * packed_size
        + block_off * num_heads * packed_size
        + head_idx * packed_size
    )

    # Pack MSE indices (2-bit: 4 per byte)
    if mse_bits == 2:
        for b in tl.static_range(mse_bytes):
            byte_val: tl.int32 = 0
            for k in tl.static_range(4):
                j = b * 4 + k
                if j < head_size:
                    idx_j = tl.sum(tl.where(d_offs == j, idx, 0))
                    byte_val |= (idx_j & 0x3) << (k * 2)
            tl.store(key_cache_ptr + cache_base + b, byte_val.to(tl.uint8))

    # Pack QJL signs (8 per byte)
    for b in tl.static_range(qjl_bytes):
        byte_val: tl.int32 = 0
        for k in tl.static_range(8):
            j = b * 8 + k
            if j < head_size:
                s_j = tl.sum(tl.where(d_offs == j, signs, 0))
                byte_val |= (s_j & 0x1) << k
        tl.store(key_cache_ptr + cache_base + mse_bytes + b, byte_val.to(tl.uint8))

    # Store norms as float16 bytes
    norm_offset = cache_base + mse_bytes + qjl_bytes
    norm_f16 = vec_norm.to(tl.float16)
    gamma_f16 = gamma.to(tl.float16)
    tl.store(key_cache_ptr + norm_offset, norm_f16.to(tl.uint8, bitcast=True),
             mask=False)
    tl.store(key_cache_ptr + norm_offset + 2, gamma_f16.to(tl.uint8, bitcast=True),
             mask=False)

    # === Store value (FP16 raw bytes, truncated to packed_size) ===
    val_base = token_idx * num_heads * head_size + head_idx * head_size
    v_vec = tl.load(value_ptr + val_base + d_offs)
    val_cache_base = (
        block_idx * block_size * num_heads * packed_size
        + block_off * num_heads * packed_size
        + head_idx * packed_size
    )
    n_val_elems = packed_size // 2
    val_offs = tl.arange(0, head_size)
    for e in tl.static_range(n_val_elems):
        if e < head_size:
            v_e = tl.sum(tl.where(val_offs == e, v_vec, 0.0)).to(tl.float16)
            v_u16 = v_e.to(tl.uint16, bitcast=True)
            tl.store(val_cache_ptr + val_cache_base + e * 2,
                     (v_u16 & 0xFF).to(tl.uint8))
            tl.store(val_cache_ptr + val_cache_base + e * 2 + 1,
                     ((v_u16 >> 8) & 0xFF).to(tl.uint8))


def triton_tq_reshape_and_cache(
    key: torch.Tensor,       # [num_tokens, num_heads, head_size]
    value: torch.Tensor,     # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor, # [num_blocks, block_size, num_heads, packed_size]
    val_cache: torch.Tensor, # [num_blocks, block_size, num_heads, packed_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    Pi: torch.Tensor,        # [head_size, head_size]
    S: torch.Tensor,         # [head_size, head_size]
    centroids: torch.Tensor, # [n_centroids]
    mse_bits: int,
):
    """Launch TurboQuant reshape+cache Triton kernel."""
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    packed_size = key_cache.shape[3]
    n_centroids = centroids.shape[0]
    mse_bytes = math.ceil(head_size * mse_bits / 8)
    qjl_bytes = math.ceil(head_size / 8)

    grid = (num_tokens, num_heads)

    _tq_quantize_store_kernel[grid](
        key, value,
        key_cache, val_cache,
        slot_mapping,
        Pi.contiguous(), S.contiguous(), centroids.contiguous(),
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        packed_size=packed_size,
        mse_bits=mse_bits,
        n_centroids=n_centroids,
        mse_bytes=mse_bytes,
        qjl_bytes=qjl_bytes,
    )
