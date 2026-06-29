# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch OSCAR INT2 KV-cache quantization / dequantization ops.

Provides the core pack/unpack, quantize/dequantize, and cache store/load
routines used by the OSCAR attention backend. All operations are expressed
in pure PyTorch (no Triton), so they work with any head_dim divisible by 4.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Hadamard rotation matrix (Sylvester construction, cached)          #
# ------------------------------------------------------------------ #


def build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Hadamard matrix, cached per (d, device_str)."""
    return _build_hadamard_cached(d, str(device))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    """Sylvester-construct a D×D orthonormal Hadamard.

    Only works for power-of-2 dimensions. For non-power-of-2 head_dim,
    callers should use ``build_rotation_matrix`` which handles padding.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1),
                        torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


def build_rotation_matrix(
    d: int, device: torch.device
) -> torch.Tensor:
    """Build an orthonormal rotation matrix for dimension ``d``.

    * Power-of-2 ``d``: uses the Hadamard (Sylvester) construction.
    * Other ``d``: uses a random orthogonal matrix (``torch.linalg.qr``
      on a random Gaussian), seeded deterministically so all layers and
      ranks share the same rotation.
    """
    if d > 0 and (d & (d - 1)) == 0:
        return build_hadamard(d, device)
    # Deterministic random orthogonal for non-power-of-2 dims.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    A = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)


# ------------------------------------------------------------------ #
#  INT2 pack / unpack                                                 #
# ------------------------------------------------------------------ #


def pack_int2(vals: torch.Tensor) -> torch.Tensor:
    """Pack INT2 values (0-3) into uint8, 4 values per byte.

    Args:
        vals: (..., D) tensor with values in {0, 1, 2, 3}.

    Returns:
        (..., D // 4) uint8 tensor.
    """
    assert vals.shape[-1] % 4 == 0
    v = vals.to(torch.uint8)
    shape = v.shape[:-1]
    D = v.shape[-1]
    v = v.view(*shape, D // 4, 4)
    packed = v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)
    return packed


def unpack_int2(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack uint8 → INT2 values (0-3).

    Args:
        packed: (..., D // 4) uint8 tensor.
        D: original dimension.

    Returns:
        (..., D) int32 tensor with values in {0, 1, 2, 3}.
    """
    p = packed.to(torch.int32).unsqueeze(-1)
    q0 = p & 0x3
    q1 = (p >> 2) & 0x3
    q2 = (p >> 4) & 0x3
    q3 = (p >> 6) & 0x3
    # Interleave correctly: q0[0], q1[0], q2[0], q3[0], q0[1]...
    q = torch.stack([q0, q1, q2, q3], dim=-1)
    return q.view(*packed.shape[:-1], D)


# ------------------------------------------------------------------ #
#  Quantize / dequantize a single row                                 #
# ------------------------------------------------------------------ #


def quantize_int2(
    row: torch.Tensor, clip_ratio: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform INT2 quantization of a row vector.

    Args:
        row: (..., D) float tensor.
        clip_ratio: fraction of sorted |values| to clip (0 = no clip).

    Returns:
        (packed, scale, zero) where:
          packed: (..., D//4) uint8
          scale:  (...) float16
          zero:   (...) float16
    """
    row_f32 = row.float()

    # Optional percentile clipping
    if clip_ratio > 0.0:
        D = row_f32.shape[-1]
        clip_idx = min(int(clip_ratio * D), D - 1)
        abs_sorted = row_f32.abs().sort(dim=-1).values
        threshold = abs_sorted[..., clip_idx: clip_idx + 1]
        row_f32 = row_f32.clamp(-threshold, threshold)

    row_min = row_f32.amin(dim=-1)
    row_max = row_f32.amax(dim=-1)
    row_range = (row_max - row_min).clamp(min=1e-8)
    scale = row_range / 3.0
    zero = -row_min / scale

    # Quantize to {0, 1, 2, 3}
    q = ((row_f32 - row_min.unsqueeze(-1)) / scale.unsqueeze(-1) + 0.5)
    q = q.clamp(0, 3).to(torch.uint8)

    packed = pack_int2(q)
    return packed, scale.to(torch.float16), zero.to(torch.float16)


def dequantize_int2(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    D: int,
) -> torch.Tensor:
    """Dequantize INT2 packed values back to float.

    Args:
        packed: (..., D//4) uint8.
        scale:  (...) float16.
        zero:   (...) float16.
        D: original dimension.

    Returns:
        (..., D) float32 tensor.
    """
    q = unpack_int2(packed, D).float()
    s = scale.float().unsqueeze(-1)
    z = zero.float().unsqueeze(-1)
    return (q - z) * s


# ------------------------------------------------------------------ #
#  Cache store / load  (combined K+V slot layout)                     #
# ------------------------------------------------------------------ #


def get_cache_layout(D: int) -> tuple[int, int, int, int, int]:
    """Compute byte offsets for the combined K+V cache slot.

    Returns:
        (k_start, ksz_start, v_start, vsz_start, slot_size)

    Each slot: [k_packed | k_scale_zero | padding | v_packed | v_scale_zero]
    """
    k_q_bytes = D // 4
    sz_bytes = 4  # 2 × fp16
    k_aligned = (k_q_bytes + sz_bytes + 15) // 16 * 16
    k_start = 0
    ksz_start = k_q_bytes
    v_start = k_aligned
    vsz_start = v_start + k_q_bytes
    slot_size = k_aligned * 2
    return k_start, ksz_start, v_start, vsz_start, slot_size


def store_kv_to_cache(
    key_rot: torch.Tensor,     # (N, Hk, D) — already rotated
    value_rot: torch.Tensor,   # (N, Hk, D) — already rotated
    kv_cache: torch.Tensor,    # (num_blocks, block_size, Hk, slot_size)
    slot_mapping: torch.Tensor,  # (N,) int
    clip_ratio_k: float,
    clip_ratio_v: float,
) -> None:
    """Quantize and store rotated K/V into combined OSCAR cache slots."""
    N, Hk, D = key_rot.shape
    k_start, ksz_start, v_start, vsz_start, _ = get_cache_layout(D)
    k_q_bytes = D // 4
    sz_bytes = 4

    nb, bs, _, S = kv_cache.shape
    cache_flat = kv_cache.view(nb * bs, Hk, S)

    # Filter out invalid slots (slot_mapping == -1)
    valid = slot_mapping >= 0
    if not valid.any():
        return

    valid_idx = valid.nonzero(as_tuple=True)[0]
    slots = slot_mapping[valid_idx]
    k_valid = key_rot[valid_idx]    # (M, Hk, D)
    v_valid = value_rot[valid_idx]  # (M, Hk, D)

    # Quantize K
    k_packed, k_scale, k_zero = quantize_int2(k_valid, clip_ratio_k)
    # Quantize V
    v_packed, v_scale, v_zero = quantize_int2(v_valid, clip_ratio_v)

    # Write K packed data
    cache_flat[slots, :, k_start:k_start + k_q_bytes] = k_packed
    # Write K scale/zero as raw uint8 (reinterpreted fp16)
    k_sz = torch.stack([k_scale, k_zero], dim=-1)  # (M, Hk, 2) fp16
    k_sz_u8 = k_sz.view(torch.uint8).view(k_valid.shape[0], Hk, sz_bytes)
    cache_flat[slots, :, ksz_start:ksz_start + sz_bytes] = k_sz_u8

    # Write V packed data
    cache_flat[slots, :, v_start:v_start + k_q_bytes] = v_packed
    # Write V scale/zero
    v_sz = torch.stack([v_scale, v_zero], dim=-1)
    v_sz_u8 = v_sz.view(torch.uint8).view(v_valid.shape[0], Hk, sz_bytes)
    cache_flat[slots, :, vsz_start:vsz_start + sz_bytes] = v_sz_u8


def load_kv_from_cache(
    kv_cache: torch.Tensor,      # (num_blocks, block_size, Hk, slot_size)
    slot_mapping: torch.Tensor,  # (seq_len,) int
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize K/V from OSCAR cache (still in rotated space).

    Returns:
        (k_rot, v_rot) each (seq_len, Hk, D) float32.
    """
    seq_len = slot_mapping.shape[0]
    Hk = kv_cache.shape[2]
    k_start, ksz_start, v_start, vsz_start, _ = get_cache_layout(D)
    k_q_bytes = D // 4
    sz_bytes = 4

    nb, bs = kv_cache.shape[:2]
    cache_flat = kv_cache.view(nb * bs, Hk, kv_cache.shape[-1])

    slots = slot_mapping.long()

    # Read K
    k_packed = cache_flat[slots, :, k_start:k_start + k_q_bytes]
    k_sz_u8 = cache_flat[slots, :, ksz_start:ksz_start + sz_bytes]
    k_sz_f16 = k_sz_u8.view(torch.float16).view(seq_len, Hk, 2)
    k_scale = k_sz_f16[..., 0]
    k_zero = k_sz_f16[..., 1]
    k_rot = dequantize_int2(k_packed, k_scale, k_zero, D)

    # Read V
    v_packed = cache_flat[slots, :, v_start:v_start + k_q_bytes]
    v_sz_u8 = cache_flat[slots, :, vsz_start:vsz_start + sz_bytes]
    v_sz_f16 = v_sz_u8.view(torch.float16).view(seq_len, Hk, 2)
    v_scale = v_sz_f16[..., 0]
    v_zero = v_sz_f16[..., 1]
    v_rot = dequantize_int2(v_packed, v_scale, v_zero, D)

    return k_rot, v_rot


def dequant_and_unrotate(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    R: torch.Tensor,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize and inverse-rotate K/V from the OSCAR cache.

    Args:
        kv_cache: the raw cache tensor.
        slot_mapping: (seq_len,) slot indices.
        R: (D, D) orthonormal rotation matrix (R^T = R^-1).
        D: head dimension.

    Returns:
        (k_unrot, v_unrot) each (seq_len, Hk, D) float16.
    """
    k_rot, v_rot = load_kv_from_cache(kv_cache, slot_mapping, D)
    # Inverse rotation: x_unrot = x_rot @ R^T
    RT = R.T.float()
    k_unrot = (k_rot @ RT).to(torch.float16)
    v_unrot = (v_rot @ RT).to(torch.float16)
    return k_unrot, v_unrot


# ------------------------------------------------------------------ #
#  Decode attention (pure PyTorch, no Triton)                         #
# ------------------------------------------------------------------ #


def oscar_decode_attention(
    query: torch.Tensor,       # (B, Hq, D)
    kv_cache: torch.Tensor,    # (num_blocks, block_size, Hk, slot_size) uint8
    block_table: torch.Tensor, # (B, max_num_blocks) int32
    seq_lens: torch.Tensor,    # (B,) int32
    R: torch.Tensor,           # (D, D) rotation matrix
    scale: float,
) -> torch.Tensor:
    """Pure PyTorch decode attention from quantized OSCAR cache.

    For each batch element:
      1. Map seq positions → cache slots via block_table
      2. Load + dequantize K/V (still in rotated space)
      3. Compute Q_rot @ K_rot^T → softmax → weighted V_rot sum
      4. Un-rotate output: out @ R^T

    Returns: (B, Hq, D) in query dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    device = query.device
    kv_group_size = Hq // Hk

    # Rotate query: q_rot = q @ R
    q_rot = (query.float() @ R.float())  # (B, Hq, D)

    k_start, ksz_start, v_start, vsz_start, _ = get_cache_layout(D)
    k_q_bytes = D // 4
    sz_bytes = 4
    nb, bs = kv_cache.shape[:2]
    cache_flat = kv_cache.view(nb * bs, Hk, kv_cache.shape[-1])

    outputs = torch.empty(B, Hq, D, dtype=query.dtype, device=device)

    for b in range(B):
        slen = int(seq_lens[b].item())
        if slen <= 0:
            outputs[b] = 0
            continue

        # Build slot mapping from block table
        positions = torch.arange(slen, device=device)
        block_idx = positions // block_size
        block_off = positions % block_size
        blocks = block_table[b, block_idx].long()
        slots = blocks * block_size + block_off

        # Load + dequant K/V (rotated space)
        k_rot, v_rot = load_kv_from_cache(kv_cache, slots, D)
        # k_rot, v_rot: (slen, Hk, D)

        # Attention: q_rot @ k_rot^T
        # GQA: expand KV heads
        q_b = q_rot[b]  # (Hq, D)
        if kv_group_size > 1:
            k_exp = k_rot.transpose(0, 1).repeat_interleave(
                kv_group_size, dim=0
            )  # (Hq, slen, D)
            v_exp = v_rot.transpose(0, 1).repeat_interleave(
                kv_group_size, dim=0
            )  # (Hq, slen, D)
        else:
            k_exp = k_rot.transpose(0, 1)  # (Hk, slen, D)
            v_exp = v_rot.transpose(0, 1)

        # (Hq, 1, D) @ (Hq, D, slen) → (Hq, 1, slen)
        scores = torch.bmm(
            q_b.unsqueeze(1), k_exp.transpose(1, 2)
        ).squeeze(1) * scale  # (Hq, slen)

        attn_weights = F.softmax(scores, dim=-1)  # (Hq, slen)

        # (Hq, 1, slen) @ (Hq, slen, D) → (Hq, D)
        out_rot = torch.bmm(
            attn_weights.unsqueeze(1), v_exp
        ).squeeze(1)  # (Hq, D)

        # Un-rotate: out = out_rot @ R^T
        out = (out_rot @ R.T.float()).to(query.dtype)
        outputs[b] = out

    return outputs
