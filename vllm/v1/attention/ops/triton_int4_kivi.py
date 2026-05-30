# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged INT4-KIVI store / dequant Triton kernels for the vLLM backend.

ACTIVE LAYOUT:
  * V  -> **per token** with 16-element head_dim blocks (symmetric INT4 [-7, 7]).
  * K  -> **per channel (KIVI)** over each full 16-token block (one scale per
    channel, computed over the 16 tokens of that channel) — this isolates K's
    per-channel outliers and is the quality-optimal layout.  Partial trailing
    blocks (a block not yet completely filled, e.g. the hot decode tail) fall
    back to **per-token** K for correctness.

KEY INSIGHT — same cache shape, no byte-budget change:
  The K-side scale region is ``head_size//16`` bytes/token.  Over a full block of
  16 tokens that is ``16 * (head_size//16) == head_size`` bytes = exactly one fp8
  scale **per channel**.  So per-channel-K reuses the identical cache tensor: we
  reinterpret a full block's K-scale bytes as ``head_size`` per-channel fp8 scales
  instead of ``16 * (head_size//16)`` per-token scales.  Channel ``c``'s scale is
  laid at K-scale byte ``c`` of the block, i.e. token ``c // (head_size//16)``,
  scale-byte ``c % (head_size//16)``.

  Store/dequant agree on per-channel vs per-token *purely from geometry* (no
  marker byte): for a request of length ``L``, logical block ``b`` is full iff
  ``(b+1)*16 <= L`` → per-channel; the trailing block with ``L % 16 != 0`` is
  partial → per-token.  In vLLM's eager prefill the whole context is stored in
  one call (all complete blocks full = per-channel), and decode grows only the
  trailing block (per-token) — so this geometric rule is self-consistent.

Numerics mirror the validated ``int4_kivi`` reference kernels:
  * code = round_to_even(x / scale).clamp(-7, 7);  deq = code * scale
  * K (per-channel) scale = MSE-optimal clip over the channel's 16 tokens.
  * V (per-token) scale = MSE-optimal clip over the 16 head_dim elements.
  * round-half-to-even via libdevice.rint, IEEE round-to-nearest division.

PAGED CACHE LAYOUT (uint8), one tensor per layer:
    kv_cache[num_blocks, 2, block_size, num_kv_heads, full_dim]
      dim 1: 0 = K side, 1 = V side
      full_dim = head_size//2 (nibble-packed INT4 data) + head_size//16 (scales)
    Within a token row: [ data_bytes | scale_bytes ].
    Scales are stored as float8_e4m3 (1 byte / 16-elem block) — viewed as uint8
    storage in the cache, reinterpreted as fp8 by the kernels.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 16  # elements per quant block (along head_dim)
QMAX = 7
PACK = BLOCK // 2  # 8 packed bytes per 16-elem block
N_MSE = 16  # MSE clip-search grid points (alpha in [0.5, 1.0])

_BLOCK = tl.constexpr(BLOCK)
_QMAX = tl.constexpr(float(QMAX))
_PACK = tl.constexpr(PACK)
_N_MSE = tl.constexpr(N_MSE)


@triton.jit
def _quant_codes(x, scale):
    q = libdevice.rint(libdevice.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -_QMAX), _QMAX)
    return q


@triton.jit
def _mse_scale(x, amax):
    """MSE-optimal clip scale for a 1-D block: grid-search alpha in [0.5,1.0],
    minimise sum((x - q*scale)^2). Mirrors kv_quant._calibrate('mse'). amax is
    the (clamped) block absmax; alpha=1 reproduces absmax so MSE <= absmax."""
    best_err = 1e38
    best_scale = libdevice.div_rn(amax, _QMAX)
    for i in tl.static_range(_N_MSE):
        a = 0.5 + i * (0.5 / (_N_MSE - 1))
        s = libdevice.div_rn(a * amax, _QMAX)
        q = _quant_codes(x, s)
        diff = x - q * s
        err = tl.sum(diff * diff)
        take = err < best_err
        best_err = tl.where(take, err, best_err)
        best_scale = tl.where(take, s, best_scale)
    return best_scale


@triton.jit
def _store_token_kernel(
    src_ptr,  # bf16 [N, H, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    slot_ptr,  # int64 [N]
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    FULL_DIM: tl.constexpr,  # DATA_BYTES + ND
    KV_SIDE: tl.constexpr,  # 0 = K, 1 = V
    BLOCK_SIZE: tl.constexpr,
    s_src_n,
    s_src_h,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
):
    """One program per (token, head, dblock).  Quantize 16 head_dim elements."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_db = tl.program_id(2)

    slot = tl.load(slot_ptr + pid_n)
    if slot < 0:
        return
    blk_idx = slot // BLOCK_SIZE
    tok_in_blk = slot % BLOCK_SIZE

    ch = pid_db * _BLOCK + tl.arange(0, _BLOCK)  # [BLOCK]
    src_off = pid_n * s_src_n + pid_h * s_src_h + ch
    x = tl.load(src_ptr + src_off).to(tl.float32)  # [BLOCK]

    amax = tl.maximum(tl.max(tl.abs(x)), 1e-9)
    scale = _mse_scale(x, amax)  # MSE-optimal clip (>= ~5% better than absmax)
    # Round the scale to fp8_e4m3 BEFORE quantizing codes so that the codes
    # are computed against the exact scale that dequant will read back. This
    # keeps store and dequant bit-consistent (no double-rounding mismatch).
    scale_byte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    codes = _quant_codes(x, scale)  # [BLOCK] in [-7, 7]

    # pack: elem 2j -> low nibble of byte j, 2j+1 -> high nibble
    ci = codes.to(tl.int32) & 0xF  # [BLOCK]
    nib2 = tl.reshape(ci, (_PACK, 2))
    lo, hi = tl.split(nib2)  # each [PACK]
    packed = (lo | (hi << 4)).to(tl.uint8)  # [PACK]

    # base byte offset of this token row in the cache
    row_base = (
        blk_idx * s_cache_blk
        + KV_SIDE * s_cache_side
        + tok_in_blk * s_cache_tok
        + pid_h * s_cache_h
    )
    # write data nibbles for this dblock
    data_off = row_base + pid_db * _PACK + tl.arange(0, _PACK)
    tl.store(cache_ptr + data_off, packed)

    # write scale (e4m3, 1 byte) into scale region
    s_off = row_base + DATA_BYTES + pid_db
    tl.store(cache_ptr + s_off, scale_byte)


@triton.jit
def _mse_scale_axis0(x, amax):
    """Per-channel MSE-optimal clip scale.  ``x`` is [16 tokens, D] fp32, ``amax``
    is [D] (per-channel absmax over the token axis).  Grid-searches alpha in
    [0.5,1.0] minimising sum over the 16 tokens of (x - q*scale)^2 per channel.
    Returns [D] fp32.  Mirrors ``_mse_scale`` but reduces along axis 0."""
    best_err = tl.full(amax.shape, 1e38, tl.float32)
    best_scale = libdevice.div_rn(amax, _QMAX)
    for i in tl.static_range(_N_MSE):
        a = 0.5 + i * (0.5 / (_N_MSE - 1))
        s = libdevice.div_rn(a * amax, _QMAX)  # [D]
        q = _quant_codes(x, s[None, :])  # [16, D]
        diff = x - q * s[None, :]
        err = tl.sum(diff * diff, axis=0)  # [D]
        take = err < best_err
        best_err = tl.where(take, err, best_err)
        best_scale = tl.where(take, s, best_scale)
    return best_scale


@triton.jit
def _store_k_channel_kernel(
    src_ptr,  # bf16 [N, H, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    blk_phys_ptr,  # int64 [NBLK]  physical block idx per full block
    blk_row0_ptr,  # int64 [NBLK]  index into src of token 0 of this full block
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    BLOCK_SIZE: tl.constexpr,
    s_src_n,
    s_src_h,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
):
    """One program per (full_block, head).  K per-channel over 16 tokens.

    KV_SIDE is always 0 (K).  Loads K[16 tokens, D channels] for one head of one
    fully-filled block, computes one MSE scale per channel over its 16 tokens,
    quantizes, packs the codes per token (identical data layout to per-token),
    and writes the D per-channel fp8 scales into the block's K-scale region.
    """
    pid_blk = tl.program_id(0)
    pid_h = tl.program_id(1)

    phys_blk = tl.load(blk_phys_ptr + pid_blk)
    row0 = tl.load(blk_row0_ptr + pid_blk)  # src row of token 0 of this block

    tok = tl.arange(0, BLOCK_SIZE)  # [16] token-in-block == row offset
    ch = tl.arange(0, D)  # [D]
    # x[t, c] = src[row0 + t, h, c]
    src_off = (row0 + tok)[:, None] * s_src_n + pid_h * s_src_h + ch[None, :]
    x = tl.load(src_ptr + src_off).to(tl.float32)  # [16, D]

    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-9)  # [D] over the 16 tokens
    scale = _mse_scale_axis0(x, amax)  # [D]
    # Round scale to fp8 BEFORE quantizing codes (store/dequant bit-consistent).
    scale_byte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)  # [D]
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)  # [D]
    codes = _quant_codes(x, scale[None, :])  # [16, D] in [-7, 7]

    blk_base = (
        phys_blk * s_cache_blk
        + 0 * s_cache_side  # K side
        + pid_h * s_cache_h
    )

    # --- write data nibbles, packed per token (same layout as per-token) ---
    # For each token t and dblock db: bytes [DATA + ...]; element 2j -> low nibble.
    ci = codes.to(tl.int32) & 0xF  # [16, D]
    nib2 = tl.reshape(ci, (BLOCK_SIZE, DATA_BYTES, 2))  # [16, DATA_BYTES, 2]
    lo, hi = tl.split(nib2)  # each [16, DATA_BYTES]
    packed = (lo | (hi << 4)).to(tl.uint8)  # [16, DATA_BYTES]
    pcol = tl.arange(0, DATA_BYTES)
    data_off = blk_base + tok[:, None] * s_cache_tok + pcol[None, :]
    tl.store(cache_ptr + data_off, packed)

    # --- write D per-channel scales into the block's K-scale region ---
    # channel c -> token (c // ND), scale-byte (c % ND) within DATA_BYTES.. .
    s_tok = ch // ND  # [D]
    s_byte = ch % ND  # [D]
    s_off = blk_base + s_tok * s_cache_tok + DATA_BYTES + s_byte
    tl.store(cache_ptr + s_off, scale_byte)


@triton.jit
def _gather_dequant_kernel(
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table_ptr,  # int32 [B, max_blocks]
    seq_lens_ptr,  # int32 [B]
    out_ptr,  # bf16 [B, H, max_seq, D]
    B,
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    FULL_DIM: tl.constexpr,
    KV_SIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ,
    s_bt,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
    s_out_b,
    s_out_h,
    s_out_s,
):
    """One program per (req, pos, head).  Dequant full head_dim of one token.

    Grid is (pos, req, head): the position axis is on grid.x because it is the
    only dimension that can exceed the CUDA 65535 grid.y/z limit (B and H are
    small).  See ``int4_kivi_gather_dequant`` for the launch.
    """
    pid_pos = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    if pid_pos >= seq_len:
        return

    logical_blk = pid_pos // BLOCK_SIZE
    tok_in_blk = pid_pos % BLOCK_SIZE
    # int64: phys_blk * s_cache_blk is a byte offset into the whole-layer KV
    # cache, which exceeds 2**31 once a layer's cache is >2GB (high
    # gpu_memory_utilization).  block_table is int32, so without this cast the
    # product overflows int32 -> wrong offset -> CUDA illegal memory access.
    phys_blk = tl.load(block_table_ptr + pid_b * s_bt + logical_blk).to(tl.int64)

    blk_base = (
        phys_blk * s_cache_blk
        + KV_SIDE * s_cache_side
        + pid_h * s_cache_h
    )
    row_base = blk_base + tok_in_blk * s_cache_tok

    # int64 for the same reason: B*H*max_seq*D can exceed 2**31 elements.
    out_base = pid_b.to(tl.int64) * s_out_b + pid_h * s_out_h + pid_pos * s_out_s

    # K side: a block is per-channel iff it is FULL (all 16 token slots filled,
    # i.e. (logical_blk+1)*BLOCK_SIZE <= seq_len).  Otherwise (partial trailing
    # block / decode hot tail) K is per-token, like V.  V is always per-token.
    k_per_channel = (KV_SIDE == 0) and (
        (logical_blk + 1) * BLOCK_SIZE <= seq_len
    )

    # loop over dblocks (compile-time unrolled)
    for db in tl.static_range(ND):
        data_off = row_base + db * _PACK + tl.arange(0, _PACK)
        packed = tl.load(cache_ptr + data_off).to(tl.int32)  # [PACK]
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        lo = tl.where(lo >= 8, lo - 16, lo)
        hi = tl.where(hi >= 8, hi - 16, hi)
        codes = tl.interleave(lo, hi).to(tl.float32)  # [BLOCK]

        ch = db * _BLOCK + tl.arange(0, _BLOCK)  # [BLOCK] channel indices
        if k_per_channel:
            # per-channel scale: channel c lives at block scale-byte c,
            # i.e. token (c // ND), scale-byte (c % ND).
            s_addr = blk_base + (ch // ND) * s_cache_tok + DATA_BYTES + (ch % ND)
            sb_pc = tl.load(cache_ptr + s_addr).to(tl.uint8)  # [BLOCK]
            scale = sb_pc.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            sb_pt = tl.load(cache_ptr + row_base + DATA_BYTES + db).to(tl.uint8)
            scale = tl.broadcast_to(
                sb_pt.to(tl.float8e4nv, bitcast=True).to(tl.float32)[None], [_BLOCK]
            )

        deq = codes * scale  # [BLOCK]
        tl.store(out_ptr + out_base + ch, deq.to(tl.bfloat16))


def _find_full_blocks(slot_mapping: torch.Tensor, block_size: int):
    """Identify the fully-filled, src-contiguous blocks present in this store.

    A block ``b`` qualifies for the per-channel-K path iff all 16 of its token
    slots ``b*block_size + [0..block_size)`` are present in ``slot_mapping`` AND
    they occupy 16 *consecutive* src rows (so the kernel can load ``row0..row0+15``
    as a contiguous [16, D] tile).  In vLLM's eager prefill the slot_mapping is
    monotone-contiguous, so a window of 16 rows whose slots are ``s, s+1, .., s+15``
    with ``s % block_size == 0`` is exactly such a block.

    Returns ``(blk_phys, blk_row0, full_mask)`` where ``blk_phys[i]`` is the
    physical block index, ``blk_row0[i]`` is the src row of token 0, and
    ``full_mask`` is a bool [N] marking rows that belong to a full block (those
    rows are handled per-channel; the rest fall back to per-token).
    """
    N = slot_mapping.shape[0]
    dev = slot_mapping.device
    full_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    if N < block_size:
        return (
            torch.empty(0, dtype=torch.int64, device=dev),
            torch.empty(0, dtype=torch.int64, device=dev),
            full_mask,
        )
    sm = slot_mapping
    # candidate block starts: rows r where slot % block_size == 0
    starts = (sm % block_size == 0) & (sm >= 0)
    cand = torch.nonzero(starts, as_tuple=False).flatten()
    cand = cand[cand + block_size <= N]
    if cand.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=dev),
            torch.empty(0, dtype=torch.int64, device=dev),
            full_mask,
        )
    # For each candidate start row r, check slots[r:r+16] == slots[r] + arange(16)
    ar = torch.arange(block_size, device=dev)
    base = sm[cand]  # [C]
    window = sm[cand[:, None] + ar[None, :]]  # [C, 16]
    expect = base[:, None] + ar[None, :]
    ok = (window == expect).all(dim=1)
    cand = cand[ok]
    base = base[ok]
    if cand.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=dev),
            torch.empty(0, dtype=torch.int64, device=dev),
            full_mask,
        )
    blk_phys = (base // block_size).to(torch.int64)
    blk_row0 = cand.to(torch.int64)
    # mark rows belonging to a full block
    rows = (blk_row0[:, None] + ar[None, :]).flatten()
    full_mask[rows] = True
    return blk_phys, blk_row0, full_mask


def int4_kivi_store(
    key: torch.Tensor,  # bf16 [N, H, D]
    value: torch.Tensor,  # bf16 [N, H, D]
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    slot_mapping: torch.Tensor,  # int64 [N]
    head_size: int,
) -> None:
    """Quantize new K/V tokens to INT4 and store them into the paged cache.

    V is stored per-token (head_dim 16-blocks).  K is stored **per-channel** for
    every full 16-token block (one scale per channel over the block's 16 tokens),
    and per-token for partial trailing blocks.
    """
    N, H, D = key.shape
    if N == 0:
        return
    ND = D // BLOCK
    DATA_BYTES = D // 2
    FULL_DIM = DATA_BYTES + ND
    BLOCK_SIZE = kv_cache.shape[2]

    s_cache_blk = kv_cache.stride(0)
    s_cache_side = kv_cache.stride(1)
    s_cache_tok = kv_cache.stride(2)
    s_cache_h = kv_cache.stride(3)

    # --- V: always per-token ---
    v = value.contiguous()
    _store_token_kernel[(N, H, ND)](
        v, kv_cache, slot_mapping, N,
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, FULL_DIM=FULL_DIM,
        KV_SIDE=1, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=v.stride(0), s_src_h=v.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )

    # --- K: per-channel for full blocks, per-token for the rest ---
    k = key.contiguous()
    blk_phys, blk_row0, full_mask = _find_full_blocks(slot_mapping, BLOCK_SIZE)

    if os.environ.get("VLLM_INT4_DEBUG"):
        nfull = int(blk_phys.numel())
        npart = int((~full_mask).sum().item())
        print(
            f"[int4_kivi.store] N={N} full_blocks={nfull} "
            f"per_channel_rows={int(full_mask.sum().item())} "
            f"per_token_rows={npart}",
            flush=True,
        )

    if blk_phys.numel() > 0:
        _store_k_channel_kernel[(blk_phys.numel(), H)](
            k, kv_cache, blk_phys, blk_row0,
            H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, BLOCK_SIZE=BLOCK_SIZE,
            s_src_n=k.stride(0), s_src_h=k.stride(1),
            s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
            s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
        )

    # partial-block K tokens -> per-token (mask full-block rows out via slot=-1)
    if bool(full_mask.any()):
        k_slots = slot_mapping.clone()
        k_slots[full_mask] = -1
    else:
        k_slots = slot_mapping
    if not bool(full_mask.all()):
        _store_token_kernel[(N, H, ND)](
            k, kv_cache, k_slots, N,
            H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, FULL_DIM=FULL_DIM,
            KV_SIDE=0, BLOCK_SIZE=BLOCK_SIZE,
            s_src_n=k.stride(0), s_src_h=k.stride(1),
            s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
            s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
        )


def int4_kivi_gather_dequant(
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table: torch.Tensor,  # int32 [B, max_blocks]
    seq_lens: torch.Tensor,  # int32 [B]
    head_size: int,
    num_kv_heads: int,
    max_seq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant the cached INT4 K and V back to bf16 dense tensors
    [B, H, max_seq, D] for use with flash/SDPA attention."""
    B = seq_lens.shape[0]
    H = num_kv_heads
    D = head_size
    ND = D // BLOCK
    DATA_BYTES = D // 2
    FULL_DIM = DATA_BYTES + ND
    BLOCK_SIZE = kv_cache.shape[2]
    device = kv_cache.device

    k_out = torch.zeros((B, H, max_seq, D), dtype=torch.bfloat16, device=device)
    v_out = torch.zeros((B, H, max_seq, D), dtype=torch.bfloat16, device=device)
    # (pos, req, head): pos goes on grid.x (limit ~2**31) since max_seq is the
    # only axis that can exceed the CUDA grid.y/z limit of 65535; B and H stay
    # on y/z where they comfortably fit.
    grid = (max_seq, B, H)
    for side, out in ((0, k_out), (1, v_out)):
        _gather_dequant_kernel[grid](
            kv_cache,
            block_table,
            seq_lens,
            out,
            B,
            H=H,
            D=D,
            ND=ND,
            DATA_BYTES=DATA_BYTES,
            FULL_DIM=FULL_DIM,
            KV_SIDE=side,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_SEQ=max_seq,
            s_bt=block_table.stride(0),
            s_cache_blk=kv_cache.stride(0),
            s_cache_side=kv_cache.stride(1),
            s_cache_tok=kv_cache.stride(2),
            s_cache_h=kv_cache.stride(3),
            s_out_b=out.stride(0),
            s_out_h=out.stride(1),
            s_out_s=out.stride(2),
        )
    return k_out, v_out
