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

import functools
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

# ---- Fused-decode tuning knobs (env-overridable for sweeps; see
# int4_kivi_paged_decode).  Defaults are the best found on B300 (sm_103). ----
_DECODE_BLOCK_N = int(os.environ.get("VLLM_INT4_DECODE_BLOCK_N", "64"))
_DECODE_NUM_WARPS = int(os.environ.get("VLLM_INT4_DECODE_WARPS", "4"))
_DECODE_NUM_STAGES = int(os.environ.get("VLLM_INT4_DECODE_STAGES", "3"))
# Split-K target: fill ~ WAVES * SM_count programs.  The bf16 tensor-core dot
# only pays off with enough split-parallelism to hide the dequant ALU; too few
# splits (large batch) leaves the inner loop serial and the bf16 cast becomes
# pure overhead.  B300 clean sweep: WAVES=4 never regresses vs the old fp32 path
# and is the per-shape optimum or within ~10% of it across B=1..32 — fewer (=2)
# regresses B=8/16, more over-splits (combine + pacc traffic + tiny segments).
# Very large batch still resolves to SPLIT==1 -> the no-combine WRITE_FINAL path.
_DECODE_WAVES = float(os.environ.get("VLLM_INT4_DECODE_WAVES", "4"))
_DECODE_MAX_SPLIT = int(os.environ.get("VLLM_INT4_DECODE_MAX_SPLIT", "64"))


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
    slot_ptr,  # int64 [N]  slot of each src row
    fbs_ptr,  # uint8 [N]  1 iff src row starts a full, contiguous, aligned block
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
    """One program per (src_row, head).  K per-channel over 16 tokens.

    Grid is (N, H): each program owns the candidate full block STARTING at src
    row ``pid_r`` and early-returns unless ``fbs_ptr[pid_r]`` marks it a real full
    block.  This on-device filter replaces a host-side nonzero/count, so the store
    takes no sync (decode stays CUDA-graph capturable).  For a real full block it
    loads K[16 tokens, D channels] for one head, computes one MSE scale per
    channel over its 16 tokens, quantizes, packs per token (identical data layout
    to per-token), and writes the D per-channel fp8 scales into the K-scale region.
    """
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)

    if tl.load(fbs_ptr + pid_r) == 0:
        return  # not a full-block start -> these tokens go via the per-token path

    row0 = pid_r  # src row of token 0 of this block
    phys_blk = tl.load(slot_ptr + pid_r) // BLOCK_SIZE

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


def _full_block_masks(slot_mapping: torch.Tensor, block_size: int):
    """Sync-free full-block detection (replaces the nonzero-based version).

    A block qualifies for the per-channel-K path iff its ``block_size`` token
    slots are present in ``slot_mapping`` on *consecutive* src rows starting at a
    block boundary (so the kernel can load ``row0..row0+15`` as a contiguous tile).
    In vLLM's eager prefill the slot_mapping is monotone-contiguous, so a run of
    16 rows whose slots are ``s, s+1, .., s+15`` with ``s % block_size == 0`` is
    exactly such a block; decode appends never form one.

    Returns ``(full_block_start, full_mask)`` both bool [N]:
      * ``full_block_start[r]`` -> src row r begins such a run (per-channel K).
      * ``full_mask[r]``        -> row r belongs to such a run (excluded from the
        per-token K store).
    Uses only dense elementwise ops (no ``torch.nonzero`` / ``.item()``), so the
    store issues no host sync and the decode step stays CUDA-graph capturable.
    """
    N = slot_mapping.shape[0]
    dev = slot_mapping.device
    if N < block_size:
        z = torch.zeros(N, dtype=torch.bool, device=dev)
        return z, z
    sm = slot_mapping
    bs = block_size
    idx = torch.arange(N, device=dev)
    aligned = (sm % bs == 0) & (sm >= 0) & (idx + bs <= N)

    # Contiguity via a sliding-window-AND over adjacency, done with one cumsum
    # (no per-offset Python loop -> a fixed handful of kernel launches, not 2*bs).
    # adj[r] == slot[r+1]==slot[r]+1.  A full run starts at r iff the bs-1
    # adjacencies adj[r..r+bs-2] are all true, i.e. their count == bs-1.
    adj = (sm[1:] == sm[:-1] + 1).to(torch.int32)  # [N-1]
    cadj = torch.zeros(N, dtype=torch.int32, device=dev)
    torch.cumsum(adj, dim=0, out=cadj[1:])  # cadj[i] = sum(adj[:i])
    run_ok = torch.zeros(N, dtype=torch.bool, device=dev)
    # r in [0, N-bs]: adj[r..r+bs-2] all true
    hi = cadj[bs - 1 : N]            # cadj[r+bs-1]
    lo = cadj[0 : N - bs + 1]        # cadj[r]
    run_ok[: N - bs + 1] = (hi - lo) == (bs - 1)
    full_block_start = aligned & run_ok

    # Dilate forward by bs: row i is in a full block iff some start in
    # [i-bs+1, i] is set.  Windowed-OR via cumsum of the start mask.
    si = full_block_start.to(torch.int32)
    csi = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    torch.cumsum(si, dim=0, out=csi[1:])  # csi[i+1] = sum(start[:i+1])
    lo_i = torch.clamp(idx - bs + 1, min=0)
    full_mask = (csi[idx + 1] - csi[lo_i]) > 0
    return full_block_start, full_mask


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
    full_block_start, full_mask = _full_block_masks(slot_mapping, BLOCK_SIZE)

    if os.environ.get("VLLM_INT4_DEBUG"):
        nfull = int(full_block_start.sum().item())
        npart = int((~full_mask).sum().item())
        print(
            f"[int4_kivi.store] N={N} full_blocks={nfull} "
            f"per_channel_rows={int(full_mask.sum().item())} "
            f"per_token_rows={npart}",
            flush=True,
        )

    # Per-channel K: grid (N, H); each program filters on-device via
    # full_block_start (no host count -> no sync).  N rides grid.x (limit ~2**31)
    # so a long single-call prefill never hits the 65535 grid.y/z cap.  For decode
    # no row is a full-block start, so every program early-returns (cheap).
    _store_k_channel_kernel[(N, H)](
        k, kv_cache, slot_mapping, full_block_start.to(torch.uint8),
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=k.stride(0), s_src_h=k.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )

    # partial-block K tokens -> per-token.  Full-block rows are masked out with
    # slot=-1 (those programs early-return) since they were stored per-channel
    # above.  Do this with a GPU select + an *unconditional* launch instead of
    # ``if full_mask.any()/all()`` host probes: those .item() syncs stall the
    # stream every layer every step and block CUDA-graph capture of decode.
    # When there are no full blocks (the decode case) k_slots == slot_mapping;
    # when every row is full, all slots are -1 and every program early-returns.
    k_slots = slot_mapping.masked_fill(full_mask, -1)
    _store_token_kernel[(N, H, ND)](
        k, kv_cache, k_slots, N,
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, FULL_DIM=FULL_DIM,
        KV_SIDE=0, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=k.stride(0), s_src_h=k.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )


# =========================================================================== #
# Fused paged flash-decode over the packed INT4-KIVI cache.
#
# The general decode path (``int4_kivi_gather_dequant`` + flash) materializes the
# WHOLE cached context as dense bf16 ``(B, H, max_seq, D)`` every decode step and
# re-reads it — O(B*H*max_seq*D) HBM writes+reads of *bf16* per step, the bulk of
# decode latency at long context.  This kernel instead streams the packed int4
# cache directly: one program handles one ``(request, kv_head, sequence-split)``
# and processes ALL ``GROUP`` query heads of that kv head together (GQA-grouped),
# dequantizing K (per-channel for full blocks, per-token for the trailing partial
# block) and V (per-token) in registers, flash-style online softmax, never
# materializing dense KV.  The K/V reads move ~3.2x fewer bytes (4-bit vs bf16).
#
# Partials ``(B, H, GROUP, SPLIT)`` are reduced by ``_decode_combine_kernel``.
# Layout note: ``((b*H + kvh)*GROUP + gr) == b*n_qh + qh`` so the combine output
# is exactly ``[B*n_qh, D]`` row-major in (b, qh).
# =========================================================================== #
@triton.jit
def _paged_decode_kernel(
    q_ptr,  # bf16 [B, n_qh, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table_ptr,  # int32 [B, max_blocks]
    seq_lens_ptr,  # int32 [B]
    pm_ptr,  # fp32 [B, H, GROUP, SPLIT]
    pl_ptr,  # fp32 [B, H, GROUP, SPLIT]
    pacc_ptr,  # fp32 [B, H, GROUP, SPLIT, D]
    out_ptr,  # bf16 [B*n_qh, D]  (written directly when WRITE_FINAL)
    sm_scale,
    n_qh,
    GROUP: tl.constexpr,
    GPAD: tl.constexpr,  # next-pow2(GROUP), >=16 for tl.dot
    D: tl.constexpr,
    H: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    BLOCK_SIZE: tl.constexpr,  # paged block (page) size in tokens
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tokens streamed per inner step
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
    s_bt,
    WRITE_FINAL: tl.constexpr,  # SPLIT==1: write final out, skip partials+combine
):
    pid = tl.program_id(0)
    s = pid % SPLIT
    tmp = pid // SPLIT
    kvh = tmp % H
    b = tmp // H
    qh0 = kvh * GROUP

    d = tl.arange(0, D)  # [D] head-dim lanes
    db = d // _BLOCK  # [D] dblock of each lane
    cin = d % _BLOCK  # [D] in-block channel
    dbyte = db * _PACK + cin // 2  # [D] data byte holding lane d
    dhi = (cin % 2) == 1  # [D] lane d in high nibble?

    gr = tl.arange(0, GPAD)  # [GPAD] query heads within the group
    gmask = gr < GROUP
    qoff = (b * n_qh + qh0 + gr[:, None]) * D + d[None, :]
    # Keep q in bf16 for the tensor-core dot; fold sm_scale into the qk score
    # after the matmul (so q stays exact bf16, matching FlashAttention).
    q = tl.load(q_ptr + qoff, mask=gmask[:, None], other=0.0).to(tl.bfloat16)

    seq_len = tl.load(seq_lens_ptr + b)
    seg = (seq_len + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, seq_len)

    m_i = tl.full([GPAD], -float("inf"), tl.float32)
    l_i = tl.zeros([GPAD], tl.float32)
    acc = tl.zeros([GPAD, D], tl.float32)

    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        tmask = tok < seg1
        logical_blk = tok // BLOCK_SIZE
        tok_in_blk = tok % BLOCK_SIZE
        phys = tl.load(
            block_table_ptr + b * s_bt + logical_blk, mask=tmask, other=0
        ).to(tl.int64)  # [BLOCK_N]
        is_full = (logical_blk + 1) * BLOCK_SIZE <= seq_len  # [BLOCK_N] per-channel K?

        # byte offset of each token's K row (side 0) and V row (side 1).
        rowK = phys * s_cache_blk + tok_in_blk.to(tl.int64) * s_cache_tok + kvh * s_cache_h
        rowV = rowK + s_cache_side
        blkK = phys * s_cache_blk + kvh * s_cache_h  # block base (K side, scales)

        # ---- dequant K -> [D, BLOCK_N] ----
        koff = rowK[None, :] + dbyte[:, None]  # [D, BLOCK_N]
        kb = tl.load(cache_ptr + koff, mask=tmask[None, :], other=0).to(tl.int32)
        knib = tl.where(dhi[:, None], (kb >> 4) & 0xF, kb & 0xF)
        knib = tl.where(knib >= 8, knib - 16, knib)
        # scale: per-channel for full blocks, per-token for the partial tail.
        ks_pc = blkK[None, :] + (d // ND)[:, None] * s_cache_tok + DATA_BYTES + (d % ND)[:, None]
        ks_pt = rowK[None, :] + DATA_BYTES + db[:, None]
        ksaddr = tl.where(is_full[None, :], ks_pc, ks_pt)
        ksb = tl.load(cache_ptr + ksaddr, mask=tmask[None, :], other=0).to(tl.uint8)
        ksc = ksb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Dequant in fp32, then cast to bf16 so the QK^T matmul runs on bf16
        # tensor cores (fp32 accumulate).  This is the dominant speedup lever:
        # fp32 tl.dot does not hit the tensor-core path.  K/V are stored from
        # bf16 anyway, so the bf16 cast does not add meaningful quant error.
        kdeq = (knib.to(tl.float32) * ksc).to(tl.bfloat16)  # [D, BLOCK_N]

        qk = tl.dot(q, kdeq, out_dtype=tl.float32) * sm_scale  # [GPAD, BLOCK_N]
        qk = tl.where(tmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))  # [GPAD]
        alpha = tl.exp(m_i - m_new)
        pblk = tl.exp(qk - m_new[:, None])
        pblk = tl.where(tmask[None, :], pblk, 0.0)  # [GPAD, BLOCK_N]

        # ---- dequant V -> [BLOCK_N, D] (always per-token) ----
        voff = rowV[:, None] + dbyte[None, :]  # [BLOCK_N, D]
        vb = tl.load(cache_ptr + voff, mask=tmask[:, None], other=0).to(tl.int32)
        vnib = tl.where(dhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
        vnib = tl.where(vnib >= 8, vnib - 16, vnib)
        vsaddr = rowV[:, None] + DATA_BYTES + db[None, :]  # [BLOCK_N, D]
        vsb = tl.load(cache_ptr + vsaddr, mask=tmask[:, None], other=0).to(tl.uint8)
        vsc = vsb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        vdeq = (vnib.to(tl.float32) * vsc).to(tl.bfloat16)  # [BLOCK_N, D]

        # P @ V on bf16 tensor cores (fp32 accumulate); acc stays fp32.
        acc = acc * alpha[:, None] + tl.dot(
            pblk.to(tl.bfloat16), vdeq, out_dtype=tl.float32
        )  # [GPAD, D]
        l_i = l_i * alpha + tl.sum(pblk, axis=1)
        m_i = m_new
        t += BLOCK_N

    if WRITE_FINAL:
        # SPLIT==1: this program saw the whole sequence, so finish the softmax
        # here and write the final attention output — no partials, no combine.
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)  # empty-request guard
        o = acc / l_safe[:, None]
        orow = b * n_qh + qh0 + gr  # [GPAD] == output row (b, qh)
        tl.store(out_ptr + orow[:, None] * D + d[None, :], o.to(tl.bfloat16),
                 mask=gmask[:, None])
    else:
        base = ((b * H + kvh) * GROUP + gr) * SPLIT + s  # [GPAD]
        tl.store(pm_ptr + base, m_i, mask=gmask)
        tl.store(pl_ptr + base, l_i, mask=gmask)
        tl.store(pacc_ptr + base[:, None] * D + d[None, :], acc, mask=gmask[:, None])


@triton.jit
def _decode_combine_kernel(
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    out_ptr,  # bf16 [B*n_qh, D]
    D: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)  # over B*n_qh
    d = tl.arange(0, D)
    m = -float("inf")
    for sp in range(0, SPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + pid * SPLIT + sp))
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for sp in range(0, SPLIT):
        ms = tl.load(pm_ptr + pid * SPLIT + sp)
        ls = tl.load(pl_ptr + pid * SPLIT + sp)
        a = tl.load(pacc_ptr + (pid * SPLIT + sp) * D + d)
        scale = tl.exp(ms - m)
        acc += a * scale
        l += ls * scale
    l = tl.where(l > 0.0, l, 1.0)  # empty request guard (l==0 -> out 0)
    tl.store(out_ptr + pid * D + d, (acc / l).to(tl.bfloat16))


def int4_kivi_paged_decode(
    q: torch.Tensor,  # bf16 [N, Hq, D], N == B (one decode token per request)
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table: torch.Tensor,  # int32 [B, max_blocks]
    seq_lens: torch.Tensor,  # int32 [B]
    sm_scale: float,
    split: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Fused INT4-KIVI flash-decode: attention straight off the packed cache.

    Returns ``out`` bf16 [N, Hq, D].  Equivalent to dequantizing the whole cache
    and running causal attention with a single query per request, but without
    materializing the dense bf16 KV.

    ``split`` (split-K over the sequence) is chosen automatically from the GPU's
    SM count so small batches still fill the device; pass an int to override.
    With ``split == 1`` the decode kernel writes the final output directly and
    the combine launch is skipped (the large-batch throughput path).
    """
    N, Hq, D = q.shape
    B = seq_lens.shape[0]
    assert N == B, "paged decode expects exactly one query token per request"
    H = kv_cache.shape[3]
    GROUP = Hq // H
    ND = D // BLOCK
    DATA_BYTES = D // 2
    BLOCK_SIZE = kv_cache.shape[2]
    dev = q.device

    if block_n is None:
        block_n = _DECODE_BLOCK_N
    # Pick split WITHOUT a host sync on seq_lens: a max().item() here would block
    # the stream every layer every step and bar CUDA-graph capture.  Oversized /
    # empty splits are numerically correct (they contribute m=-inf, l=0, acc=0,
    # handled by the combine guard), so we don't need max_seq to bound this.
    if split is None:
        env = os.environ.get("VLLM_INT4_DECODE_SPLIT")
        if env is not None:
            split = int(env)
        else:
            target = int(_sm_count(dev.index) * _DECODE_WAVES)
            split = max(1, min(_DECODE_MAX_SPLIT, -(-target // (B * H))))
    GPAD = 1 << max(0, (GROUP - 1)).bit_length()
    GPAD = max(GPAD, 16)  # tl.dot needs M >= 16

    qc = q.reshape(B, Hq, D).contiguous()
    bt = block_table.to(torch.int32)
    sl = seq_lens.to(torch.int32)

    out = torch.empty((B * Hq, D), dtype=torch.bfloat16, device=dev)
    npart = B * H * GROUP * split
    pm = torch.empty((npart,), dtype=torch.float32, device=dev)
    pl = torch.empty((npart,), dtype=torch.float32, device=dev)
    pacc = torch.empty((npart, D), dtype=torch.float32, device=dev)
    write_final = split == 1

    _paged_decode_kernel[(B * H * split,)](
        qc, kv_cache, bt, sl, pm, pl, pacc, out,
        sm_scale, Hq,
        GROUP=GROUP, GPAD=GPAD, D=D, H=H, ND=ND, DATA_BYTES=DATA_BYTES,
        BLOCK_SIZE=BLOCK_SIZE, SPLIT=split, BLOCK_N=block_n,
        s_cache_blk=kv_cache.stride(0), s_cache_side=kv_cache.stride(1),
        s_cache_tok=kv_cache.stride(2), s_cache_h=kv_cache.stride(3),
        s_bt=bt.stride(0), WRITE_FINAL=write_final,
        num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
    )
    if not write_final:
        _decode_combine_kernel[(B * Hq,)](pm, pl, pacc, out, D=D, SPLIT=split)
    return out.reshape(N, Hq, D)


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
