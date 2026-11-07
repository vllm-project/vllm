# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 index-cache layout and glue for AITER's gfx942 FP4 scorers.

A 128-token cache block holds two 64-token kernel pages, each stored as
``[packed data 4096 B | e8m0 scales 256 B]`` in AITER's preshuffled order, so a
single page stride addresses both operands and the scorers read the cache in
place.
"""

import torch

from vllm.triton_utils import tl, triton

try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
except ImportError:
    _mxfp4_quant_op = None

HEAD_DIM = 128
KERNEL_PAGE_TOKENS = 64
MANAGER_BLOCK_TOKENS = 128
KERNEL_PAGES_PER_BLOCK = MANAGER_BLOCK_TOKENS // KERNEL_PAGE_TOKENS
PACKED_DIM = HEAD_DIM // 2
K_CHUNKS = HEAD_DIM // 32
K_CHUNK_BYTES = 16
# The row-packed scorer's MFMA M tile: 16 query rows share each decoded K tile.
KERNEL_HEADS = 16
PAGE_DATA_BYTES = KERNEL_PAGE_TOKENS * PACKED_DIM
PAGE_SCALE_BYTES = KERNEL_PAGE_TOKENS * K_CHUNKS
PAGE_SLOT_BYTES = PAGE_DATA_BYTES + PAGE_SCALE_BYTES
MANAGER_BLOCK_BYTES = KERNEL_PAGES_PER_BLOCK * PAGE_SLOT_BYTES


def expand_block_table_page64(block_table: torch.Tensor) -> torch.Tensor:
    """Map each 128-token block id to its two 64-token page ids."""
    if block_table.dim() != 2:
        raise ValueError("MiniMax-M3 block table must be rank 2")
    pages = block_table.to(torch.int64).unsqueeze(-1) * KERNEL_PAGES_PER_BLOCK
    pages = pages + torch.arange(
        KERNEL_PAGES_PER_BLOCK, dtype=torch.int64, device=block_table.device
    )
    return pages.flatten(1).to(block_table.dtype)


def expand_block_table_page64_into(
    block_table: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``expand_block_table_page64`` into a persistent buffer, overwriting every
    mapped column."""
    batch, blocks = block_table.shape
    if out.shape[0] < batch or out.shape[1] < KERNEL_PAGES_PER_BLOCK * blocks:
        raise ValueError(
            f"page table {tuple(out.shape)} too small for {batch}x{blocks}"
        )
    if not block_table.is_cuda:
        pages = out[:batch, : KERNEL_PAGES_PER_BLOCK * blocks]
        for p in range(KERNEL_PAGES_PER_BLOCK):
            col = pages[:, p::KERNEL_PAGES_PER_BLOCK]
            torch.mul(block_table, KERNEL_PAGES_PER_BLOCK, out=col)
            if p:
                col.add_(p)
        return out[:batch]
    if batch and blocks:
        # One launch (this runs on the host every decode step, outside the
        # captured graph).
        BLOCK = 1024
        _expand_page64_kernel[(batch, triton.cdiv(blocks, BLOCK))](
            block_table,
            out,
            blocks,
            block_table.stride(0),
            out.stride(0),
            PAGES=KERNEL_PAGES_PER_BLOCK,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out[:batch]


@triton.jit
def _expand_page64_kernel(
    bt_ptr,
    out_ptr,
    blocks,
    stride_bt,
    stride_out,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = j < blocks
    v = tl.load(bt_ptr + b.to(tl.int64) * stride_bt + j, mask=m, other=0)
    for p in tl.static_range(PAGES):
        tl.store(
            out_ptr + b.to(tl.int64) * stride_out + j * PAGES + p, v * PAGES + p, mask=m
        )


def _check_cache(cache: torch.Tensor) -> torch.Tensor:
    cache = cache.reshape(cache.shape[0], -1)
    if cache.dtype != torch.uint8 or cache.shape[1] != MANAGER_BLOCK_BYTES:
        raise ValueError(
            f"gfx942 MXFP4 index cache needs uint8 [blocks, {MANAGER_BLOCK_BYTES}], "
            f"got {cache.dtype} {tuple(cache.shape)}"
        )
    if cache.stride(1) != 1 or cache.stride(0) != MANAGER_BLOCK_BYTES:
        raise ValueError("gfx942 MXFP4 index cache needs contiguous blocks")
    return cache


def gfx942_mxfp4_cache_strides(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """The cache as the scorers' ``kv_cache``/``kv_scale`` operands, in place."""
    cache = _check_cache(cache)
    return (
        cache,
        cache,
        dict(
            kv_page_stride=PAGE_SLOT_BYTES,
            kv_scale_page_stride=PAGE_SLOT_BYTES,
            kv_scale_offset=PAGE_DATA_BYTES,
        ),
    )


def gfx942_mxfp4_cache_views(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense copies of the pages' data and scales, in AITER's default layout."""
    slots = _check_cache(cache).view(-1, PAGE_SLOT_BYTES)
    data = slots[:, :PAGE_DATA_BYTES].contiguous()
    scales = slots[:, PAGE_DATA_BYTES:].contiguous()
    return (
        data.view(-1, 1, K_CHUNKS, KERNEL_PAGE_TOKENS, K_CHUNK_BYTES),
        scales.view(-1, 1, K_CHUNKS, KERNEL_PAGE_TOKENS),
    )


@triton.jit
def _pack_decode_rows_kernel(
    q_ptr,
    packed_ptr,
    scale_ptr,
    meta_ptr,
    seq_lens_ptr,
    stride_q_row,
    stride_q_head,
    rows,
    slots,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    D: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Slot ``h * QUERY_LEN + r`` of request ``b``'s 16-row group holds index
    head ``h`` of its decode row ``r``: MXFP4 data, its e8m0 bytes in the
    row-packed scale layout and the slot's output row and causal end."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    slot = b * SLOTS + s
    live = s < HEADS * QUERY_LEN
    head = s // QUERY_LEN
    r = s % QUERY_LEN
    row = b * QUERY_LEN + r
    d = tl.arange(0, D)
    x = tl.load(
        q_ptr + row.to(tl.int64) * stride_q_row + head * stride_q_head + d,
        mask=live & (d < D),
        other=0.0,
    ).to(tl.float32)
    q, e8m0 = _mxfp4_quant_op(tl.reshape(x, (1, D)), D, 1, 32)
    t = tl.arange(0, D // 2)
    tl.store(
        packed_ptr + slot.to(tl.int64) * (D // 2) + t,
        tl.reshape(q, (D // 2,)).to(tl.uint8),
    )
    # [group, chunk, slot, pad=4]: byte 0 of each slot's pad is live.
    c = tl.arange(0, D // 32)
    tl.store(
        scale_ptr + b.to(tl.int64) * (D // 32) * SLOTS * 4 + c * SLOTS * 4 + s * 4,
        tl.reshape(e8m0, (D // 32,)).to(tl.uint8),
    )
    seq_len = tl.load(seq_lens_ptr + b)
    end = tl.maximum(seq_len - QUERY_LEN + r + 1, 0)
    # Dead slots are never stored: output row -1, empty window.
    tl.store(meta_ptr + slot, tl.where(live, head * rows + row, -1))
    tl.store(meta_ptr + slots + slot, 0)
    tl.store(meta_ptr + 2 * slots + slot, tl.where(live, end, 0))


def pack_decode_rows_gfx942(
    index_query: torch.Tensor,  # [batch * query_len, H, 128]
    seq_lens: torch.Tensor,  # [batch] int32
    q_fp4: torch.Tensor,  # [>= batch, 16, 64] uint8
    q_scale: torch.Tensor,  # [>= batch, 1, 4, 16, 4] uint8
    slot_meta: torch.Tensor,  # [>= 3 * 16 * batch] int32, flat
    *,
    query_len: int,
) -> torch.Tensor:
    """Pack every (index head, decode row) of a request into one 16-slot MFMA
    group for AITER's row-packed page-max scorer; returns the ``[3, slots]``
    slot metadata view. Score rows are laid out ``head * rows + row``."""
    batch = seq_lens.shape[0]
    heads = index_query.shape[1]
    if heads * query_len > KERNEL_HEADS:
        raise ValueError(
            f"{heads} index heads x {query_len} decode rows exceed the "
            f"{KERNEL_HEADS}-slot MFMA group"
        )
    slots = batch * KERNEL_HEADS
    meta = slot_meta[: 3 * slots].view(3, slots)
    if batch:
        _pack_decode_rows_kernel[(batch, KERNEL_HEADS)](
            index_query,
            q_fp4,
            q_scale,
            meta,
            seq_lens,
            index_query.stride(0),
            index_query.stride(1),
            batch * query_len,
            slots,
            HEADS=heads,
            QUERY_LEN=query_len,
            D=HEAD_DIM,
            SLOTS=KERNEL_HEADS,
            num_warps=1,
        )
    return meta


@triton.jit
def store_k_row_gfx942(
    x,  # [D] fp32 (already rounded to the activation dtype)
    cache_ptr,
    slot,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    PAGE_DATA: tl.constexpr,
    PAGE_SLOT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """MXFP4-quantize one index-K row and store it into its cache slot:
    the in-kernel twin of ``store_k_gfx942`` (same quant op, same layout),
    for fusing into the producer of K."""
    valid = slot >= 0
    s = tl.maximum(slot, 0).to(tl.int64)
    block = s // BLOCK_TOKENS
    in_block = s % BLOCK_TOKENS
    page_off = in_block % PAGE_TOKENS
    page = block * BLOCK_BYTES + (in_block // PAGE_TOKENS) * PAGE_SLOT
    q, e8m0 = _mxfp4_quant_op(tl.reshape(x, (1, D)), D, 1, 32)
    byte = tl.arange(0, D // 2)
    data_off = (byte // CHUNK_BYTES * PAGE_TOKENS + page_off) * CHUNK_BYTES + (
        byte % CHUNK_BYTES
    )
    tl.store(
        cache_ptr + page + data_off, tl.reshape(q, (D // 2,)).to(tl.uint8), mask=valid
    )
    chunk = tl.arange(0, D // 32)
    scale_off = PAGE_DATA + chunk * PAGE_TOKENS + (page_off % 16) * 4 + page_off // 16
    tl.store(
        cache_ptr + page + scale_off,
        tl.reshape(e8m0, (D // 32,)).to(tl.uint8),
        mask=valid,
    )


# e4m3fnuz max is 240; mapping max|q| to 224 keeps a margin for rounding.
FP8Q_ROW_MAX = 224.0


@triton.jit
def _pack_decode_rows_fp8_kernel(
    q_ptr,
    packed_ptr,
    scale_ptr,
    meta_ptr,
    seq_lens_ptr,
    stride_q_row,
    stride_q_head,
    rows,
    slots,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    D: tl.constexpr,
    SLOTS: tl.constexpr,
    ROW_MAX: tl.constexpr,
):
    """FP8-Q variant of ``_pack_decode_rows_kernel``: slot ``h * QUERY_LEN + r``
    of request ``b``'s group holds e4m3fnuz ``q * ROW_MAX / max|q|`` and the
    fp32 per-slot scale ``max|q| / ROW_MAX``."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    slot = b * SLOTS + s
    live = s < HEADS * QUERY_LEN
    head = s // QUERY_LEN
    r = s % QUERY_LEN
    row = b * QUERY_LEN + r
    d = tl.arange(0, D)
    x = tl.load(
        q_ptr + row.to(tl.int64) * stride_q_row + head * stride_q_head + d,
        mask=live & (d < D),
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # Correctly rounded divides, so the bytes and scale are reproducible by a
    # host-side (torch) quantization.
    safe = tl.where(amax > 0, amax, 1.0)
    inv = tl.where(amax > 0, tl.math.div_rn(ROW_MAX + tl.zeros_like(safe), safe), 0.0)
    q8 = (x * inv).to(tl.float8e4b8)
    tl.store(packed_ptr + slot.to(tl.int64) * D + d, q8.to(tl.uint8, bitcast=True))
    tl.store(scale_ptr + slot, tl.math.div_rn(amax, ROW_MAX + tl.zeros_like(amax)))
    seq_len = tl.load(seq_lens_ptr + b)
    end = tl.maximum(seq_len - QUERY_LEN + r + 1, 0)
    tl.store(meta_ptr + slot, tl.where(live, head * rows + row, -1))
    tl.store(meta_ptr + slots + slot, 0)
    tl.store(meta_ptr + 2 * slots + slot, tl.where(live, end, 0))


def pack_decode_rows_fp8_gfx942(
    index_query: torch.Tensor,  # [batch * query_len, H, 128]
    seq_lens: torch.Tensor,  # [batch] int32
    q_fp8: torch.Tensor,  # [>= batch, 16, 128] uint8 (e4m3fnuz bytes)
    q_scale: torch.Tensor,  # [>= batch, 16] fp32
    slot_meta: torch.Tensor,  # [>= 3 * 16 * batch] int32, flat
    *,
    query_len: int,
) -> torch.Tensor:
    """FP8-Q counterpart of :func:`pack_decode_rows_gfx942` for the row-packed
    scorer's ``q_fp8`` mode: Q is e4m3fnuz with one fp32 scale per slot (max
    |q| -> 224). Returns the ``[3, slots]`` slot metadata view."""
    batch = seq_lens.shape[0]
    heads = index_query.shape[1]
    if heads * query_len > KERNEL_HEADS:
        raise ValueError(
            f"{heads} index heads x {query_len} decode rows exceed the "
            f"{KERNEL_HEADS}-slot MFMA group"
        )
    slots = batch * KERNEL_HEADS
    meta = slot_meta[: 3 * slots].view(3, slots)
    if batch:
        _pack_decode_rows_fp8_kernel[(batch, KERNEL_HEADS)](
            index_query,
            q_fp8,
            q_scale,
            meta,
            seq_lens,
            index_query.stride(0),
            index_query.stride(1),
            batch * query_len,
            slots,
            HEADS=heads,
            QUERY_LEN=query_len,
            D=HEAD_DIM,
            SLOTS=KERNEL_HEADS,
            ROW_MAX=FP8Q_ROW_MAX,
            num_warps=1,
        )
    return meta


@triton.jit
def _store_k_kernel(
    packed_ptr,
    scale_ptr,
    cache_ptr,
    slots_ptr,
    stride_p_tok,
    stride_p_byte,
    stride_s_tok,
    stride_s_chunk,
    PACKED: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    PAGE_DATA: tl.constexpr,
    PAGE_SLOT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots_ptr + token)
    valid = slot >= 0
    block = slot // BLOCK_TOKENS
    in_block = slot % BLOCK_TOKENS
    page_off = in_block % PAGE_TOKENS
    page = block * BLOCK_BYTES + (in_block // PAGE_TOKENS) * PAGE_SLOT

    byte = tl.arange(0, PACKED)
    value = tl.load(
        packed_ptr + token * stride_p_tok + byte * stride_p_byte, mask=valid, other=0
    )
    data_off = (
        byte // CHUNK_BYTES * PAGE_TOKENS + page_off
    ) * CHUNK_BYTES + byte % CHUNK_BYTES
    tl.store(cache_ptr + page + data_off, value, mask=valid)

    chunk = tl.arange(0, CHUNKS)
    scale = tl.load(
        scale_ptr + token * stride_s_tok + chunk * stride_s_chunk, mask=valid, other=0
    )
    scale_off = PAGE_DATA + chunk * PAGE_TOKENS + (page_off % 16) * 4 + page_off // 16
    tl.store(cache_ptr + page + scale_off, scale, mask=valid)


def store_k_gfx942(
    index_k: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """Quantize normed/rotated index K ``[tokens, 128]`` into its cache slots."""
    if index_k.dim() != 2 or index_k.shape[1] != HEAD_DIM:
        raise ValueError("gfx942 MXFP4 index K requires [tokens, 128]")
    if slot_mapping.shape != (index_k.shape[0],):
        raise ValueError("gfx942 MXFP4 index slots must be a length-token vector")
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    cache = _check_cache(cache)
    packed, scales = dynamic_mxfp4_quant(index_k)
    scales = scales.view(torch.uint8).view(index_k.shape[0], K_CHUNKS)
    _store_k_kernel[(index_k.shape[0],)](
        packed,
        scales,
        cache,
        slot_mapping,
        packed.stride(0),
        packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        PACKED=PACKED_DIM,
        BLOCK_TOKENS=MANAGER_BLOCK_TOKENS,
        PAGE_TOKENS=KERNEL_PAGE_TOKENS,
        CHUNKS=K_CHUNKS,
        CHUNK_BYTES=K_CHUNK_BYTES,
        PAGE_DATA=PAGE_DATA_BYTES,
        PAGE_SLOT=PAGE_SLOT_BYTES,
        BLOCK_BYTES=MANAGER_BLOCK_BYTES,
        num_warps=4,
    )


@triton.jit
def _e2m1(code):
    m = code & 7
    ex = (m >> 1).to(tl.float32)
    mag = tl.where(
        m < 2,
        m.to(tl.float32) * 0.5,
        (1.0 + 0.5 * (m & 1).to(tl.float32)) * tl.exp2(ex - 1.0),
    )
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def _pages_to_fp8_kernel(
    cache_ptr,  # [blocks, BLOCK_BYTES] uint8 MXFP4 index cache
    src_ptr,  # [cap] int32 physical block of compact block i (-1: unused)
    out_ptr,  # [cap, 128, 128] fp8 (e4m3fnuz)
    PAGE_TOKENS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    PAGE_DATA: tl.constexpr,
    PAGE_SLOT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """One (compact block, 64-token page, 32-dim chunk) MXFP4 -> e4m3fnuz.

    e2m1 * 2^(e-127) is exactly representable in e4m3fnuz whenever the scaled
    value is a normal/subnormal e4m3fnuz number (|x| in [2^-10, 240]); the
    index K after its RMSNorm sits well inside that range.
    """
    i = tl.program_id(0)
    half = tl.program_id(1)
    c = tl.program_id(2)
    phys = tl.load(src_ptr + i)
    if phys < 0:
        return
    page = cache_ptr + phys.to(tl.int64) * BLOCK_BYTES + half * PAGE_SLOT
    t = tl.arange(0, PAGE_TOKENS)
    b = tl.arange(0, CHUNK_BYTES)
    raw = tl.load(page + (c * PAGE_TOKENS + t[:, None]) * CHUNK_BYTES + b[None, :]).to(
        tl.int32
    )
    e = tl.load(page + PAGE_DATA + c * PAGE_TOKENS + (t % 16) * 4 + t // 16)
    scale = tl.exp2(e.to(tl.float32) - 127.0)
    lo = _e2m1(raw & 15) * scale[:, None]
    hi = _e2m1(raw >> 4) * scale[:, None]
    vals = tl.interleave(lo, hi)  # [64, 32], dim 2*byte + {lo, hi}
    d = tl.arange(0, 2 * CHUNK_BYTES)
    dst = (
        out_ptr
        + i.to(tl.int64) * (2 * PAGE_TOKENS) * 128
        + (half * PAGE_TOKENS + t[:, None]) * 128
        + c * (2 * CHUNK_BYTES)
        + d[None, :]
    )
    tl.store(dst, vals.to(out_ptr.dtype.element_ty))


def mxfp4_pages_to_fp8_gfx942(
    cache: torch.Tensor, src_blocks: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Gather ``src_blocks`` of the MXFP4 index cache into a compact e4m3fnuz
    ``[cap, 128, 128]`` copy (blocks with src -1 are skipped)."""
    cache = _check_cache(cache)
    cap = src_blocks.shape[0]
    if cap:
        _pages_to_fp8_kernel[(cap, KERNEL_PAGES_PER_BLOCK, K_CHUNKS)](
            cache,
            src_blocks,
            out,
            PAGE_TOKENS=KERNEL_PAGE_TOKENS,
            CHUNK_BYTES=K_CHUNK_BYTES,
            PAGE_DATA=PAGE_DATA_BYTES,
            PAGE_SLOT=PAGE_SLOT_BYTES,
            BLOCK_BYTES=MANAGER_BLOCK_BYTES,
            num_warps=2,
        )
    return out


@triton.jit
def _decode_page_max_kernel(
    page_max_ptr,
    seq_lens_ptr,
    out_ptr,
    rows,
    query_len,
    init_blocks,
    local_blocks,
    stride_ph,
    stride_pr,
    stride_oh,
    stride_or,
    BLOCK: tl.constexpr,
    SENTINEL_INIT: tl.constexpr,
    SENTINEL_LOCAL: tl.constexpr,
):
    """Two 64-token page maxima -> one pinned 128-token block score, reading
    only the row's causal blocks (the page buffer past them is never written)."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + row // query_len)
    kv_len = tl.maximum(seq_len - query_len + row % query_len + 1, 0)
    valid = (kv_len + 127) // 128
    src = page_max_ptr + head.to(tl.int64) * stride_ph + row.to(tl.int64) * stride_pr
    dst = out_ptr + head.to(tl.int64) * stride_oh + row.to(tl.int64) * stride_or
    for start in range(0, valid, BLOCK):
        blk = start + tl.arange(0, BLOCK)
        live = blk < valid
        pair = tl.load(
            src + 2 * blk[:, None] + tl.arange(0, 2)[None, :],
            mask=live[:, None]
            & ((2 * blk[:, None] + tl.arange(0, 2)[None, :]) * 64 < kv_len),
            other=float("-inf"),
        )
        score = tl.max(pair, axis=1)
        score = tl.where(blk < init_blocks, SENTINEL_INIT, score)
        score = tl.where(
            blk >= tl.maximum(valid - local_blocks, 0), SENTINEL_LOCAL, score
        )
        tl.store(dst + blk, score, mask=live)


def decode_page_max_to_block_scores_gfx942(
    page_max: torch.Tensor,  # [H, rows, >= 2 * blocks] fp32 64-token page maxima
    seq_lens: torch.Tensor,
    out: torch.Tensor,  # [H, rows, width] fp32
    *,
    query_len: int,
    init_blocks: int,
    local_blocks: int,
    sentinel_init: float,
    sentinel_local: float,
) -> torch.Tensor:
    """Row-packed scorer page maxima to pinned 128-token block scores.

    Only each row's causal blocks are read and written: the top-k reads no
    further, and the scorer leaves pages past a row's window unwritten."""
    heads, rows, _ = out.shape
    if rows:
        _decode_page_max_kernel[(rows, heads)](
            page_max,
            seq_lens,
            out,
            rows,
            query_len,
            init_blocks,
            local_blocks,
            page_max.stride(0),
            page_max.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=512,
            SENTINEL_INIT=sentinel_init,
            SENTINEL_LOCAL=sentinel_local,
            num_warps=4,
        )
    return out
