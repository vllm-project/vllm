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
# The scorer's MFMA M tile; MiniMax fills slot 0 and zero-weights the rest.
KERNEL_HEADS = 16
Q_SCALE_PAD = 4
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
    pages = out[:batch, : KERNEL_PAGES_PER_BLOCK * blocks]
    for p in range(KERNEL_PAGES_PER_BLOCK):
        col = pages[:, p::KERNEL_PAGES_PER_BLOCK]
        torch.mul(block_table, KERNEL_PAGES_PER_BLOCK, out=col)
        if p:
            col.add_(p)
    return out[:batch]


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
def _quantize_q_kernel(
    q_ptr,
    packed_ptr,
    scale_ptr,
    weights_ptr,
    stride_q,
    D: tl.constexpr,
    SLOTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunks: tl.constexpr = D // 32
    x = tl.load(q_ptr + row * stride_q + tl.arange(0, D)).to(tl.float32)
    q, e8m0 = _mxfp4_quant_op(tl.reshape(x, (1, D)), D, 1, 32)

    tile: tl.constexpr = SLOTS * D // 2
    t = tl.arange(0, tile)
    qv = tl.reshape(q, (D // 2,)).to(tl.int32)
    packed = tl.sum(
        tl.where(t[:, None] == tl.arange(0, D // 2)[None, :], qv[None, :], 0), axis=1
    )
    tl.store(packed_ptr + row * tile + t, packed.to(tl.uint8))

    # [chunk][slot][pad]: slot 0 carries the row's e8m0, every other byte is 0.
    n: tl.constexpr = chunks * SLOTS * 4
    s = tl.arange(0, n)
    e = tl.reshape(e8m0, (chunks,)).to(tl.int32)
    scale = tl.sum(
        tl.where((s[:, None] // (SLOTS * 4)) == tl.arange(0, chunks)[None, :], e, 0),
        axis=1,
    )
    scale = tl.where(s % (SLOTS * 4) == 0, scale, 0)
    tl.store(scale_ptr + row * n + s, scale.to(tl.uint8))

    w = tl.where(tl.arange(0, SLOTS) == 0, 1.0, 0.0)
    tl.store(weights_ptr + row * SLOTS + tl.arange(0, SLOTS), w.to(tl.bfloat16))


def quantize_q_gfx942(
    index_query: torch.Tensor, head: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 of one index-Q head as the scorer's 16-slot tile, its preshuffled
    scales and the one-hot slot weights."""
    rows = index_query.shape[0]
    src = index_query[:, head]
    if src.stride(-1) != 1:
        src = src.contiguous()
    dev = index_query.device
    packed = torch.empty(
        (rows, KERNEL_HEADS, PACKED_DIM), dtype=torch.uint8, device=dev
    )
    scales = torch.empty(
        (rows, 1, K_CHUNKS, KERNEL_HEADS, Q_SCALE_PAD), dtype=torch.uint8, device=dev
    )
    weights = torch.empty((rows, KERNEL_HEADS), dtype=torch.bfloat16, device=dev)
    if rows:
        _quantize_q_kernel[(rows,)](
            src,
            packed,
            scales,
            weights,
            src.stride(0),
            D=HEAD_DIM,
            SLOTS=KERNEL_HEADS,
            num_warps=1,
        )
    return packed, scales, weights


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
def _decode_block_scores_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    max_seq_len,
    width,
    query_len,
    init_blocks,
    local_blocks,
    stride_l,
    stride_o,
    BLOCK_TOKENS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SENTINEL_INIT: tl.constexpr,
    SENTINEL_LOCAL: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    tok = blk[:, None] * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)[None, :]
    x = tl.load(
        logits_ptr + row.to(tl.int64) * stride_l + tok,
        mask=tok < max_seq_len,
        other=float("-inf"),
    )
    score = tl.max(x, axis=1)
    seq_len = tl.load(seq_lens_ptr + row // query_len)
    kv_len = tl.maximum(seq_len - query_len + row % query_len + 1, 0)
    valid = (kv_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    in_ctx = blk < valid
    score = tl.where(in_ctx & (blk < init_blocks), SENTINEL_INIT, score)
    local = in_ctx & (blk >= tl.maximum(valid - local_blocks, 0))
    score = tl.where(local, SENTINEL_LOCAL, score)
    tl.store(out_ptr + row.to(tl.int64) * stride_o + blk, score, mask=blk < width)


def decode_block_scores_gfx942(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    query_len: int,
    init_blocks: int,
    local_blocks: int,
    sentinel_init: float,
    sentinel_local: float,
) -> torch.Tensor:
    """Token logits ``[rows, max_seq_len]`` to pinned 128-token block scores."""
    rows, max_seq_len = logits.shape
    blocks = 16
    _decode_block_scores_kernel[(rows, triton.cdiv(out.shape[1], blocks))](
        logits,
        seq_lens,
        out,
        max_seq_len,
        out.shape[1],
        query_len,
        init_blocks,
        local_blocks,
        logits.stride(0),
        out.stride(0),
        BLOCK_TOKENS=MANAGER_BLOCK_TOKENS,
        BLOCKS=blocks,
        SENTINEL_INIT=sentinel_init,
        SENTINEL_LOCAL=sentinel_local,
        num_warps=4,
    )
    return out
