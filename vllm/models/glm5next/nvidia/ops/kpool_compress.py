# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kpool (key-pooling) Triton kernels for the sparse-attention indexer.

The cache stores POOLS (1 entry per ``pool_size`` consecutive tokens) rather
than individual tokens. ``compress_ratio == pool_size`` on the kv_cache_spec
makes the metadata builder emit pool-granular slot_mapping / seq_lens /
cu_seq_lens / page_table for free; this file supplies the compress-write
kernel (replacing ``indexer_k_quant_and_cache``) and the pool-level topk
helpers (select pools -> expand to tokens -> append tail).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# The indexer head dim is fixed at 128 in the current GLM5Next config; the
# Hadamard rotation below is the hard-coded H128 transform.
INDEX_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Hadamard-128 rotation (ported verbatim from sglang)
# ---------------------------------------------------------------------------


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845  # 1/sqrt(128)


def _hadamard128_torch(x: torch.Tensor) -> torch.Tensor:
    """Reference / fallback Hadamard-128 on the last dim (must be 128)."""
    import math

    n = x.shape[-1]
    assert n == 128, f"_hadamard128 expects last dim 128, got {n}"
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32, device=x.device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / math.sqrt(n)
    return x @ h


# ---------------------------------------------------------------------------
# compute_pooled_write_locs : pool_id -> physical flat slot
# ---------------------------------------------------------------------------


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Map logical pooled-K ids to physical flat cache slots.

    ``pool_size`` consecutive tokens share one pool slot that lives at the
    *first* token page of each page-group. ``page_table_64`` maps token pages
    to physical block ids; we gather the block id of each pool's page-group
    and add the in-block pool offset.
    """
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    block_size = 64  # indexer cache page size (matches sglang hard-code)
    pool_page_group = torch.div(pool_ids, block_size, rounding_mode="floor")
    token_page_row = pool_page_group * pool_size
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * block_size + torch.remainder(
        pool_ids, block_size
    )


def build_pooled_page_table(
    page_table: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Build a pool-granular page table by taking every ``pool_size``-th
    token-page column (one pool maps to ``pool_size`` token pages).

    Uses gather (not strided slicing) so the result is always a fresh
    row-major tensor — some downstream kernels require stride(-1) == 1.
    """
    block_size = page_table.shape[-1]
    assert block_size % pool_size == 0, (
        f"pool_size ({pool_size}) must divide page columns ({block_size})"
    )
    idx = torch.arange(0, block_size, pool_size, device=page_table.device)
    return page_table[..., idx].contiguous()


# ---------------------------------------------------------------------------
# kpool_softmax_rotate_write_cache : the fused compress-write kernel
# ---------------------------------------------------------------------------


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
    Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        max_score = tl.maximum(max_score, score)

    # --- Pass 2: softmax-weighted sum of K ---
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)

    # Hadamard-128 rotation (spreads energy for uniform fp8 quant error).
    x = _hadamard128(x)

    # --- per-vector absmax fp8 quant ---
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )
        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)

    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            quantized,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


def kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Compress ``pool_size`` tokens into one fp8 K and write at ``loc``.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        slot_k: ``[n_pools, pool_size, head_dim]`` bf16 — raw per-token K.
        slot_score: ``[n_pools, pool_size, head_dim]`` — per-token gate score.
        ape: ``[pool_size, head_dim]`` fp32 — per-slot position bias.
        loc: ``[n_pools]`` int64 — flat physical slot per pool.
    """
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == head_dim
    assert slot_k.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert write_cache or return_compressed

    page_size = kv_cache.shape[1]
    buf = kv_cache
    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        write_mask = write_mask.contiguous()
        has_write_mask = True
        assert not return_compressed

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    # bytes per page (last dim of kv_cache) viewed as uint8
    buf_numel_per_page = buf.stride(0)
    s_offset_nbytes_in_page = page_size * head_dim

    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], head_dim),
            dtype=torch.float8_e4m3fn,
            device=slot_k.device,
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32

    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    if return_compressed:
        return compressed_k, compressed_scale
    return None


# ---------------------------------------------------------------------------
# kpool_decode_update_and_maybe_write_cache_batched : decode step
# Append each request's verify tokens to its per-request tail ring; when a pool
# fills (pos % pool_size == pool_size-1), compress and write at the pool slot.
# One launch over [num_requests, next_n]; the kernel iterates each request's
# tokens in position order (see the kernel docstring for the completion
# read-after-stash dependency). Plain decode collapses to next_n == 1.
#
# vLLM simplification vs sglang: compress_ratio makes slot_mapping hand us the
# pool slot directly at pool completion, so the write loc == cache_loc. No
# block_table recomputation needed.
# ---------------------------------------------------------------------------


@triton.jit
def _kpool_decode_update_batched_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,  # runtime token count per request (no .item() needed)
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per request; iterates its NEXT_N verify tokens in order.

    Replaces the caller's per-token sequential launch loop. The intra-request
    iteration MUST stay in position order: a pool-completion at token t* reads
    the tail-ring slots that tokens t < t* (same request) just stashed in this
    same invocation. ``tl.range`` iterates sequentially within the program, so
    those stashes are visible to the later completion read. Cross-request
    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE
    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one
    completion can occur per request per call, but the ordered loop is correct
    for any NEXT_N.
    """
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for t in tl.range(0, NEXT_N):
        idx = req * NEXT_N + t
        cache_loc = tl.load(slot_mapping_ptr + idx)
        pos = tl.load(positions_ptr + idx)
        safe_pos = tl.maximum(pos, 0)
        pos_valid = (cache_loc >= 0) & (pos >= 0)

        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % POOL_SIZE

        # Derive the tail block from THIS token's tail_slot (the request's block
        # is constant across a pool, but a padded / invalid entry carries a
        # negative sentinel -- reading it from token 0 would poison every
        # token's base address). Clamp so an invalid entry can never form an
        # out-of-bounds base; the accesses below are gated on pos_valid anyway.
        tail_slot = tl.load(tail_slot_mapping_ptr + idx)
        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE
        block_base = block * TAIL_BLOCK_ELEMS

        key = tl.load(
            key_ptr + req * key_stride_b + t * key_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score_current = tl.load(
            slot_score_ptr + req * ss_stride_b + t * ss_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if pos_valid & (slot == POOL_SIZE - 1):
            pool_logical_start = safe_pos - slot

            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                prob = tl.exp(score - max_score)
                denom += prob
                k_buf = tl.load(
                    tail_kv_ptr + block_base + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                k = tl.where(is_current, key, k_buf)
                acc += k * prob

            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

            fp8_max = 448.0
            fp8_max_inv = 1.0 / fp8_max
            absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
            else:
                scale = absmax * fp8_max_inv
            quantized = tl.minimum(tl.maximum(x / scale, -fp8_max), fp8_max)

            loc = cache_loc.to(tl.int64)
            loc_page_index = loc // PAGE_SIZE
            loc_token_offset_in_page = loc % PAGE_SIZE
            out_k_offsets = (
                loc_page_index * BUF_NUMEL_PER_PAGE
                + loc_token_offset_in_page * HEAD_DIM
                + offs
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale)

        # Stash the current token AFTER any completion read so the completion
        # uses prior stashes (and the current token's own key/score via
        # is_current), then leaves this token for future pools. Order matches
        # the per-token kernel: completion read first, stash second.
        update_mask = dim_mask & pos_valid
        tl.store(
            tail_kv_ptr + block_base + phys_slot * HEAD_DIM + offs,
            key,
            mask=update_mask,
        )
        tl.store(
            tail_kv_ptr + block_base + KPOOL_HEAD + phys_slot * HEAD_DIM + offs,
            score_current,
            mask=update_mask,
        )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Batched decode-step kpool update for spec verify (``next_n > 1``).

    One launch replaces the caller's per-token loop. Inputs are grouped per
    request: ``[num_requests, next_n, ...]``. Each program handles one
    request's ``next_n`` tokens in position order (see the kernel docstring for
    why ordering is required for pool-completion correctness).

    Plain decode (``next_n == 1``) is handled here too — the kernel collapses
    to a single-iteration loop.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        tail_kv_cache: paged tail cache ``[num_blocks, 2, pool_size, head_dim]``
            bf16 (K at half 0, gate score at half 1).
        tail_slot_mapping: ``[num_requests, next_n]`` int32.
        key / slot_score: ``[num_requests, next_n, head_dim]`` bf16.
        ape: ``[pool_size, head_dim]`` fp32.
        slot_mapping / positions: ``[num_requests, next_n]`` int32.
    """
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.shape[2] == pool_size
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    buf = kv_cache
    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    # The kernel indexes the int tensors as ``req * next_n + t`` (row-major),
    # so they must be contiguous. Callers pass either a view of a contiguous
    # slice or a freshly scattered tensor, making these no-ops; the calls guard
    # against a future caller handing over a strided view.
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()

    _kpool_decode_update_batched_kernel[(num_requests,)](
        buf_fp8,
        buf_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf.stride(0),
        POOL_SIZE=pool_size,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


# ---------------------------------------------------------------------------
# Pool-level topk helpers: select pools -> expand to tokens -> append tail
# ---------------------------------------------------------------------------


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """Number of pools to select so that expanding yields ``topk`` tokens."""
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after expanded history tokens.

    ``index_kpool_always_select_tail`` keeps the (incomplete) trailing pool so
    the most recent tokens are always attended to.
    """
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    # tail tokens: [pool_len*pool_size, seq_len) for each row.
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start  # in [0, pool_size)

    cols = torch.arange(out_cols, device=topk_result.device)[None, :]
    history_len = n_cols
    is_history = cols < history_len
    tail_off = cols - history_len
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    # safe_hist must be per-row [rows, out_cols] so the gather reads each row's
    # OWN history. cols is [1, out_cols]; if used directly, gather (which does
    # NOT broadcast the index) would read only row 0 of topk_result, making every
    # query inherit row 0's history (empty for the first token) and lose all its
    # selected tokens — only the per-row tail would survive. This only manifests
    # for multi-row sparse PREFILL (decode has 1 row, so it reads its own row 0).
    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(topk_result, 1, safe_hist)

    tail_raw = tail_start[:, None] + tail_off
    tail_val = tail_raw.to(torch.int32)
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_val = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail_val = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    out = torch.where(is_history, history_val, -1)
    out = torch.where(is_tail, tail_val, out)
    return out


@triton.jit
def _expand_pools_and_append_tail_kernel(
    pool_ids_ptr,  # [rows, n_groups], int (any int dtype)
    seq_lens_ptr,  # [rows], int32 (token-granular seq_len)
    out_ptr,  # [rows, out_cols], int32
    topk,  # n_groups * pool_size
    out_cols,  # topk + pool_size - 1
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    pid_s0,
    out_s0,
):
    # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
    # single kernel. Each program writes one (row, column-tile) of the output.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < out_cols

    seq_len = tl.load(seq_lens_ptr + row)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

    # History region [0, topk): expand selected pool g = cols // POOL_SIZE.
    is_history = cols < topk
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
    hist_val = (pid * POOL_SIZE + o).to(tl.int32)
    hist_out = tl.where(pid >= 0, hist_val, -1)

    # Tail region [topk, out_cols): the request's trailing incomplete pool.
    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).to(tl.int32)
    tail_out = tl.where(is_tail, tail_val, -1)

    result = tl.where(is_history, hist_out, tail_out)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Fuse ``expand_pools_to_tokens`` + ``append_tail_to_topk`` (identity path).

    Produces the same ``[rows, topk + pool_size - 1]`` int32 output as calling
    the two functions in sequence when neither ``page_table`` nor
    ``topk_offsets`` is passed — the only path the GLM5Next indexer exercises.
    The kernel derives ``pool_len = seq_len // pool_size`` internally, so the
    caller no longer needs to precompute it. Replaces ~25 elementwise kernels
    with one Triton launch.
    """
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    BLOCK_COLS = 128
    n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
    _expand_pools_and_append_tail_kernel[(rows, n_tiles)](
        pool_ids,
        seq_lens,
        out,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=out.stride(0),
    )
    return out
