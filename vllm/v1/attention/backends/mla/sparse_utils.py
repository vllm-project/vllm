# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton


def flat_kv_row_view(
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim]
    block_size: int,
) -> tuple[torch.Tensor, int]:
    """Flat [row, head_dim] view of a paged cache and its physical rows per block.

    Token offset is  ``block_idx * block_stride_rows + offset_in_block``.
    When other layers' pages sit between consecutive blocks of this cache,
    ``block_stride_rows`` exceeds ``block_size``; those in-between rows are never
    indexed (`triton_convert_req_index_to_global_index` ensures this).
    """
    num_blocks, _, head_dim = kv_cache.shape
    assert kv_cache.stride(0) % head_dim == 0, (
        "block stride is not a whole number of rows; flat row indexing would "
        "silently misaddress"
    )
    block_stride_rows = kv_cache.stride(0) // head_dim
    num_rows = (num_blocks - 1) * block_stride_rows + block_size
    rows = kv_cache.as_strided((num_rows, head_dim), (head_dim, 1))
    return rows, block_stride_rows


def topology_witness_indices(
    context_lens: torch.Tensor,  # int32 [num_rows]
    learned_indices: torch.Tensor,  # int32 [num_rows, topk]
    num_segments: int,
) -> torch.Tensor:
    """Structural witnesses for context segments the learned prefix misses.

    Sparse MLA top-k is a density selector: it concentrates wherever the indexer
    scores are high and offers no guarantee that any given stretch of context is
    represented at all. This splits each request's context into equal segments
    and returns one witness token per segment the learned row does not already
    cover, which bounds the uncovered run at ``context_len / num_segments`` tokens.

    Occupancy is marked from the whole learned row, not just the prefix the
    merge retains. A tail slot the merge happens not to overwrite still survives
    into the output, so a witness matching it would duplicate a token and double
    its softmax weight.

    Witnesses are segment left edges rounded up. That rounding is what makes the
    de-duplication free: ``segment_of(witness(s)) == s`` exactly when the segment
    count does not exceed the context length, so a witness emitted for an
    unoccupied segment cannot collide with the row that marked occupancy. The
    per-row segment count is capped at the context length to hold that condition.

    Returns:
        ``torch.int32`` ``[num_rows, num_segments]`` of request-local token
        offsets, ``-1`` where the segment is already covered or does not exist.
        Every non-negative entry is distinct within its row, absent from
        ``learned_indices``, and less than that row's ``seq_len``.
    """
    if learned_indices.ndim != 2:
        raise ValueError("learned_indices must be 2D [num_rows, topk]")
    if context_lens.ndim != 1 or context_lens.shape[0] != learned_indices.shape[0]:
        raise ValueError("context_lens must be 1D and cover every learned row")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")

    device = learned_indices.device
    num_rows = learned_indices.shape[0]
    raw_lens = context_lens.to(torch.int64).unsqueeze(1)
    lens = raw_lens.clamp_min(1)
    segments = torch.arange(num_segments, device=device, dtype=torch.int64)

    # Capping at seq_len keeps segment_of(witness(s)) == s exact.
    effective = lens.clamp_max(num_segments)
    in_range = (segments.unsqueeze(0) < effective) & (raw_lens > 0)
    edges = -(-segments.unsqueeze(0) * lens // effective)

    # A learned entry occupies the segment it falls in; padding is routed to a
    # scratch column so it cannot mark anything occupied.
    learned = learned_indices.to(torch.int64)
    learned_segment = torch.where(
        learned >= 0,
        (learned.clamp_min(0) * effective // lens).clamp_(0, num_segments - 1),
        torch.full_like(learned, num_segments),
    )
    occupied = torch.zeros(
        (num_rows, num_segments + 1), dtype=torch.bool, device=device
    )
    occupied.scatter_(1, learned_segment, True)

    keep = in_range & ~occupied[:, :num_segments]
    return torch.where(keep, edges, torch.full_like(edges, -1)).to(torch.int32)


def merge_topology_witnesses(
    learned_indices: torch.Tensor,  # int32 [num_rows, topk]
    witnesses: torch.Tensor,  # int32 [num_rows, num_segments]
    learned_keep: int,
    max_replacements: int,
) -> torch.Tensor:
    """Overwrite a bounded learned tail with structural witnesses.

    Columns ``[0, learned_keep)`` survive untouched. Witnesses are already
    distinct and already disjoint from the whole learned row by construction in
    :func:`topology_witness_indices`, so this compacts them into the tail with a
    prefix sum and never re-scans the learned row.
    """
    if learned_indices.shape[0] != witnesses.shape[0]:
        raise ValueError("learned_indices and witnesses rows must match")
    if not 0 <= learned_keep <= learned_indices.shape[1]:
        raise ValueError("learned_keep must fit within the learned top-k dimension")
    if max_replacements < 0:
        raise ValueError("max_replacements must be non-negative")

    topk = learned_indices.shape[1]
    budget = min(max_replacements, topk - learned_keep)
    if budget == 0:
        return learned_indices.clone()

    present = witnesses >= 0
    slot = present.cumsum(1) - 1
    write = present & (slot < budget)

    out = learned_indices.clone()
    # Masked-off lanes are parked on a scratch column that is sliced away, which
    # keeps this a single scatter with no index clamping games.
    column = torch.where(write, learned_keep + slot, torch.full_like(slot, topk))
    padded = torch.cat([out, out.new_empty((out.shape[0], 1))], dim=1)
    padded.scatter_(1, column, witnesses)
    return padded[:, :topk].contiguous()


MAX_TOPOLOGY_SEGMENTS = 64
"""Segment occupancy is carried as an int64 bitmask, one bit per segment."""


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _topology_witness_merge_kernel(
    learned_ptr,
    context_lens_ptr,
    out_ptr,
    topk,
    learned_keep,
    budget,
    learned_stride0,
    learned_stride1,
    out_stride0,
    out_stride1,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    raw_len = tl.load(context_lens_ptr + row).to(tl.int64)
    seq_len = tl.maximum(raw_len, 1)
    effective = tl.minimum(seq_len, NUM_SEGMENTS)

    # One pass over the learned row copies it out and folds every entry's
    # segment into a bitmask. Reducing to a mask instead of a [segment, token]
    # comparison tile is what keeps this O(topk) rather than O(topk * segments).
    occupied_mask = tl.zeros([1], dtype=tl.int64)
    for start in tl.range(0, topk, BLOCK_K):
        k_offsets = start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < topk
        learned = tl.load(
            learned_ptr + row * learned_stride0 + k_offsets * learned_stride1,
            mask=k_mask,
            other=-1,
        )
        tl.store(
            out_ptr + row * out_stride0 + k_offsets * out_stride1,
            learned,
            mask=k_mask,
        )
        token = learned.to(tl.int64)
        # An index at or past seq_len is not a legal top-k output, but clamping
        # keeps the shift below in range instead of leaving it undefined.
        segment = tl.minimum((token * effective) // seq_len, NUM_SEGMENTS - 1)
        one = tl.full([BLOCK_K], 1, dtype=tl.int64)
        bit = tl.where(k_mask & (token >= 0), one << segment, 0)
        occupied_mask |= tl.reduce(bit, 0, _or_combine)

    segments = tl.arange(0, BLOCK_SEGMENTS).to(tl.int64)
    # Rounding the segment edge up is what makes the witness land back in its own
    # segment, so an unoccupied segment yields a token the learned row cannot
    # already hold.
    edges = (segments * seq_len + effective - 1) // effective
    keep = (
        (segments < effective)
        & (raw_len > 0)
        & (((occupied_mask >> segments) & 1) == 0)
    )

    is_valid = keep.to(tl.int32)
    slot = tl.cumsum(is_valid) - is_valid
    write = (is_valid == 1) & (slot < budget)

    # The learned row is already stored above and these lanes overwrite part of
    # it from different threads, so the barrier is what orders the two stores.
    tl.debug_barrier()
    tl.store(
        out_ptr + row * out_stride0 + (learned_keep + slot) * out_stride1,
        edges.to(tl.int32),
        mask=write,
    )


def apply_topology_witnesses(
    learned_indices: torch.Tensor,  # int32 [num_rows, topk]
    context_lens: torch.Tensor,  # int32 [num_rows]
    learned_keep: int,
    num_segments: int,
    max_replacements: int,
) -> torch.Tensor:
    """Fused witness generation and bounded tail replacement.

    Equivalent to :func:`merge_topology_witnesses` applied to
    :func:`topology_witness_indices`, in a single pass over the learned row.
    Falls back to that pair on CPU, which is what the two are for.
    """
    if learned_indices.ndim != 2:
        raise ValueError("learned_indices must be 2D [num_rows, topk]")
    if context_lens.ndim != 1 or context_lens.shape[0] != learned_indices.shape[0]:
        raise ValueError("context_lens must be 1D and cover every learned row")
    if not 0 < num_segments <= MAX_TOPOLOGY_SEGMENTS:
        raise ValueError(
            f"num_segments must be in [1, {MAX_TOPOLOGY_SEGMENTS}]; occupancy is "
            "carried as an int64 bitmask"
        )
    if not 0 <= learned_keep <= learned_indices.shape[1]:
        raise ValueError("learned_keep must fit within the learned top-k dimension")
    if max_replacements < 0:
        raise ValueError("max_replacements must be non-negative")

    topk = learned_indices.shape[1]
    budget = min(max_replacements, topk - learned_keep)
    if budget == 0 or learned_indices.shape[0] == 0:
        return learned_indices.clone()

    if learned_indices.device.type != "cuda":
        witnesses = topology_witness_indices(
            context_lens, learned_indices, num_segments
        )
        return merge_topology_witnesses(
            learned_indices, witnesses, learned_keep, max_replacements
        )

    learned_c = learned_indices.contiguous()
    context_lens_c = context_lens.to(torch.int32).contiguous()
    out = torch.empty_like(learned_c)
    _topology_witness_merge_kernel[(learned_c.shape[0],)](
        learned_c,
        context_lens_c,
        out,
        topk,
        learned_keep,
        budget,
        learned_c.stride(0),
        learned_c.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_SEGMENTS=num_segments,
        BLOCK_SEGMENTS=MAX_TOPOLOGY_SEGMENTS,
        BLOCK_K=1024,
        num_warps=4,
    )
    return out


def scatter_topology_witnesses_(
    topk_indices: torch.Tensor,  # int32 [num_rows, topk]
    context_lens: torch.Tensor,  # int32 [num_rows]
    num_segments: int,
) -> None:
    """Replace the last ``num_segments`` slots of each top-k row, in place.

    The sparse MLA indexer hands its top-k buffer straight to the attention
    kernels, so the witnesses have to land in that same buffer.
    """
    topk = topk_indices.shape[1]
    learned_keep = max(0, topk - num_segments)
    topk_indices.copy_(
        apply_topology_witnesses(
            topk_indices,
            context_lens,
            learned_keep,
            num_segments,
            num_segments,
        )
    )


# Kernel with prefill workspace support and valid count tracking
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # BLOCK_N == NUM_TOPK_TOKENS: one program owns the row, so the valid count
    # is an in-register reduction and needs no atomic.
    SINGLE_TILE: tl.constexpr,
    # When set, scatter valid slots to a contiguous prefix [0, valid_count) using
    # valid_count_ptr as an atomic slot allocator (DCP filtering leaves interior
    # -1 gaps; the trtllm-gen sparse kernel reads the first valid_count entries).
    # Requires COUNT_VALID and an out buffer pre-filled with -1. Order within the
    # prefix is unspecified (only the selected set matters).
    COMPACT_TO_FRONT: tl.constexpr,
    # DCP de-interleave: with DCP_SIZE == 1 these are an exact no-op
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    # DCP de-interleave the global token id into this rank's local slot.
    # Tokens are interleaved in groups of DCP_INTERLEAVE across ranks. With
    # DCP_SIZE == 1 (and any interleave) owning_rank == 0 == DCP_RANK (never
    # remote) and local_idx == tok, so this reduces to the non-DCP path; with
    # DCP_INTERLEAVE == 1 it reduces to plain round-robin (tok % / // DCP_SIZE).
    owning_rank = (tok // DCP_INTERLEAVE) % DCP_SIZE
    is_remote = owning_rank != DCP_RANK
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE

    # Compute block id and in-block offset
    block_id = local_idx // BLOCK_SIZE
    inblock_off = local_idx % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block | is_remote
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill & ~is_remote, other=0)
    out_val = base * BLOCK_STRIDE_ROWS + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    if COMPACT_TO_FRONT:
        # Scatter valid slots to a contiguous prefix. A per-tile exclusive prefix
        # sum gives each valid lane a distinct local offset; one atomic add of the
        # tile's valid count reserves a contiguous base across racing tiles. The
        # out buffer is pre-filled with -1, so unwritten tail slots stay -1.
        # With no racing tiles the base is 0 and the allocator becomes a store.
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        if SINGLE_TILE:
            base = 0
            tl.store(valid_count_ptr + token_id, tile_valid_count)
        else:
            base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
        dest = base + local_offset
        out_ptr_dest = out_ptr + token_id * out_stride0 + dest * out_stride1
        tl.store(out_ptr_dest, out_val, mask=is_valid == 1)
    else:
        # Store results in place (input column == output column).
        out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
        tl.store(out_ptr_ij, out_val)

        # Accumulate the tile's valid count into the row total; a single tile's
        # reduction *is* the total.
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            if SINGLE_TILE:
                tl.store(valid_count_ptr + token_id, tile_valid_count)
            else:
                tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def _remap_tiling(
    NUM_TOPK_TOKENS: int, BLOCK_N: int, count_valid: bool
) -> tuple[bool, int, int, int]:
    """Pick the column tiling for the index remap kernel.

    Counting the valid slots per row is the only reason the column tiles have to
    talk to each other, so when counting give one program the whole row: the
    count becomes an in-register reduction plus a plain store, needing neither
    atomics nor a zero-initialized counter. The row is one ``tl.arange``, so this
    needs a power-of-two width; other top-k sizes stay tiled and atomic.

    Returns:
        (single_tile, block_n, tiles_per_row, num_warps)
    """
    single_tile = (
        count_valid and triton.next_power_of_2(NUM_TOPK_TOKENS) == NUM_TOPK_TOKENS
    )
    if single_tile:
        return True, NUM_TOPK_TOKENS, 1, 8
    return False, BLOCK_N, NUM_TOPK_TOKENS // BLOCK_N, 4


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    BLOCK_STRIDE_ROWS: int | None = None,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert req_id.shape[0] == token_indices.shape[0], (
        f"req_id ({req_id.shape[0]}) and token_indices ({token_indices.shape[0]}) "
        "must cover the same tokens; the grid is sized by req_id but the output "
        "is allocated like token_indices, so a longer req_id writes out of bounds"
    )
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]

    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, return_valid_counts
    )

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        # Zero-init only matters for the atomic accumulation path.
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_STRIDE_ROWS if BLOCK_STRIDE_ROWS is not None else BLOCK_SIZE,
        block_n,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        single_tile,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        # DCP disabled (no-op de-interleave)
        1,
        0,
        1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=num_warps,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


def triton_filter_and_convert_dcp_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int = 1,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Filter global per-request indices to this DCP rank's local slots.

    With ``compact_valid_to_front`` (default), the conversion kernel scatters
    this rank's owned slots to a contiguous prefix ``[0, valid_count)`` and
    leaves the rest ``-1``. DCP filtering marks non-owned slots ``-1`` and so
    creates interior gaps; the trtllm-gen sparse kernel reads the first
    ``valid_count`` entries of each row, so they must be a contiguous prefix.
    Compaction is fused into the kernel (atomic slot allocator) rather than a
    separate sort/gather pass. Prefix order is unspecified (only the set matters).
    """
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size
    # Interleave groups must align to KV blocks (globally enforced by
    # VllmConfig: block_size % cp_kv_cache_interleave_size == 0); assert the
    # local invariant so local_idx // BLOCK_SIZE never straddles a group.
    assert BLOCK_SIZE % cp_kv_cache_interleave_size == 0, (
        f"BLOCK_SIZE ({BLOCK_SIZE}) must be divisible by "
        f"cp_kv_cache_interleave_size ({cp_kv_cache_interleave_size})."
    )
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    if dcp_size == 1:
        return triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
            BLOCK_N=BLOCK_N,
            return_valid_counts=return_valid_counts,
        )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # The compaction uses the valid-count buffer as a slot allocator, so it
    # requires counting. Pre-fill out with -1 so the unwritten tail stays -1.
    count_valid = return_valid_counts or compact_valid_to_front

    # The compaction builds on the counting, so it shares the tiling.
    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, count_valid
    )

    if compact_valid_to_front:
        out = torch.full_like(token_indices_c, -1)
    else:
        out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if count_valid:
        # Zero-init only matters for the atomic accumulation path.
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    _convert_req_index_to_global_index_kernel[(num_tokens, tiles_per_row)](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        # No prefill workspace on the DCP decode path.
        None,
        None,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_SIZE,  # dense caches on the DCP path
        block_n,
        False,  # HAS_PREFILL
        count_valid,
        single_tile,
        compact_valid_to_front,
        dcp_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=num_warps,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
