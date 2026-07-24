# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _build_rotated_dcp_peer_block_table_kernel(
    gathered_block_tables_ptr,
    out_ptr,
    num_requests: tl.constexpr,
    max_owner_pages: tl.constexpr,
    dcp_size: tl.constexpr,
    local_rank: tl.constexpr,
    peer_block_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
    gathered_stride0,
    gathered_stride1,
    gathered_stride2,
    out_stride0,
    out_stride1,
):
    request = tl.program_id(0)
    logical_page = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    max_logical_pages = dcp_size * max_owner_pages
    page_mask = logical_page < max_logical_pages

    owner = logical_page % dcp_size
    owner_local_page = logical_page // dcp_size
    owner_i64 = owner.to(tl.int64)
    request_i64 = request.to(tl.int64)
    owner_local_page_i64 = owner_local_page.to(tl.int64)
    physical_block = tl.load(
        gathered_block_tables_ptr
        + owner_i64 * gathered_stride0
        + request_i64 * gathered_stride1
        + owner_local_page_i64 * gathered_stride2,
        mask=page_mask,
        other=-1,
    )

    # The rank-local VMM alias rotates segments so this rank is segment zero.
    # Use int64 before multiplying by the padded peer stride: VMM allocations
    # can make the intermediate address larger than signed int32 even though
    # such an entry must fail closed in the int32 block table.
    rotated_owner = (owner + dcp_size - local_rank) % dcp_size
    peer_block = rotated_owner.to(tl.int64) * peer_block_stride + physical_block.to(
        tl.int64
    )
    valid = (
        page_mask
        & (physical_block >= 0)
        & (physical_block < peer_block_stride)
        & (peer_block >= 0)
        & (peer_block <= 2147483647)
    )
    tl.store(
        out_ptr + request * out_stride0 + logical_page * out_stride1,
        tl.where(valid, peer_block, -1).to(tl.int32),
        mask=page_mask,
    )


def build_rotated_dcp_peer_block_table(
    gathered_block_tables: torch.Tensor,
    *,
    local_rank: int,
    peer_block_stride: int,
    cp_kv_cache_interleave_size: int,
    block_size: int,
    BLOCK_N: int = 128,
) -> torch.Tensor:
    """Build a logical page table over a rank-local rotated DCP peer view.

    ``gathered_block_tables`` has shape
    ``[dcp_size, num_requests, max_owner_pages]``. Entry ``[owner, req, p]``
    is the physical block assigned to owner-local logical page ``p`` for that
    request. This helper supports the page-sharded layout where one interleave
    is exactly one KV block:

    .. code-block:: text

        owner(logical_page) = logical_page % dcp_size
        owner_page          = logical_page // dcp_size
        output_block        = (
            (owner - local_rank) % dcp_size
        ) * peer_block_stride + physical_block

    The result has shape ``[num_requests, dcp_size * max_owner_pages]`` and can
    be passed to an attention kernel reading a rank-local rotated VMM alias.
    Negative/padded entries, physical blocks outside their owner's padded
    segment, and peer block IDs outside signed int32 map to ``-1``.

    Inputs may be strided, but must be CUDA int32. Address arithmetic in the
    Triton kernel is int64 even though the output block table remains int32.
    """
    if gathered_block_tables.dtype != torch.int32:
        raise TypeError(
            "gathered_block_tables must have dtype int32, got "
            f"{gathered_block_tables.dtype}."
        )
    if gathered_block_tables.ndim != 3:
        raise ValueError(
            "gathered_block_tables must have shape "
            "[dcp_size, num_requests, max_owner_pages], got "
            f"{tuple(gathered_block_tables.shape)}."
        )
    dcp_size, num_requests, max_owner_pages = gathered_block_tables.shape
    if dcp_size < 1:
        raise ValueError("gathered_block_tables must contain at least one DCP owner.")
    if not 0 <= local_rank < dcp_size:
        raise ValueError(f"local_rank must be in [0, {dcp_size}), got {local_rank}.")
    if peer_block_stride < 1:
        raise ValueError(
            f"peer_block_stride must be positive, got {peer_block_stride}."
        )
    if peer_block_stride > torch.iinfo(torch.int64).max // dcp_size:
        raise ValueError("The rotated peer block-table address space exceeds int64.")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if cp_kv_cache_interleave_size != block_size:
        raise ValueError(
            "Page-sharded DCP peer tables require "
            "cp_kv_cache_interleave_size == block_size, got "
            f"{cp_kv_cache_interleave_size} and {block_size}."
        )
    if BLOCK_N < 1 or BLOCK_N & (BLOCK_N - 1):
        raise ValueError(f"BLOCK_N must be a positive power of two, got {BLOCK_N}.")
    if gathered_block_tables.device.type != "cuda":
        raise ValueError("Rotated DCP peer block-table conversion requires CUDA.")

    max_logical_pages = dcp_size * max_owner_pages
    out = torch.empty(
        (num_requests, max_logical_pages),
        dtype=torch.int32,
        device=gathered_block_tables.device,
    )
    if num_requests > 0 and max_logical_pages > 0:
        _build_rotated_dcp_peer_block_table_kernel[
            (num_requests, triton.cdiv(max_logical_pages, BLOCK_N))
        ](
            gathered_block_tables,
            out,
            num_requests,
            max_owner_pages,
            dcp_size,
            local_rank,
            peer_block_stride,
            BLOCK_N,
            gathered_block_tables.stride(0),
            gathered_block_tables.stride(1),
            gathered_block_tables.stride(2),
            out.stride(0),
            out.stride(1),
        )
    return out


@triton.jit
def _convert_global_indices_to_dcp_peer_slots_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    valid_count_ptr,
    num_requests: tl.constexpr,
    max_num_blocks_per_req: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COUNT_VALID: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    PEER_BLOCK_STRIDE: tl.constexpr,
    req_stride,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = column < NUM_TOPK_TOKENS

    req = tl.load(req_id_ptr + token_id * req_stride)
    token = tl.load(
        token_indices_ptr + token_id * ti_stride0 + column * ti_stride1,
        mask=column_mask,
        other=-1,
    )
    owner = (token // DCP_INTERLEAVE) % DCP_SIZE
    local = (
        token // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + token % DCP_INTERLEAVE
    logical_block = local // BLOCK_SIZE
    block_offset = local % BLOCK_SIZE

    valid = (
        column_mask
        & (token >= 0)
        & (req >= 0)
        & (req < num_requests)
        & (logical_block >= 0)
        & (logical_block < max_num_blocks_per_req)
    )
    physical_block = tl.load(
        block_table_ptr + req * bt_stride0 + logical_block * bt_stride1,
        mask=valid,
        other=-1,
    )
    valid &= (physical_block >= 0) & (physical_block < PEER_BLOCK_STRIDE)
    peer_slot = (owner * PEER_BLOCK_STRIDE + physical_block) * BLOCK_SIZE + block_offset
    output = tl.where(valid, peer_slot, -1)
    tl.store(
        out_ptr + token_id * out_stride0 + column * out_stride1,
        output,
        mask=column_mask,
    )

    if COUNT_VALID:
        tl.atomic_add(
            valid_count_ptr + token_id,
            tl.sum(valid.to(tl.int32)),
        )


def convert_global_indices_to_dcp_peer_slots(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    blocks_per_peer: int,
    cp_kv_cache_interleave_size: int = 1,
    block_size: int = 64,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Map global sparse token indices into a rank-major DCP peer view.

    DCP assigns groups of ``cp_kv_cache_interleave_size`` consecutive tokens
    round-robin across ranks. Each owner stores its tokens de-interleaved in a
    local logical sequence. ``block_table`` maps that local logical sequence to
    physical blocks in one peer's allocation.

    The peer view is laid out as ``[dcp_rank, physical_block, block_offset]``.
    ``blocks_per_peer`` is its rank stride in physical blocks (including any
    VMM padding), so a valid global token index ``t`` maps as follows::

        owner = (t // interleave) % dcp_size
        local = (t // (dcp_size * interleave)) * interleave + t % interleave
        physical_block = block_table[request, local // block_size]
        peer_slot = (
            owner * blocks_per_peer + physical_block
        ) * block_size + local % block_size

    Invalid token indices, out-of-range requests or logical blocks, and invalid
    physical block-table entries map to ``-1``. The returned tensor has the same
    shape and ``int32`` dtype as ``token_indices``. When
    ``return_valid_counts`` is true, the kernel also returns the number of valid
    peer slots in each row.

    The conversion is one Triton pass. It accepts strided inputs directly.
    Callers may supply persistent ``out`` and ``valid_counts_out`` buffers to
    reuse the translated slots across layers. ``valid_counts_out`` is cleared
    before the kernel because each column tile atomically contributes its count.
    """
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}.")
    if blocks_per_peer < 1:
        raise ValueError(f"blocks_per_peer must be positive, got {blocks_per_peer}.")
    if cp_kv_cache_interleave_size < 1:
        raise ValueError(
            "cp_kv_cache_interleave_size must be positive, got "
            f"{cp_kv_cache_interleave_size}."
        )
    if block_size < cp_kv_cache_interleave_size or (
        block_size % cp_kv_cache_interleave_size != 0
    ):
        raise ValueError(
            f"block_size ({block_size}) must be greater than or equal to and "
            "divisible by cp_kv_cache_interleave_size "
            f"({cp_kv_cache_interleave_size})."
        )
    if dcp_size * blocks_per_peer * block_size > torch.iinfo(torch.int32).max:
        raise ValueError("Rank-major peer slots do not fit in int32.")
    if BLOCK_N < 1 or BLOCK_N & (BLOCK_N - 1):
        raise ValueError(f"BLOCK_N must be a positive power of two, got {BLOCK_N}.")
    if req_id.dtype != torch.int32:
        raise TypeError(f"req_id must have dtype int32, got {req_id.dtype}.")
    if block_table.dtype != torch.int32:
        raise TypeError(f"block_table must have dtype int32, got {block_table.dtype}.")
    if token_indices.dtype != torch.int32:
        raise TypeError(
            f"token_indices must have dtype int32, got {token_indices.dtype}."
        )
    if req_id.ndim != 1:
        raise ValueError(f"req_id must be 1D, got shape {tuple(req_id.shape)}.")
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be 2D, got shape {tuple(block_table.shape)}."
        )
    if token_indices.ndim != 2:
        raise ValueError(
            f"token_indices must be 2D, got shape {tuple(token_indices.shape)}."
        )
    if req_id.shape[0] != token_indices.shape[0]:
        raise ValueError(
            f"req_id ({req_id.shape[0]}) and token_indices "
            f"({token_indices.shape[0]}) must have the same row count."
        )
    if block_table.shape[0] == 0 or block_table.shape[1] == 0:
        raise ValueError("block_table must have at least one row and one column.")
    if token_indices.shape[1] == 0:
        raise ValueError("token_indices must contain at least one column.")
    if not (req_id.device == block_table.device == token_indices.device):
        raise ValueError("req_id, block_table, and token_indices must share a device.")
    if token_indices.device.type != "cuda":
        raise ValueError("DCP peer-slot conversion requires CUDA tensors.")

    num_tokens, num_topk_tokens = token_indices.shape
    if out is None:
        out = torch.empty(
            token_indices.shape,
            dtype=torch.int32,
            device=token_indices.device,
        )
    elif (
        out.shape != token_indices.shape
        or out.dtype != torch.int32
        or out.device != token_indices.device
    ):
        raise ValueError(
            "out must match token_indices shape, dtype, and device; got "
            f"shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}."
        )
    elif out.numel() > 0 and out.data_ptr() == token_indices.data_ptr():
        raise ValueError("out must not alias token_indices.")

    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        if valid_counts_out is None:
            valid_counts = torch.zeros(
                num_tokens,
                dtype=torch.int32,
                device=token_indices.device,
            )
        elif (
            valid_counts_out.shape != (num_tokens,)
            or valid_counts_out.dtype != torch.int32
            or valid_counts_out.device != token_indices.device
        ):
            raise ValueError(
                "valid_counts_out must be a one-dimensional int32 tensor with "
                f"{num_tokens} entries on {token_indices.device}; got "
                f"shape={tuple(valid_counts_out.shape)}, "
                f"dtype={valid_counts_out.dtype}, device={valid_counts_out.device}."
            )
        else:
            valid_counts = valid_counts_out
            valid_counts.zero_()
    elif valid_counts_out is not None:
        raise ValueError("valid_counts_out requires return_valid_counts=True.")

    if num_tokens > 0:
        _convert_global_indices_to_dcp_peer_slots_kernel[
            (num_tokens, triton.cdiv(num_topk_tokens, BLOCK_N))
        ](
            req_id,
            block_table,
            token_indices,
            out,
            valid_counts,
            block_table.shape[0],
            block_table.shape[1],
            num_topk_tokens,
            block_size,
            BLOCK_N,
            return_valid_counts,
            dcp_size,
            cp_kv_cache_interleave_size,
            blocks_per_peer,
            req_id.stride(0),
            block_table.stride(0),
            block_table.stride(1),
            token_indices.stride(0),
            token_indices.stride(1),
            out.stride(0),
            out.stride(1),
        )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


@triton.jit
def _filter_peer_slots_to_owner_local_kernel(
    peer_slots_ptr,
    local_slots_ptr,
    valid_counts_ptr,
    peer_slots_stride0,
    peer_slots_stride1,
    local_slots_stride0,
    local_slots_stride1,
    NUM_TOPK_TOKENS: tl.constexpr,
    OWNER_SLOT_START: tl.constexpr,
    OWNER_SLOT_STOP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    column_mask = columns < NUM_TOPK_TOKENS
    slots = tl.load(
        peer_slots_ptr + row * peer_slots_stride0 + columns * peer_slots_stride1,
        mask=column_mask,
        other=-1,
    )
    valid = column_mask & (slots >= OWNER_SLOT_START) & (slots < OWNER_SLOT_STOP)
    valid_int = valid.to(tl.int32)
    compacted_columns = tl.cumsum(valid_int) - valid_int
    tl.store(
        local_slots_ptr
        + row * local_slots_stride0
        + compacted_columns * local_slots_stride1,
        slots - OWNER_SLOT_START,
        mask=valid,
    )
    tl.store(valid_counts_ptr + row, tl.sum(valid_int, axis=0))


def filter_peer_slots_to_owner_local(
    peer_slots: torch.Tensor,
    *,
    owner_rank: int,
    dcp_world_size: int,
    blocks_per_peer: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter rank-major peer slots to one owner's local cache.

    One Triton program handles a complete selected-token row, so prefix
    compaction is stable and requires no atomics. The result is suitable for
    FlashInfer's sparse MLA block-table input: valid local slots occupy
    ``[0, valid_count)`` and the tail remains ``-1``.
    """
    if peer_slots.dtype != torch.int32 or peer_slots.ndim != 2:
        raise ValueError("Owner-compute peer slots must be a 2D int32 tensor.")
    if peer_slots.device.type != "cuda":
        raise ValueError("Owner-compute peer-slot filtering requires CUDA.")
    if dcp_world_size <= 1 or not 0 <= owner_rank < dcp_world_size:
        raise ValueError("Owner-compute filtering requires a valid DCP owner.")
    if blocks_per_peer <= 0 or block_size <= 0:
        raise ValueError("Owner-compute peer stride and block size must be positive.")
    if peer_slots.shape[1] == 0:
        raise ValueError("Owner-compute selected-slot rows cannot be empty.")

    rows, topk = peer_slots.shape
    block_n = triton.next_power_of_2(topk)
    if block_n > 65536:
        raise ValueError(
            f"Owner-compute top-k is too large for one Triton row: {topk}."
        )
    slots_per_peer = blocks_per_peer * block_size
    owner_start = owner_rank * slots_per_peer
    owner_stop = owner_start + slots_per_peer
    local_slots = torch.full_like(peer_slots, -1)
    valid_counts = torch.empty(
        rows,
        dtype=torch.int32,
        device=peer_slots.device,
    )
    if rows > 0:
        _filter_peer_slots_to_owner_local_kernel[(rows,)](
            peer_slots,
            local_slots,
            valid_counts,
            peer_slots.stride(0),
            peer_slots.stride(1),
            local_slots.stride(0),
            local_slots.stride(1),
            topk,
            owner_start,
            owner_stop,
            block_n,
            num_warps=8,
        )
    return local_slots, valid_counts


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
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
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
    # Dense prefill workspaces are addressed by the original global token ID,
    # so owner-local paged-cache bounds and ownership do not apply to them.
    # Applying these checks before the workspace override silently discarded
    # valid long-context selections beyond one DCP owner's BlockTable width.
    is_invalid_tok |= (~is_prefill) & (~valid_block | is_remote)
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill & ~is_remote, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

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
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
        dest = base + local_offset
        out_ptr_dest = out_ptr + token_id * out_stride0 + dest * out_stride1
        tl.store(out_ptr_dest, out_val, mask=is_valid == 1)
    else:
        # Store results in place (input column == output column).
        out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
        tl.store(out_ptr_ij, out_val)

        # Count valid indices in this tile and atomically add to row total
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
    out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
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
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    # `out`/`valid_counts_out` allow writing into a stable pre-allocated buffer
    # (used by the shared-physical-index cache to stay cudagraph-safe).
    out = torch.empty_like(token_indices_c) if out is None else out

    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        if valid_counts_out is None:
            valid_counts = torch.zeros(
                num_tokens, dtype=torch.int32, device=token_indices.device
            )
        else:
            valid_counts = valid_counts_out
            valid_counts.zero_()

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
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
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
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # The compaction uses the valid-count buffer as an atomic slot allocator, so
    # it requires counting. Pre-fill out with -1 so the unwritten tail stays -1.
    count_valid = return_valid_counts or compact_valid_to_front
    if compact_valid_to_front:
        out = torch.full_like(token_indices_c, -1)
    else:
        out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if count_valid:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

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
        BLOCK_N,
        False,  # HAS_PREFILL
        count_valid,
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
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


# --- Physical-index shadow (DSA index_topk_freq > 1) -------------------------
# GLM-5.2 uses index_topk_freq=4: only ~1/4 of layers write a fresh top-k into
# the single shared topk_indices_buffer; the rest reuse it. Since block_table
# and req_id_per_token are constant within a decode step, the physical indices
# (block_table lookup of the logical top-k) are identical across a freq-group.
# Fresh layers convert once into a STABLE shadow of the logical buffer and
# skip layers read it -- eliminating ~3/4 of the per-layer convert kernels.
#
# Invariant: shadow[j] == convert(logical[j]) row-wise. Whoever re-arranges
# the logical buffer (e.g. the MTP draft compact between the multi-token
# step-0 layout and the single-token steps-1+ layout) must apply the same
# gather to the shadow.
#
# Shadows are registered ONLY at buffer creation time (model init, never
# inside cudagraph capture) via register_phys_shadow; every other access is
# the read-only phys_shadow lookup, so addresses are stable across graph
# replays and no allocation can happen during capture.
import weakref as _weakref  # noqa: E402

_PHYS_SHADOWS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def register_phys_shadow(topk_buf: torch.Tensor) -> None:
    """Allocate the (physical-index, valid-count) shadow for a logical top-k
    buffer. Call once where the buffer is created. The finalizer drops the
    entry with the owning buffer, so a recycled id can never alias a stale
    shadow."""
    key = id(topk_buf)
    if key not in _PHYS_SHADOWS:
        _PHYS_SHADOWS[key] = (
            torch.empty_like(topk_buf),
            torch.empty(topk_buf.shape[0], dtype=torch.int32, device=topk_buf.device),
        )
        _weakref.finalize(topk_buf, _PHYS_SHADOWS.pop, key, None)


def phys_shadow(
    topk_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The registered shadow for this buffer, or None if it was never
    registered. Never allocates; capture-safe."""
    return _PHYS_SHADOWS.get(id(topk_buf))
