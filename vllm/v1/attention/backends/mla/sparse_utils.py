# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class TopologyIndexConfig:
    """Opt-in shadow policy for sparse MLA logical token indices."""

    enabled: bool = False
    learned_fraction: float = 0.5
    max_segments: int = 8
    barrier_strength: float = 0.0
    diversity_strength: float = 0.0
    max_replacements: int | None = None


@dataclass(frozen=True)
class TopologyIndexResult:
    indices: torch.Tensor
    applied: bool
    fallback_reason: str | None
    learned_retained: int = 0
    structural_inserted: int = 0


def apply_topology_tail_index_policy(
    learned_indices: torch.Tensor,
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    config: TopologyIndexConfig = TopologyIndexConfig(),
) -> TopologyIndexResult:
    """Inject topology witnesses into a sparse MLA logical-token index.

    The coordinate contract is the same as sparse MLA ``topk_indices_buffer``:
    ``torch.int32`` with shape ``[num_tokens, topk]``, logical request-local
    token offsets, and ``-1`` sentinels for padding. The disabled and fallback
    paths return ``learned_indices`` unchanged.

    This is a CPU shadow helper. It is not wired into sparse MLA kernels and
    intentionally falls back for CUDA tensors to avoid unbenchmarked Python
    synchronization in an attention hot path.
    """
    if not config.enabled:
        return TopologyIndexResult(
            indices=learned_indices,
            applied=False,
            fallback_reason="disabled",
        )

    fallback_reason = _validate_topology_index_inputs(
        learned_indices, scores, segment_ids
    )
    if fallback_reason is not None:
        return TopologyIndexResult(
            indices=learned_indices,
            applied=False,
            fallback_reason=fallback_reason,
        )

    num_rows, topk = learned_indices.shape
    if topk == 0:
        return TopologyIndexResult(
            indices=learned_indices,
            applied=False,
            fallback_reason="empty_budget",
        )

    learned_keep = round(topk * config.learned_fraction)
    learned_keep = max(1, min(topk, learned_keep))
    replacement_budget = topk - learned_keep
    if config.max_replacements is not None:
        replacement_budget = min(replacement_budget, max(0, config.max_replacements))

    out = torch.full_like(learned_indices, -1)
    learned_retained = 0
    structural_inserted = 0
    max_context = scores.shape[1]
    active_segments = _active_topology_segments(segment_ids, config.max_segments)

    for row_idx in range(num_rows):
        row = learned_indices[row_idx]
        learned = _valid_unique_indices(row, max_context)
        selected = learned[:learned_keep]
        inserted_for_row = 0

        while len(selected) < topk and inserted_for_row < replacement_budget:
            candidate = _best_topology_tail_candidate(
                scores=scores,
                row_idx=row_idx,
                segment_ids=segment_ids,
                selected=selected,
                active_segments=active_segments,
                config=config,
            )
            if candidate is None:
                break
            selected.append(candidate)
            inserted_for_row += 1

        for index in learned[learned_keep:]:
            if len(selected) >= topk:
                break
            if index not in selected:
                selected.append(index)

        if selected:
            out[row_idx, : len(selected)] = torch.tensor(
                selected,
                dtype=torch.int32,
                device=learned_indices.device,
            )
        learned_retained += min(len(learned), learned_keep)
        structural_inserted += inserted_for_row

    return TopologyIndexResult(
        indices=out,
        applied=True,
        fallback_reason=None,
        learned_retained=learned_retained,
        structural_inserted=structural_inserted,
    )


def _validate_topology_index_inputs(
    learned_indices: torch.Tensor,
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
) -> str | None:
    if learned_indices.device.type != "cpu":
        return "cuda_not_supported"
    if (
        scores.device != learned_indices.device
        or segment_ids.device != learned_indices.device
    ):
        return "device_mismatch"
    if learned_indices.dtype != torch.int32:
        return "dtype"
    if learned_indices.ndim != 2:
        return "index_shape"
    if scores.ndim != 2 or scores.shape[0] != learned_indices.shape[0]:
        return "scores_shape"
    if segment_ids.dtype != torch.int32:
        return "segment_dtype"
    if segment_ids.ndim != 1 or segment_ids.shape[0] != scores.shape[1]:
        return "segment_shape"
    if torch.any(learned_indices < -1) or torch.any(learned_indices >= scores.shape[1]):
        return "index_bounds"
    return None


def _valid_unique_indices(row: torch.Tensor, max_context: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for value in row.tolist():
        index = int(value)
        if index < 0 or index >= max_context or index in seen:
            continue
        selected.append(index)
        seen.add(index)
    return selected


def _active_topology_segments(segment_ids: torch.Tensor, max_segments: int) -> set[int]:
    segments = torch.unique(segment_ids, sorted=True)
    if max_segments > 0:
        segments = segments[:max_segments]
    return {int(segment) for segment in segments.tolist()}


def _best_topology_tail_candidate(
    scores: torch.Tensor,
    row_idx: int,
    segment_ids: torch.Tensor,
    selected: list[int],
    active_segments: set[int],
    config: TopologyIndexConfig,
) -> int | None:
    selected_set = set(selected)
    selected_segments = {int(segment_ids[index]) for index in selected}
    best_candidate = None
    best_score = None

    for candidate in range(scores.shape[1]):
        if candidate in selected_set:
            continue
        segment = int(segment_ids[candidate])
        if segment not in active_segments:
            continue
        barrier = 1.0 if segment not in selected_segments else 0.0
        diversity = _topology_diversity_score(scores, candidate, selected)
        value = (
            float(scores[row_idx, candidate])
            + config.barrier_strength * barrier
            + config.diversity_strength * diversity
        )
        if best_score is None or value > best_score:
            best_score = value
            best_candidate = candidate
    return best_candidate


def _topology_diversity_score(
    scores: torch.Tensor,
    candidate: int,
    selected: list[int],
) -> float:
    if not selected:
        return 1.0
    candidate_profile = scores[:, candidate].float()
    selected_profiles = scores[:, selected].float()
    candidate_norm = torch.linalg.vector_norm(candidate_profile).clamp_min(1e-12)
    selected_norms = torch.linalg.vector_norm(selected_profiles, dim=0).clamp_min(1e-12)
    similarities = torch.matmul(candidate_profile, selected_profiles) / (
        candidate_norm * selected_norms
    )
    return float((1.0 - similarities.max()).clamp_min(0.0))


def merge_topology_tail_indices_reference(
    learned_indices: torch.Tensor,
    topology_indices: torch.Tensor,
    learned_keep: int,
    max_replacements: int,
) -> torch.Tensor:
    """Reference bounded tail replacement for sparse MLA topology witnesses."""
    _validate_topology_tail_merge_inputs(
        learned_indices,
        topology_indices,
        learned_keep,
        max_replacements,
    )
    out = learned_indices.clone()
    topk = learned_indices.shape[1]

    for row_idx in range(learned_indices.shape[0]):
        selected: set[int] = set()
        for value in learned_indices[row_idx].tolist():
            index = int(value)
            if index >= 0:
                selected.add(index)

        inserted = 0
        for value in topology_indices[row_idx].tolist():
            index = int(value)
            if inserted >= max_replacements or learned_keep + inserted >= topk:
                break
            if index < 0 or index in selected:
                continue
            out[row_idx, learned_keep + inserted] = index
            selected.add(index)
            inserted += 1
    return out


def merge_topology_tail_indices(
    learned_indices: torch.Tensor,
    topology_indices: torch.Tensor,
    learned_keep: int,
    max_replacements: int,
) -> torch.Tensor:
    """Replace a bounded learned sparse MLA tail with topology witnesses."""
    _validate_topology_tail_merge_inputs(
        learned_indices,
        topology_indices,
        learned_keep,
        max_replacements,
    )
    if learned_indices.device.type != "cuda":
        return merge_topology_tail_indices_reference(
            learned_indices,
            topology_indices,
            learned_keep,
            max_replacements,
        )

    topk = learned_indices.shape[1]
    topology_width = topology_indices.shape[1]
    out = torch.empty_like(learned_indices)
    block_topk = _next_power_of_2(topk)
    block_topology = _next_power_of_2(topology_width)

    _merge_topology_tail_indices_kernel[(learned_indices.shape[0],)](
        learned_indices.contiguous(),
        topology_indices.contiguous(),
        out,
        topk,
        topology_width,
        learned_keep,
        max_replacements,
        learned_indices.stride(0),
        learned_indices.stride(1),
        topology_indices.stride(0),
        topology_indices.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_TOPK=block_topk,
        BLOCK_TOPOLOGY=block_topology,
    )
    return out


def _validate_topology_tail_merge_inputs(
    learned_indices: torch.Tensor,
    topology_indices: torch.Tensor,
    learned_keep: int,
    max_replacements: int,
) -> None:
    if learned_indices.dtype != torch.int32 or topology_indices.dtype != torch.int32:
        raise ValueError("learned_indices and topology_indices must be torch.int32")
    if learned_indices.ndim != 2 or topology_indices.ndim != 2:
        raise ValueError("learned_indices and topology_indices must be 2D tensors")
    if learned_indices.device != topology_indices.device:
        raise ValueError("learned_indices and topology_indices must share a device")
    if learned_indices.shape[0] != topology_indices.shape[0]:
        raise ValueError("learned_indices and topology_indices rows must match")
    if learned_indices.shape[1] == 0:
        raise ValueError("learned_indices must have a non-empty top-k dimension")
    if topology_indices.shape[1] == 0:
        raise ValueError("topology_indices must have a non-empty candidate dimension")
    if learned_keep < 0 or learned_keep > learned_indices.shape[1]:
        raise ValueError("learned_keep must fit within the learned top-k dimension")
    if max_replacements < 0:
        raise ValueError("max_replacements must be non-negative")


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _merge_topology_tail_indices_kernel(
    learned_ptr,
    topology_ptr,
    out_ptr,
    TOPK: tl.constexpr,
    TOPOLOGY_WIDTH: tl.constexpr,
    LEARNED_KEEP: tl.constexpr,
    MAX_REPLACEMENTS: tl.constexpr,
    learned_stride0,
    learned_stride1,
    topology_stride0,
    topology_stride1,
    out_stride0,
    out_stride1,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOPOLOGY: tl.constexpr,
):
    row_idx = tl.program_id(0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    topology_offsets = tl.arange(0, BLOCK_TOPOLOGY)

    learned = tl.load(
        learned_ptr + row_idx * learned_stride0 + topk_offsets * learned_stride1,
        mask=topk_offsets < TOPK,
        other=-1,
    )
    topology = tl.load(
        topology_ptr
        + row_idx * topology_stride0
        + topology_offsets * topology_stride1,
        mask=topology_offsets < TOPOLOGY_WIDTH,
        other=-1,
    )

    tl.store(
        out_ptr + row_idx * out_stride0 + topk_offsets * out_stride1,
        learned,
        mask=topk_offsets < TOPK,
    )

    insert_count = 0
    for topology_pos in tl.static_range(0, BLOCK_TOPOLOGY):
        candidate = tl.load(
            topology_ptr
            + row_idx * topology_stride0
            + topology_pos * topology_stride1,
            mask=topology_pos < TOPOLOGY_WIDTH,
            other=-1,
        )
        duplicate_learned = tl.sum(
            tl.where((topk_offsets < TOPK) & (learned == candidate), 1, 0),
            axis=0,
        ) > 0
        duplicate_topology = tl.sum(
            tl.where(
                (topology_offsets < topology_pos) & (topology == candidate),
                1,
                0,
            ),
            axis=0,
        ) > 0
        valid = (
            (candidate >= 0)
            & (insert_count < MAX_REPLACEMENTS)
            & (LEARNED_KEEP + insert_count < TOPK)
            & ~duplicate_learned
            & ~duplicate_topology
        )
        tl.store(
            out_ptr
            + row_idx * out_stride0
            + (LEARNED_KEEP + insert_count) * out_stride1,
            candidate,
            mask=valid,
        )
        insert_count += valid.to(tl.int32)


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
    is_invalid_tok |= ~valid_block | is_remote
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
    out = torch.empty_like(token_indices_c)

    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

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
