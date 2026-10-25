# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned FlashInfer ReplaySSM state and request lifecycle."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec


def initialize_replayssm_ring_trackers(
    kv_cache_config: KVCacheConfig,
    forward_context: Mapping[str, Any],
) -> None:
    """Allocate and bind one tracker pair per ReplaySSM cache group."""
    for group in kv_cache_config.kv_cache_groups:
        layers = [
            layer
            for name in group.layer_names
            if isinstance(layer := forward_context.get(name), MambaBase)
            and layer.use_flashinfer_replayssm
        ]
        if not layers:
            continue
        state = layers[0].kv_cache[0]
        num_blocks, device = state.shape[0], state.device
        ring_start = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        num_committed = torch.zeros_like(ring_start)
        for layer in layers:
            layer._replayssm_ring_start = ring_start
            layer._replayssm_prev_num_accepted = num_committed


def get_replayssm_block_copy_tensors(
    forward_context: Mapping[str, Any],
) -> list[torch.Tensor]:
    """Include shared trackers in scheduler block copies alongside packed pages."""
    tensors: list[torch.Tensor] = []
    for layer in forward_context.values():
        if isinstance(layer, MambaBase) and layer.use_flashinfer_replayssm:
            tensors.extend(
                (layer._replayssm_ring_start, layer._replayssm_prev_num_accepted)
            )
    return tensors


def _reinterpret_u64_as_i64(value: int) -> int:
    """Preserve a uint64 pointer bit pattern in a torch.int64 tensor."""
    return value if value < (1 << 63) else value - (1 << 64)


@triton.jit
def mamba_state_copy_boundary(
    num_tokens_running_state,
    new_num_computed,
    block_size: tl.constexpr,
):
    """Return the aligned Mamba state-copy decision and destination."""
    aligned_new_computed = (new_num_computed // block_size) * block_size
    needs_copy = aligned_new_computed >= num_tokens_running_state
    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1
    return needs_copy, accept_token_bias, dest_block_idx


@triton.jit
def _reset_new_replayssm_slots_kernel(
    idx_mapping,
    src_cols,
    dst_cols,
    block_table,
    tracker_start,
    tracker_committed,
    block_table_stride_req: tl.int64,
    PAD_SLOT_ID: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
) -> None:
    """Copy live cursors or reset a fresh destination in this cache group.

    Source and destination are logical block-table columns. Their slot lookups
    distinguish a missing group-local source from a continuation.
    """
    batch_idx = tl.program_id(0)
    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx)
    valid_req = req_idx >= 0

    dst_col = tl.load(dst_cols + req_idx, mask=valid_req, other=-1)
    valid_dst_col = valid_req & (dst_col >= 0)
    dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=valid_dst_col,
        other=PAD_SLOT_ID,
    )
    valid_dst = valid_dst_col & (dst_slot != PAD_SLOT_ID)

    src_col = tl.load(src_cols + req_idx, mask=valid_req, other=-1)
    valid_src_col = valid_req & (src_col >= 0)
    src_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + src_col,
        mask=valid_src_col,
        other=PAD_SLOT_ID,
    )
    valid_src = valid_src_col & (src_slot != PAD_SLOT_ID)

    changed = valid_dst & (~valid_src | (src_slot != dst_slot))
    start = tl.load(tracker_start + src_slot, mask=valid_src, other=0)
    committed = tl.load(tracker_committed + src_slot, mask=valid_src, other=0)
    tl.store(tracker_start + dst_slot, start, mask=changed)
    tl.store(tracker_committed + dst_slot, committed, mask=changed)


@triton.jit
def _postprocess_replayssm_kernel(
    idx_mapping,
    query_metadata,
    num_computed_tokens,
    num_accepted_tokens,
    is_prefilling,
    live_cols,
    block_table,
    tracker_start,
    tracker_committed,
    src_slots,
    dst_slots,
    plan_ring_start,
    plan_flush_count,
    block_table_stride_req: tl.int64,
    slot_table_stride_layer: tl.int64,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    LOGICAL_WINDOW: tl.constexpr,
    RING_BUFFER_LEN: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    QUERY_METADATA_IS_CUMULATIVE: tl.constexpr,
    NUM_COMPUTED_IS_POST_STEP: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
    MATERIALIZE_PREFIXES: tl.constexpr,
    LIVE_COL_IS_ZERO: tl.constexpr,
) -> None:
    """Commit a completed step and prepare an optional prefix snapshot."""
    batch_idx = tl.program_id(0)

    tl.store(plan_ring_start + batch_idx, 0)
    tl.store(plan_flush_count + batch_idx, -1)

    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx)
        if req_idx < 0:
            return

    if QUERY_METADATA_IS_CUMULATIVE:
        query_len = tl.load(query_metadata + batch_idx + 1) - tl.load(
            query_metadata + batch_idx
        )
    else:
        query_len = tl.load(query_metadata + batch_idx)

    computed = tl.load(num_computed_tokens + req_idx)
    computed_before = tl.where(
        NUM_COMPUTED_IS_POST_STEP, computed - query_len, computed
    )
    # Staged from forward metadata using Mamba attention's classification.
    prefilling = tl.load(is_prefilling + batch_idx)
    accepted = tl.maximum(tl.load(num_accepted_tokens + req_idx), 1)

    # Derive this request's pre/post-step positions from ReplaySSM metadata.
    computed_after = tl.where(
        prefilling,
        computed_before + query_len,
        computed if NUM_COMPUTED_IS_POST_STEP else computed_before + accepted,
    )
    running_state_pos = tl.where(
        prefilling, computed_after, computed_after - accepted + 1
    )
    boundary, accept_token_bias, dst_col = mamba_state_copy_boundary(
        running_state_pos,
        computed_after,
        MAMBA_BLOCK_SIZE,
    )

    live_col = 0 if LIVE_COL_IS_ZERO else tl.load(live_cols + req_idx)
    valid_live_col = live_col >= 0
    live_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + live_col,
        mask=valid_live_col,
        other=PAD_SLOT_ID,
    )
    valid_live = valid_live_col & (live_slot != PAD_SLOT_ID)
    wants_materialize = MATERIALIZE_PREFIXES & valid_live & boundary & (dst_col >= 0)
    dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=wants_materialize,
        other=PAD_SLOT_ID,
    )
    # Block-table writers emit either the null sentinel or an in-capacity ID.
    materialize = wants_materialize & (dst_slot != PAD_SLOT_ID)
    if MATERIALIZE_PREFIXES:
        # FlashInfer's ABI requires packed [layer, batch] tables even though all
        # layers in this cache group share the same physical slot namespace.
        for layer_idx in tl.static_range(0, NUM_LAYERS):
            slot_offset = layer_idx * slot_table_stride_layer + batch_idx
            tl.store(
                src_slots + slot_offset,
                tl.where(materialize, live_slot, PAD_SLOT_ID),
            )
            tl.store(
                dst_slots + slot_offset,
                tl.where(materialize, dst_slot, PAD_SLOT_ID),
            )
    if prefilling:
        if MATERIALIZE_PREFIXES:
            first_col = computed_before // MAMBA_BLOCK_SIZE
            last_col = tl.maximum(
                (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1,
                0,
            )
            # Reset every touched prefix block, including intermediate blocks
            # that can later become another request's live source.
            for col in tl.range(first_col, last_col + 1):
                prefill_slot = tl.load(
                    block_table + batch_idx * block_table_stride_req + col
                )
                valid_prefill_slot = prefill_slot != PAD_SLOT_ID
                tl.store(tracker_start + prefill_slot, 0, mask=valid_prefill_slot)
                tl.store(tracker_committed + prefill_slot, 0, mask=valid_prefill_slot)
        else:
            # Mode none has one live column, regardless of the prompt length.
            tl.store(tracker_start + live_slot, 0, mask=valid_live)
            tl.store(tracker_committed + live_slot, 0, mask=valid_live)
        if materialize & (live_slot != dst_slot):
            # Prefill already wrote canonical state; only another slot needs a copy.
            tl.store(plan_flush_count + batch_idx, 0)
    elif valid_live:
        old_start = tl.load(tracker_start + live_slot)
        old_committed = tl.load(tracker_committed + live_slot)
        checkpointed = old_committed + query_len > LOGICAL_WINDOW
        next_start = tl.where(
            checkpointed,
            (old_start + old_committed) % RING_BUFFER_LEN,
            old_start,
        )
        next_committed = tl.where(checkpointed, accepted, old_committed + accepted)
        tl.store(tracker_start + live_slot, next_start)
        tl.store(tracker_committed + live_slot, next_committed)

        if materialize:
            tl.store(plan_ring_start + batch_idx, next_start)
            tl.store(
                plan_flush_count + batch_idx,
                accept_token_bias + 1 + tl.where(checkpointed, 0, old_committed),
            )

    # The published destination is canonical and therefore has no live replay.
    # When dst_slot aliases live_slot, these stores intentionally supersede the
    # live tracker update above: the newly published snapshot is canonical.
    tl.store(tracker_start + dst_slot, 0, mask=materialize)
    tl.store(tracker_committed + dst_slot, 0, mask=materialize)


@triton.jit(do_not_specialize=["num_reqs"])
def _compact_replayssm_requests_kernel(
    plan_flush_count,
    active_request_indices,
    num_reqs,
    MAX_NUM_REQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Build FlashInfer's active-prefix request list in one parallel program."""
    offsets = tl.arange(0, BLOCK_SIZE)
    in_capacity = offsets < MAX_NUM_REQS
    active = (offsets < num_reqs) & (
        tl.load(plan_flush_count + offsets, mask=in_capacity, other=-1) >= 0
    )
    active_i32 = active.to(tl.int32)
    output_offsets = tl.cumsum(active_i32, axis=0) - 1
    num_active = tl.sum(active_i32, axis=0)
    tl.store(
        active_request_indices + output_offsets,
        offsets,
        mask=in_capacity & active,
    )
    tl.store(
        active_request_indices + offsets,
        -1,
        mask=in_capacity & (offsets >= num_active),
    )


class _ReplaySSMMaterializeTables(NamedTuple):
    state_ptrs: torch.Tensor
    state_slot_strides: torch.Tensor
    x_cache_ptrs: torch.Tensor
    x_cache_slot_strides: torch.Tensor
    B_cache_ptrs: torch.Tensor
    B_cache_slot_strides: torch.Tensor
    dt_cache_ptrs: torch.Tensor
    dt_cache_slot_strides: torch.Tensor
    A_ptrs: torch.Tensor
    state_scale_ptrs: torch.Tensor
    state_scale_slot_strides: torch.Tensor


@dataclass
class _ReplaySSMGroupContext:
    """ReplaySSM state sharing one physical cache-slot namespace."""

    mixers: list[Any]
    block_table: torch.Tensor
    ring_start: torch.Tensor
    num_committed: torch.Tensor
    materialize_tables: _ReplaySSMMaterializeTables
    src_slots: torch.Tensor
    dst_slots: torch.Tensor
    plan_ring_start: torch.Tensor
    plan_flush_count: torch.Tensor
    active_request_indices: torch.Tensor
    max_num_reqs: int
    mamba_block_size: int
    logical_window: int
    ring_buffer_len: int
    materialize_prefixes: bool

    @classmethod
    def create(
        cls,
        mixers: list[Any],
        block_table: torch.Tensor,
        cache_mode: str,
        mamba_block_size: int,
        max_num_reqs: int,
    ) -> "_ReplaySSMGroupContext":
        first = mixers[0]
        first_ssm = first.kv_cache[1]
        first_x = first.kv_cache[2]

        device = first_ssm.device
        zero_table = torch.zeros(len(mixers), dtype=torch.int64, device=device)
        return cls(
            mixers=mixers,
            block_table=block_table,
            ring_start=first._replayssm_ring_start,
            num_committed=first._replayssm_prev_num_accepted,
            materialize_tables=_ReplaySSMMaterializeTables(
                state_ptrs=_cuda_i64_ptrs([m.kv_cache[1] for m in mixers]),
                state_slot_strides=_cuda_i64_slot_strides(
                    [m.kv_cache[1] for m in mixers]
                ),
                x_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[2] for m in mixers]),
                x_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.kv_cache[2] for m in mixers]
                ),
                B_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[4] for m in mixers]),
                B_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.kv_cache[4] for m in mixers]
                ),
                dt_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[3] for m in mixers]),
                dt_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.kv_cache[3] for m in mixers]
                ),
                A_ptrs=_cuda_i64_ptrs([m.A for m in mixers]),
                state_scale_ptrs=zero_table,
                state_scale_slot_strides=zero_table,
            ),
            src_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            dst_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            plan_ring_start=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
            plan_flush_count=torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            ),
            active_request_indices=torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            ),
            max_num_reqs=max_num_reqs,
            mamba_block_size=mamba_block_size,
            logical_window=int(first.replayssm_buffer_len),
            ring_buffer_len=first_x.size(2),
            materialize_prefixes=cache_mode == "align",
        )

    def reset_new_slots(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        src_cols: torch.Tensor,
        dst_cols: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Migrate live cursors and reset fresh physical slots."""
        if num_reqs == 0:
            return
        _reset_new_replayssm_slots_kernel[(num_reqs,)](
            idx_mapping,
            src_cols,
            dst_cols,
            self.block_table,
            self.ring_start,
            self.num_committed,
            self.block_table.stride(0),
            PAD_SLOT_ID=NULL_BLOCK_ID,
            HAS_IDX_MAPPING=idx_mapping is not None,
        )

    def postprocess(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        query_metadata: torch.Tensor,
        query_metadata_is_cumulative: bool,
        num_computed_tokens: torch.Tensor,
        num_computed_is_post_step: bool,
        num_accepted_tokens: torch.Tensor,
        is_prefilling: torch.Tensor,
        live_cols: torch.Tensor | None,
        num_reqs: int,
    ) -> None:
        """Commit a completed step and prepare an optional prefix snapshot."""
        if num_reqs == 0:
            return
        _postprocess_replayssm_kernel[(num_reqs,)](
            idx_mapping,
            query_metadata,
            num_computed_tokens,
            num_accepted_tokens,
            is_prefilling,
            live_cols,
            self.block_table,
            self.ring_start,
            self.num_committed,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.block_table.stride(0),
            self.src_slots.stride(0),
            MAMBA_BLOCK_SIZE=self.mamba_block_size,
            LOGICAL_WINDOW=self.logical_window,
            RING_BUFFER_LEN=self.ring_buffer_len,
            NUM_LAYERS=len(self.mixers),
            PAD_SLOT_ID=NULL_BLOCK_ID,
            QUERY_METADATA_IS_CUMULATIVE=query_metadata_is_cumulative,
            NUM_COMPUTED_IS_POST_STEP=num_computed_is_post_step,
            HAS_IDX_MAPPING=idx_mapping is not None,
            MATERIALIZE_PREFIXES=self.materialize_prefixes,
            LIVE_COL_IS_ZERO=live_cols is None,
        )
        if self.materialize_prefixes:
            _compact_replayssm_requests_kernel[(1,)](
                self.plan_flush_count,
                self.active_request_indices,
                num_reqs,
                MAX_NUM_REQS=self.max_num_reqs,
                BLOCK_SIZE=triton.next_power_of_2(self.max_num_reqs),
            )

    def materialize(self) -> None:
        """Publish the canonical prefix snapshots prepared by ``postprocess``."""
        first = self.mixers[0]
        mamba_config = first.mamba_config
        rand_seed = None
        philox_rounds = 0
        if mamba_config.enable_stochastic_rounding:
            rand_seed = torch.randint(
                0, 2**32, (1,), device=self.src_slots.device, dtype=torch.int64
            )
            philox_rounds = mamba_config.stochastic_rounding_philox_rounds or 10
        _load_replayssm_materialize()(
            *self.materialize_tables,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.active_request_indices,
            state_dtype=first.kv_cache[1].dtype,
            input_dtype=first.kv_cache[2].dtype,
            matrixA_dtype=first.A.dtype,
            dim=first.kv_cache[1].size(2),
            dstate=first.kv_cache[1].size(3),
            num_heads=first.kv_cache[1].size(1),
            heads_per_group=(first.kv_cache[1].size(1) // first.kv_cache[4].size(1)),
            max_window=self.logical_window,
            ring_buffer_len=self.ring_buffer_len,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
        )


@dataclass
class ReplaySSMModelContext:
    """ReplaySSM lifecycle split by physical cache-slot namespace."""

    groups: list[_ReplaySSMGroupContext]

    @classmethod
    def create(
        cls,
        grouped: Sequence[tuple[list[Any], MambaSpec, torch.Tensor]],
        max_num_reqs: int,
    ) -> "ReplaySSMModelContext":
        return cls(
            groups=[
                _ReplaySSMGroupContext.create(
                    mixers,
                    block_table,
                    spec.mamba_cache_mode,
                    spec.block_size,
                    max_num_reqs,
                )
                for mixers, spec, block_table in grouped
            ]
        )

    def reset_new_slots(self, **kwargs: Any) -> None:
        for group in self.groups:
            group.reset_new_slots(**kwargs)

    def postprocess(self, **kwargs: Any) -> None:
        """Commit all groups before publishing their canonical prefix states."""
        for group in self.groups:
            group.postprocess(**kwargs)
        if not self.groups[0].materialize_prefixes:
            return
        for group in self.groups:
            group.materialize()


def _cuda_i64_ptrs(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [_reinterpret_u64_as_i64(t.data_ptr()) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def _cuda_i64_slot_strides(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [t.stride(0) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


@cache
def _load_replayssm_materialize() -> Callable[..., None]:
    try:
        from flashinfer.mamba.replayssm_materialize import (
            replayssm_materialize,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInfer ReplaySSM prefix caching requires "
            "flashinfer.mamba.replayssm_materialize"
        ) from e
    return replayssm_materialize
