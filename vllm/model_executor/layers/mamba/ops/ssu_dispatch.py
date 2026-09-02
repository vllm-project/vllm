# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
the Triton, FlashInfer, or CPU backend based on the configured
`MambaBackendEnum`. On CPU-only platforms (PowerPC, x86 without CUDA)
the backend defaults to 'cpu'.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit(do_not_specialize=["num_reqs"])
def _postprocess_replayssm_modelwide_kernel(
    # Per-request step metadata.
    idx_mapping,
    query_metadata,
    num_computed_tokens,
    num_accepted_tokens,
    is_prefilling,
    live_cols,
    materialize_src_cols,
    materialize_dst_cols,
    materialize_token_counts,
    # Per-group address tables.
    block_table_ptrs,
    tracker_ring_start_ptrs,
    tracker_num_committed_ptrs,
    tracker_capacities,
    group_layer_offsets,
    # FlashInfer plan outputs.
    src_slots,
    dst_slots,
    plan_ring_start,
    plan_flush_count,
    # Runtime sizes.
    block_table_stride_req: tl.int64,
    slot_table_stride_layer: tl.int64,
    num_reqs,
    # Compile-time model constants.
    MAX_LAYERS_PER_GROUP: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    LOGICAL_WINDOW: tl.constexpr,
    RING_BUFFER_LEN: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    QUERY_IS_CUMULATIVE: tl.constexpr,
    NUM_COMPUTED_IS_AFTER: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
) -> None:
    """Plan materialization and commit all ReplaySSM trackers in one launch.

    One CTA owns one ``(batch row, cache group)`` pair, and is therefore the
    only writer of that group's tracker for the request. Group zero writes the
    request-level ``ring_start``/``flush_count`` snapshot shared by every layer;
    every group fills the layer rows belonging to its physical slot namespace.
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    active = batch_idx < num_reqs
    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx, mask=active, other=-1)
    valid_req = active & (req_idx >= 0)

    if group_idx == 0:
        # Always overwrite the request decision, including padded rows, so a
        # fixed-capacity FlashInfer call never observes stale work.
        tl.store(plan_ring_start + batch_idx, 0)
        tl.store(plan_flush_count + batch_idx, -1)

    block_table = tl.load(block_table_ptrs + group_idx).to(tl.pointer_type(tl.int32))
    tracker_start = tl.load(tracker_ring_start_ptrs + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    tracker_committed = tl.load(tracker_num_committed_ptrs + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    tracker_capacity = tl.load(tracker_capacities + group_idx)

    live_col = tl.load(live_cols + req_idx, mask=valid_req, other=-1)
    valid_live_col = valid_req & (live_col >= 0)
    live_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + live_col,
        mask=valid_live_col,
        other=PAD_SLOT_ID,
    )
    valid_live = (
        valid_live_col
        & (live_slot != PAD_SLOT_ID)
        & (live_slot >= 0)
        & (live_slot < tracker_capacity)
    )

    src_col = tl.load(materialize_src_cols + batch_idx, mask=active, other=-1)
    dst_col = tl.load(materialize_dst_cols + batch_idx, mask=active, other=-1)
    wants_materialize = valid_req & (src_col >= 0) & (dst_col >= 0)
    materialize_src_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + src_col,
        mask=wants_materialize,
        other=PAD_SLOT_ID,
    )
    materialize_dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=wants_materialize,
        other=PAD_SLOT_ID,
    )
    valid_materialize = (
        wants_materialize
        & (materialize_src_slot != PAD_SLOT_ID)
        & (materialize_dst_slot != PAD_SLOT_ID)
        & (materialize_src_slot >= 0)
        & (materialize_dst_slot >= 0)
        & (materialize_src_slot < tracker_capacity)
        & (materialize_dst_slot < tracker_capacity)
    )

    # Fill every flattened layer row for this group. Invalid rows still receive
    # the pad sentinel; request-level flush_count=-1 suppresses native writes.
    layer_begin = tl.load(group_layer_offsets + group_idx)
    layer_end = tl.load(group_layer_offsets + group_idx + 1)
    for layer_offset in tl.static_range(0, MAX_LAYERS_PER_GROUP):
        layer_idx = layer_begin + layer_offset
        layer_valid = layer_idx < layer_end
        table_offset = layer_idx * slot_table_stride_layer + batch_idx
        tl.store(
            src_slots + table_offset,
            tl.where(valid_materialize, materialize_src_slot, PAD_SLOT_ID),
            mask=layer_valid,
        )
        tl.store(
            dst_slots + table_offset,
            tl.where(valid_materialize, materialize_dst_slot, PAD_SLOT_ID),
            mask=layer_valid,
        )

    prefilling = tl.load(is_prefilling + batch_idx, mask=active, other=1)
    if QUERY_IS_CUMULATIVE:
        query_len = tl.load(
            query_metadata + batch_idx + 1, mask=active, other=0
        ) - tl.load(
            query_metadata + batch_idx,
            mask=active,
            other=0,
        )
    else:
        query_len = tl.load(query_metadata + batch_idx, mask=active, other=0)

    if valid_req & prefilling:
        computed = tl.load(num_computed_tokens + req_idx)
        computed_before = tl.where(
            NUM_COMPUTED_IS_AFTER, computed - query_len, computed
        )
        computed_after = computed_before + query_len
        first_col = tl.maximum(computed_before // MAMBA_BLOCK_SIZE, 0)
        last_col = tl.maximum(
            (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1,
            0,
        )
        # All-mode prefill writes every boundary state in this interval. Reset
        # every corresponding cursor so any later prefix hit copies an exact
        # canonical state instead of replaying rows from the slot's old owner.
        for col in tl.range(first_col, last_col + 1):
            prefill_slot = tl.load(
                block_table + batch_idx * block_table_stride_req + col
            )
            valid_prefill_slot = (
                (prefill_slot != PAD_SLOT_ID)
                & (prefill_slot >= 0)
                & (prefill_slot < tracker_capacity)
            )
            tl.store(tracker_start + prefill_slot, 0, mask=valid_prefill_slot)
            tl.store(tracker_committed + prefill_slot, 0, mask=valid_prefill_slot)

    if valid_live:
        if prefilling:
            if valid_materialize & (group_idx == 0):
                # The prefill kernel already produced an exact canonical state;
                # count zero asks FlashInfer to copy it byte-for-byte.
                tl.store(plan_ring_start + batch_idx, 0)
                tl.store(plan_flush_count + batch_idx, 0)
        else:
            old_start = tl.load(tracker_start + live_slot)
            old_committed = tl.load(tracker_committed + live_slot)
            accepted = tl.maximum(tl.load(num_accepted_tokens + req_idx), 1)
            checkpointed = old_committed + query_len > LOGICAL_WINDOW
            next_start = tl.where(
                checkpointed,
                (old_start + old_committed) % RING_BUFFER_LEN,
                old_start,
            )
            next_committed = tl.where(checkpointed, accepted, old_committed + accepted)

            if valid_materialize & (group_idx == 0):
                boundary_count = tl.load(materialize_token_counts + batch_idx)
                flush_count = next_committed - (accepted - boundary_count)
                tl.store(plan_ring_start + batch_idx, next_start)
                tl.store(plan_flush_count + batch_idx, flush_count)

            tl.store(tracker_start + live_slot, next_start)
            tl.store(tracker_committed + live_slot, next_committed)

    if valid_materialize:
        # The immutable plan above preserves any in-place transition for the
        # materializer; subsequent forwards see a canonical empty replay.
        tl.store(tracker_start + materialize_dst_slot, 0)
        tl.store(tracker_committed + materialize_dst_slot, 0)


@triton.jit(do_not_specialize=["num_reqs"])
def _copy_reassigned_replayssm_slots_kernel(
    idx_mapping,
    src_cols,
    dst_cols,
    block_table_ptrs,
    tracker_ring_start_ptrs,
    tracker_num_committed_ptrs,
    tracker_capacities,
    group_layer_offsets,
    src_slots,
    dst_slots,
    plan_ring_start,
    plan_flush_count,
    block_table_stride_req: tl.int64,
    slot_table_stride_layer: tl.int64,
    num_reqs,
    MAX_LAYERS_PER_GROUP: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
) -> None:
    """Plan an exact copy when align reassigns a request's writable slot."""
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    active = batch_idx < num_reqs
    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx, mask=active, other=-1)
    valid_req = active & (req_idx >= 0)

    if group_idx == 0:
        tl.store(plan_ring_start + batch_idx, 0)
        tl.store(plan_flush_count + batch_idx, -1)

    block_table = tl.load(block_table_ptrs + group_idx).to(tl.pointer_type(tl.int32))
    tracker_start = tl.load(tracker_ring_start_ptrs + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    tracker_committed = tl.load(tracker_num_committed_ptrs + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    tracker_capacity = tl.load(tracker_capacities + group_idx)
    src_col = tl.load(src_cols + req_idx, mask=valid_req, other=-1)
    dst_col = tl.load(dst_cols + req_idx, mask=valid_req, other=-1)
    wants_copy = valid_req & (src_col >= 0) & (dst_col >= 0) & (src_col != dst_col)
    src_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + src_col,
        mask=wants_copy,
        other=PAD_SLOT_ID,
    )
    dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=wants_copy,
        other=PAD_SLOT_ID,
    )
    valid_mapping = (
        wants_copy
        & (src_slot != PAD_SLOT_ID)
        & (dst_slot != PAD_SLOT_ID)
        & (src_slot >= 0)
        & (dst_slot >= 0)
        & (src_slot < tracker_capacity)
        & (dst_slot < tracker_capacity)
    )
    needs_copy = valid_mapping & (src_slot != dst_slot)
    if valid_mapping & (group_idx == 0):
        # Snapshot the source cursor before resetting the distinct destination.
        # The materializer uses it to copy the exact live state, including any
        # committed replay rows that have not reached a prefix boundary. The
        # logical migration activates the shared plan even when group 0 aliases;
        # every group independently suppresses unchanged physical slots below.
        tl.store(plan_ring_start + batch_idx, tl.load(tracker_start + src_slot))
        tl.store(
            plan_flush_count + batch_idx,
            tl.load(tracker_committed + src_slot),
        )

    layer_begin = tl.load(group_layer_offsets + group_idx)
    layer_end = tl.load(group_layer_offsets + group_idx + 1)
    for layer_offset in tl.static_range(0, MAX_LAYERS_PER_GROUP):
        layer_idx = layer_begin + layer_offset
        layer_valid = layer_idx < layer_end
        table_offset = layer_idx * slot_table_stride_layer + batch_idx
        tl.store(
            src_slots + table_offset,
            tl.where(needs_copy, src_slot, PAD_SLOT_ID),
            mask=layer_valid,
        )
        tl.store(
            dst_slots + table_offset,
            tl.where(needs_copy, dst_slot, PAD_SLOT_ID),
            mask=layer_valid,
        )

    if needs_copy:
        # The reassigned destination must not inherit its prior owner's cursor.
        tl.store(tracker_start + dst_slot, 0)
        tl.store(tracker_committed + dst_slot, 0)


def _replayssm_specialization_key(mixer: Any) -> tuple[Any, ...]:
    ssm = mixer.kv_cache[1]
    x_cache = mixer.kv_cache[2]
    b_cache = mixer.kv_cache[4]
    return (
        ssm.dtype,
        x_cache.dtype,
        mixer.A.dtype,
        ssm.size(1),
        ssm.size(2),
        ssm.size(3),
        ssm.size(1) // b_cache.size(1),
        int(mixer.replayssm_buffer_len),
        x_cache.size(2),
        bool(mixer.mamba_config.enable_stochastic_rounding),
        int(mixer.mamba_config.stochastic_rounding_philox_rounds or 0),
    )


@dataclass
class ReplaySSMModelContext:
    """Persistent all-layer tables for ReplaySSM post-step maintenance."""

    mixers: list[Any]
    group_layer_offsets: torch.Tensor
    block_table_ptrs: torch.Tensor
    tracker_ring_start_ptrs: torch.Tensor
    tracker_num_committed_ptrs: torch.Tensor
    tracker_capacities: torch.Tensor
    state_ptrs: torch.Tensor
    state_slot_strides: torch.Tensor
    x_cache_ptrs: torch.Tensor
    x_cache_slot_strides: torch.Tensor
    b_cache_ptrs: torch.Tensor
    b_cache_slot_strides: torch.Tensor
    dt_cache_ptrs: torch.Tensor
    dt_cache_slot_strides: torch.Tensor
    a_ptrs: torch.Tensor
    scale_ptrs: torch.Tensor
    scale_slot_strides: torch.Tensor
    src_slots: torch.Tensor
    dst_slots: torch.Tensor
    plan_ring_start: torch.Tensor
    plan_flush_count: torch.Tensor
    precopy_src_slots: torch.Tensor
    precopy_dst_slots: torch.Tensor
    precopy_ring_start: torch.Tensor
    precopy_flush_count: torch.Tensor
    block_table_stride_req: int
    max_num_reqs: int
    num_groups: int
    max_layers_per_group: int
    logical_window: int
    ring_buffer_len: int

    @classmethod
    def create(
        cls,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: Sequence[int],
        forward_context: Mapping[str, Any],
        block_tables: Sequence[torch.Tensor],
        max_num_reqs: int,
    ) -> "ReplaySSMModelContext | None":
        grouped = _flashinfer_replayssm_mixers_by_group(
            kv_cache_config, mamba_group_ids, forward_context
        )
        if not grouped:
            return None
        if len(block_tables) != len(mamba_group_ids):
            raise ValueError(
                f"expected {len(mamba_group_ids)} Mamba block tables, "
                f"got {len(block_tables)}"
            )
        block_table_by_gid = dict(zip(mamba_group_ids, block_tables))
        replayssm_block_tables = [block_table_by_gid[gid] for gid, _ in grouped]

        mixers = [mixer for _, group_mixers in grouped for mixer in group_mixers]
        if not _replayssm_materialize_ready(mixers):
            return None
        first = mixers[0]
        first_ssm = first.kv_cache[1]
        first_x = first.kv_cache[2]
        compatibility = _replayssm_specialization_key(first)
        for mixer in mixers[1:]:
            current = _replayssm_specialization_key(mixer)
            if current != compatibility:
                raise ValueError(
                    "A single model-wide FlashInfer ReplaySSM materialization "
                    "launch requires identical layer specialization; got "
                    f"{compatibility} and {current}"
                )

        device = first_ssm.device
        group_offsets = [0]
        for _, group_mixers in grouped:
            group_offsets.append(group_offsets[-1] + len(group_mixers))
        max_layers_per_group = max(
            group_offsets[i + 1] - group_offsets[i]
            for i in range(len(group_offsets) - 1)
        )
        tracker_owners = [group_mixers[0] for _, group_mixers in grouped]
        strides = {int(block_table.stride(0)) for block_table in replayssm_block_tables}
        if len(strides) != 1:
            raise ValueError(
                "model-wide ReplaySSM requires one block-table row stride; "
                f"got {sorted(strides)}"
            )

        zero_table = torch.zeros(len(mixers), dtype=torch.int64, device=device)
        return cls(
            mixers=mixers,
            group_layer_offsets=torch.tensor(
                group_offsets, dtype=torch.int32, device=device
            ),
            block_table_ptrs=_cuda_i64_ptrs(replayssm_block_tables),
            tracker_ring_start_ptrs=_cuda_i64_ptrs(
                [m._replayssm_ring_start for m in tracker_owners]
            ),
            tracker_num_committed_ptrs=_cuda_i64_ptrs(
                [m._replayssm_prev_num_accepted for m in tracker_owners]
            ),
            tracker_capacities=torch.tensor(
                [m._replayssm_ring_start.numel() for m in tracker_owners],
                dtype=torch.int32,
                device=device,
            ),
            state_ptrs=_cuda_i64_ptrs([m.kv_cache[1] for m in mixers]),
            state_slot_strides=_cuda_i64_slot_strides([m.kv_cache[1] for m in mixers]),
            x_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[2] for m in mixers]),
            x_cache_slot_strides=_cuda_i64_slot_strides(
                [m.kv_cache[2] for m in mixers]
            ),
            b_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[4] for m in mixers]),
            b_cache_slot_strides=_cuda_i64_slot_strides(
                [m.kv_cache[4] for m in mixers]
            ),
            dt_cache_ptrs=_cuda_i64_ptrs([m.kv_cache[3] for m in mixers]),
            dt_cache_slot_strides=_cuda_i64_slot_strides(
                [m.kv_cache[3] for m in mixers]
            ),
            a_ptrs=_cuda_i64_ptrs([m.A for m in mixers]),
            scale_ptrs=zero_table,
            scale_slot_strides=zero_table.clone(),
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
            precopy_src_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            precopy_dst_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            precopy_ring_start=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            precopy_flush_count=torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            ),
            block_table_stride_req=next(iter(strides)),
            max_num_reqs=max_num_reqs,
            num_groups=len(grouped),
            max_layers_per_group=max_layers_per_group,
            logical_window=int(first.replayssm_buffer_len),
            ring_buffer_len=first_x.size(2),
        )

    def postprocess_and_materialize(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        query_metadata: torch.Tensor,
        query_is_cumulative: bool,
        num_computed_tokens: torch.Tensor,
        num_computed_is_after: bool,
        num_accepted_tokens: torch.Tensor,
        is_prefilling: torch.Tensor,
        live_cols: torch.Tensor,
        materialize_src_cols: torch.Tensor,
        materialize_dst_cols: torch.Tensor,
        materialize_token_counts: torch.Tensor,
        mamba_block_size: int,
        num_reqs: int,
    ) -> None:
        """Commit lifecycle metadata, then materialize all layers once."""
        if num_reqs == 0:
            return
        _postprocess_replayssm_modelwide_kernel[(self.max_num_reqs, self.num_groups)](
            idx_mapping,
            query_metadata,
            num_computed_tokens,
            num_accepted_tokens,
            is_prefilling,
            live_cols,
            materialize_src_cols,
            materialize_dst_cols,
            materialize_token_counts,
            self.block_table_ptrs,
            self.tracker_ring_start_ptrs,
            self.tracker_num_committed_ptrs,
            self.tracker_capacities,
            self.group_layer_offsets,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.block_table_stride_req,
            self.src_slots.stride(0),
            num_reqs,
            MAX_LAYERS_PER_GROUP=self.max_layers_per_group,
            MAMBA_BLOCK_SIZE=mamba_block_size,
            LOGICAL_WINDOW=self.logical_window,
            RING_BUFFER_LEN=self.ring_buffer_len,
            PAD_SLOT_ID=NULL_BLOCK_ID,
            QUERY_IS_CUMULATIVE=query_is_cumulative,
            NUM_COMPUTED_IS_AFTER=num_computed_is_after,
            HAS_IDX_MAPPING=idx_mapping is not None,
        )

        self._materialize_planned(
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
        )

    def copy_reassigned_slots(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        src_cols: torch.Tensor,
        dst_cols: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Copy exact live state when align assigns a new writable slot."""
        if num_reqs == 0:
            return
        _copy_reassigned_replayssm_slots_kernel[
            (self.max_num_reqs, self.num_groups)
        ](
            idx_mapping,
            src_cols,
            dst_cols,
            self.block_table_ptrs,
            self.tracker_ring_start_ptrs,
            self.tracker_num_committed_ptrs,
            self.tracker_capacities,
            self.group_layer_offsets,
            self.precopy_src_slots,
            self.precopy_dst_slots,
            self.precopy_ring_start,
            self.precopy_flush_count,
            self.block_table_stride_req,
            self.precopy_src_slots.stride(0),
            num_reqs,
            MAX_LAYERS_PER_GROUP=self.max_layers_per_group,
            PAD_SLOT_ID=NULL_BLOCK_ID,
            HAS_IDX_MAPPING=idx_mapping is not None,
        )
        self._materialize_planned(
            self.precopy_src_slots,
            self.precopy_dst_slots,
            self.precopy_ring_start,
            self.precopy_flush_count,
        )

    def _materialize_planned(
        self,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
        ring_start: torch.Tensor,
        flush_count: torch.Tensor,
    ) -> None:
        first = self.mixers[0]
        mamba_config = first.mamba_config
        rand_seed = None
        philox_rounds = 0
        if mamba_config.enable_stochastic_rounding:
            rand_seed = torch.randint(
                0, 2**32, (1,), device=src_slots.device, dtype=torch.int64
            )
            philox_rounds = mamba_config.stochastic_rounding_philox_rounds or 10
        _load_replayssm_materialize()(
            self.state_ptrs,
            self.state_slot_strides,
            self.x_cache_ptrs,
            self.x_cache_slot_strides,
            self.b_cache_ptrs,
            self.b_cache_slot_strides,
            self.dt_cache_ptrs,
            self.dt_cache_slot_strides,
            self.a_ptrs,
            self.scale_ptrs,
            self.scale_slot_strides,
            src_slots,
            dst_slots,
            ring_start,
            flush_count,
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


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install flashinfer (>= 0.6.4): "
                "pip install flashinfer-python"
            ) from e
        logger.info_once("Using FlashInfer Mamba SSU algorithm: %s", self._algorithm)
        self._kernel = _fi_ssu

    @property
    def _algorithm(self) -> MambaSSUAlgorithm:
        return self._mamba_config.ssu_algorithm or "auto"

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            algorithm=self._algorithm,
        )


class CPUSSUBackend(MambaSSUBackend):
    """CPU SSU backend using the compiled C++ VSX/scalar kernel.

    On CPU-only platforms (PowerPC, x86 without CUDA) this dispatches to
    the vectorized C++ kernel registered as ``torch.ops._C.selective_state_update_cpu``.
    That kernel uses vec_op SIMD intrinsics (VSX on ppc64le, AVX2 on x86,
    scalar fallback elsewhere) and is parallelised with OpenMP across heads.

    Falls back to the pure-PyTorch implementation only if the C++ op is
    unavailable (e.g. a CPU-less build).
    """

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm import _custom_ops as ops

        self._cpp_kernel = ops.selective_state_update_cpu
        logger.info("CPUSSUBackend: using compiled C++ selective_state_update kernel.")

    @property
    def name(self) -> str:
        return "cpu"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        # C++ kernel: state shape expected as (nstates, nheads, dim, dstate)
        # The kernel writes in-place into `out` and updates `state`.
        self._cpp_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            state_batch_indices,
            dst_state_batch_indices,
            null_block_id,
            out,
            num_accepted_tokens,
            cu_seqlens,
        )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
    MambaBackendEnum.CPU: CPUSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


_flashinfer_replayssm_kernel: Callable[..., torch.Tensor] | None = None


@cache
def flashinfer_replayssm_autotune_supported() -> bool:
    """Return True when FlashInfer exposes ReplaySSM autotuning."""
    try:
        from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
            CheckpointingSSURunner,
        )
    except ImportError:
        return False
    return True


def selective_state_update_replayssm_flashinfer(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    enable_stochastic_rounding: bool = False,
    stochastic_rounding_philox_rounds: int = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run FlashInfer checkpointing SSU with model-owned tracker metadata."""
    if _flashinfer_replayssm_kernel is None:
        raise RuntimeError(
            "FlashInfer ReplaySSM has not been initialized. "
            "Call initialize_mamba_ssu_backend() with use_replayssm=True."
        )

    if x.dim() == 3:
        dim = 0 if cu_seqlens is not None else 1
        x = x.unsqueeze(dim)
        dt = dt.unsqueeze(dim)
        B = B.unsqueeze(dim)
        C = C.unsqueeze(dim)
        out = out.unsqueeze(dim)

    indices = state_batch_indices
    if indices is not None and indices.dim() > 1:
        indices = indices[:, 0]

    cb_scaled = cumAdt_vec = cb_old = None
    if scratch is not None:
        cb_scaled, cumAdt_vec, cb_old = scratch

    rand_seed = (
        torch.randint(0, 2**32, (1,), device=state.device, dtype=torch.int64)
        if enable_stochastic_rounding
        else None
    )
    return _flashinfer_replayssm_kernel(
        state,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        x,
        dt,
        A,
        B,
        C,
        out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=indices,
        pad_slot_id=null_block_id,
        rand_seed=rand_seed,
        philox_rounds=stochastic_rounding_philox_rounds or 10,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        enable_pdl=enable_pdl,
        cb_scaled=cb_scaled,
        cumAdt_vec=cumAdt_vec,
        cb_old=cb_old,
    )


def _reinterpret_u64_as_i64(value: int) -> int:
    """Preserve a uint64 pointer bit pattern in a torch.int64 tensor."""
    return value if value < (1 << 63) else value - (1 << 64)


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


def _flashinfer_replayssm_mixers_by_group(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: Sequence[int],
    forward_context: Mapping[str, Any],
) -> list[tuple[int, list[Any]]]:
    grouped: list[tuple[int, list[Any]]] = []
    for gid in mamba_group_ids:
        mixers: list[Any] = []
        for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache = getattr(layer, "kv_cache", ())
            mamba_config = getattr(layer, "mamba_config", None)
            backend = getattr(mamba_config, "backend", None)
            if (
                getattr(layer, "use_replayssm", False)
                and backend == MambaBackendEnum.FLASHINFER
                and len(kv_cache) >= 5
            ):
                mixers.append(layer)
        if mixers:
            grouped.append((gid, mixers))
    return grouped


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


def _replayssm_materialize_ready(mixers: list[Any]) -> bool:
    """False only before the caches are allocated; raises on a bad cache.

    A skip here is not free: ``state_skip_postprocess`` has already told the
    fused postprocess kernel not to copy this temporal state, so silently
    doing nothing would leave the destination block holding stale SSM state.
    The empty-cache case (profiling and other pre-allocation runs) is the one
    legitimate no-op; anything else is a misconfiguration and must be loud.
    """
    ssm = mixers[0].kv_cache[1]
    x_cache = mixers[0].kv_cache[2]
    if ssm.numel() == 0:
        return False
    if not ssm.is_cuda:
        raise RuntimeError(
            "FlashInfer ReplaySSM prefix materialization requires a CUDA SSM "
            f"state cache; got device {ssm.device}"
        )
    if x_cache.numel() == 0 or mixers[0]._replayssm_ring_start.numel() == 0:
        raise RuntimeError(
            "FlashInfer ReplaySSM prefix materialization requires allocated "
            "replay ring buffers and ring trackers"
        )
    return True


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_replayssm: bool = False,
) -> None:
    """Initialize the Mamba SSU backend and optional FlashInfer ReplaySSM."""
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _flashinfer_replayssm_kernel, _mamba_ssu_backend
    backend = mamba_config.backend

    if backend == MambaBackendEnum.TRITON:
        from vllm.platforms import current_platform

        if current_platform.is_cpu():
            logger.info(
                "CPU platform detected: overriding Mamba SSU backend "
                "from 'triton' to 'cpu'."
            )
            backend = MambaBackendEnum.CPU

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )
    if use_replayssm and backend not in (
        MambaBackendEnum.TRITON,
        MambaBackendEnum.FLASHINFER,
    ):
        raise ValueError(f"ReplaySSM does not support mamba backend {backend.value!r}")

    backend_cls = _BACKEND_REGISTRY[backend]
    if not isinstance(_mamba_ssu_backend, backend_cls):
        _mamba_ssu_backend = backend_cls(mamba_config)
        logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)

    _flashinfer_replayssm_kernel = None
    if use_replayssm and backend == MambaBackendEnum.FLASHINFER:
        try:
            from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer ReplaySSM requires a compatible flashinfer-python package"
            ) from e
        _flashinfer_replayssm_kernel = checkpointing_ssu
    if use_replayssm:
        logger.info("Using %s ReplaySSM backend.", backend.value)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    is_blackwell: bool = False,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        is_blackwell=is_blackwell,
    )
