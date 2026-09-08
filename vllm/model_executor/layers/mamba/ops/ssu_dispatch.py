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
from typing import Any, NamedTuple

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


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
    """Reset a fresh destination in this cache group's physical slot space.

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

    fresh = valid_dst & ~valid_src
    tl.store(tracker_start + dst_slot, 0, mask=fresh)
    tl.store(tracker_committed + dst_slot, 0, mask=fresh)


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
    # Mamba attention runs a one-token final prefill chunk with prior state as
    # decode. Commit the same transition here instead of resetting its cursors.
    prefilling = tl.load(is_prefilling + batch_idx)
    prefilling = prefilling & ((query_len != 1) | (computed_before <= 0))
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
        first_col = computed_before // MAMBA_BLOCK_SIZE
        last_col = tl.maximum(
            (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1,
            0,
        )
        # Any block touched by prefill can later become another request's live
        # source through a prefix-cache hit. Clear every such block's cursors,
        # including intermediate blocks that are neither live_slot nor dst_slot.
        for col in tl.range(first_col, last_col + 1):
            prefill_slot = tl.load(
                block_table + batch_idx * block_table_stride_req + col
            )
            valid_prefill_slot = prefill_slot != PAD_SLOT_ID
            tl.store(tracker_start + prefill_slot, 0, mask=valid_prefill_slot)
            tl.store(tracker_committed + prefill_slot, 0, mask=valid_prefill_slot)
        if materialize:
            # Prefill produced canonical state, so publish an exact copy.
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


@triton.jit
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
        first_x = first.replayssm_cache[0]

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
                x_cache_ptrs=_cuda_i64_ptrs([m.replayssm_cache[0] for m in mixers]),
                x_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.replayssm_cache[0] for m in mixers]
                ),
                B_cache_ptrs=_cuda_i64_ptrs([m.replayssm_cache[2] for m in mixers]),
                B_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.replayssm_cache[2] for m in mixers]
                ),
                dt_cache_ptrs=_cuda_i64_ptrs([m.replayssm_cache[1] for m in mixers]),
                dt_cache_slot_strides=_cuda_i64_slot_strides(
                    [m.replayssm_cache[1] for m in mixers]
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
            materialize_prefixes=cache_mode in ("align", "all"),
        )

    def reset_new_slots(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        src_cols: torch.Tensor,
        dst_cols: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Reset cursors for fresh physical slots in this cache group."""
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

    def materialize(self, materialize_fn: Callable[..., None]) -> None:
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
        materialize_fn(
            *self.materialize_tables,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.active_request_indices,
            state_dtype=first.kv_cache[1].dtype,
            input_dtype=first.replayssm_cache[0].dtype,
            matrixA_dtype=first.A.dtype,
            dim=first.kv_cache[1].size(2),
            dstate=first.kv_cache[1].size(3),
            num_heads=first.kv_cache[1].size(1),
            heads_per_group=(
                first.kv_cache[1].size(1) // first.replayssm_cache[2].size(1)
            ),
            max_window=self.logical_window,
            ring_buffer_len=self.ring_buffer_len,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
        )


@dataclass
class ReplaySSMModelContext:
    """ReplaySSM lifecycle split by physical cache-slot namespace."""

    groups: list[_ReplaySSMGroupContext]

    @property
    def materialize_prefixes(self) -> bool:
        return self.groups[0].materialize_prefixes

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
        modes = set()
        group_args = []
        for gid, mixers in grouped:
            spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            if not isinstance(spec, MambaSpec):
                raise TypeError(
                    "FlashInfer ReplaySSM layers require a Mamba cache spec; "
                    f"got {type(spec).__name__}"
                )
            modes.add(spec.mamba_cache_mode)
            group_args.append(
                (
                    mixers,
                    block_table_by_gid[gid],
                    spec.mamba_cache_mode,
                    spec.block_size,
                )
            )
        if len(modes) != 1:
            raise ValueError(
                "model-wide ReplaySSM requires one Mamba cache mode; "
                f"got {sorted(modes)}"
            )

        groups = [
            _ReplaySSMGroupContext.create(*args, max_num_reqs) for args in group_args
        ]
        return cls(groups=groups)

    def reset_new_slots(self, **kwargs: Any) -> None:
        for group in self.groups:
            group.reset_new_slots(**kwargs)

    def postprocess(self, **kwargs: Any) -> None:
        for group in self.groups:
            group.postprocess(**kwargs)

    def materialize(self) -> None:
        # Keep the optional FlashInfer dependency lazy: mode ``none`` never
        # reaches this path. Resolve the cached callable once for this model
        # operation, then invoke it once per physical cache-slot namespace.
        materialize_fn = _load_replayssm_materialize()
        for group in self.groups:
            group.materialize(materialize_fn)


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
            mamba_config = getattr(layer, "mamba_config", None)
            backend = getattr(mamba_config, "backend", None)
            if (
                getattr(layer, "use_replayssm", False)
                and backend == MambaBackendEnum.FLASHINFER
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
