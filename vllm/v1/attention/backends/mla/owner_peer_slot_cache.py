# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-local cache for owner-sharded sparse-MLA peer slots."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mla.sparse_utils import (
    convert_global_indices_to_dcp_peer_slots,
)


@dataclass(frozen=True)
class _PeerSlotLayout:
    block_table_ptr: int
    block_table_shape: tuple[int, ...]
    block_table_stride: tuple[int, ...]
    dcp_size: int
    blocks_per_peer: int
    cp_kv_cache_interleave_size: int
    block_size: int


@dataclass(frozen=True)
class _OwnerLocalSlotLayout:
    generation: int
    peer_layout: _PeerSlotLayout
    rows: int
    source_stride: int
    owner_rank: int
    dcp_world_size: int
    blocks_per_peer: int
    block_size: int


class OwnerPeerSlotCache:
    """Persist translated top-k slots across IndexShare attention layers.

    GLM IndexShare recomputes logical top-k indices only on selected layers.
    Physical Main-KV slots are layer-independent within the shared attention
    cache group, so the producing Indexer layer translates once and subsequent
    shared layers consume the persistent result.
    """

    def __init__(self, peer_slots: torch.Tensor, valid_counts: torch.Tensor) -> None:
        if peer_slots.ndim != 2 or peer_slots.dtype != torch.int32:
            raise ValueError("peer_slots must be a two-dimensional int32 tensor.")
        if (
            valid_counts.shape != (peer_slots.shape[0],)
            or valid_counts.dtype != torch.int32
            or valid_counts.device != peer_slots.device
        ):
            raise ValueError(
                "valid_counts must be a matching one-dimensional int32 tensor."
            )
        self.peer_slots = peer_slots
        self.valid_counts = valid_counts
        self.valid = False
        self.row_count = 0
        self.generation = 0
        self._layout: _PeerSlotLayout | None = None
        self._owner_local_layout: _OwnerLocalSlotLayout | None = None
        self._owner_local_slots: torch.Tensor | None = None
        self._owner_local_valid_counts: torch.Tensor | None = None
        self._owner_local_metadata_key: tuple[object, ...] | None = None
        self._owner_local_metadata: object | None = None

    def invalidate(self) -> None:
        self.valid = False
        self.row_count = 0
        self._layout = None
        self._invalidate_owner_local()

    def _invalidate_owner_local(self) -> None:
        self._owner_local_layout = None
        self._owner_local_slots = None
        self._owner_local_valid_counts = None
        self._owner_local_metadata_key = None
        self._owner_local_metadata = None

    @staticmethod
    def _make_layout(
        block_table: torch.Tensor,
        *,
        dcp_size: int,
        blocks_per_peer: int,
        cp_kv_cache_interleave_size: int,
        block_size: int,
    ) -> _PeerSlotLayout:
        return _PeerSlotLayout(
            block_table_ptr=block_table.data_ptr(),
            block_table_shape=tuple(block_table.shape),
            block_table_stride=tuple(block_table.stride()),
            dcp_size=dcp_size,
            blocks_per_peer=blocks_per_peer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
        )

    def refresh(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        dcp_size: int,
        blocks_per_peer: int,
        cp_kv_cache_interleave_size: int,
        block_size: int,
    ) -> None:
        rows, topk = token_indices.shape
        if rows > self.peer_slots.shape[0] or topk != self.peer_slots.shape[1]:
            raise RuntimeError(
                "Owner peer-slot cache cannot hold the refreshed top-k: "
                f"input={tuple(token_indices.shape)}, "
                f"capacity={tuple(self.peer_slots.shape)}."
            )
        slots = self.peer_slots[:rows]
        counts = self.valid_counts[:rows]
        convert_global_indices_to_dcp_peer_slots(
            req_id,
            block_table,
            token_indices,
            dcp_size=dcp_size,
            blocks_per_peer=blocks_per_peer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
            return_valid_counts=True,
            out=slots,
            valid_counts_out=counts,
        )
        self.row_count = rows
        self._layout = self._make_layout(
            block_table,
            dcp_size=dcp_size,
            blocks_per_peer=blocks_per_peer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
        )
        self.generation += 1
        self.valid = True
        self._invalidate_owner_local()

    def get(
        self,
        rows: int,
        block_table: torch.Tensor,
        *,
        dcp_size: int,
        blocks_per_peer: int,
        cp_kv_cache_interleave_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_layout = self._make_layout(
            block_table,
            dcp_size=dcp_size,
            blocks_per_peer=blocks_per_peer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
        )
        if not self.valid:
            raise RuntimeError(
                "Owner peer-slot cache was consumed before an Indexer refresh."
            )
        if rows < 0 or rows > self.row_count:
            raise RuntimeError(
                "Owner peer-slot cache row count is stale: "
                f"requested={rows}, available={self.row_count}."
            )
        if self._layout != expected_layout:
            raise RuntimeError(
                "Owner peer-slot cache layout does not match this attention layer."
            )
        return self.peer_slots[:rows], self.valid_counts[:rows]

    def get_or_build_owner_local(
        self,
        rows: int,
        block_table: torch.Tensor,
        *,
        source_stride: int,
        owner_rank: int,
        dcp_world_size: int,
        blocks_per_peer: int,
        cp_kv_cache_interleave_size: int,
        block_size: int,
        build: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return owner-local routed slots for the current IndexShare epoch.

        ``build`` performs the cross-rank slot exchange and owner filtering.
        Its result is layer-independent until the next Indexer refresh, so
        shared attention layers reuse it. Query exchange remains outside this
        cache and therefore still runs for every layer.
        """
        peer_slots, _ = self.get(
            rows,
            block_table,
            dcp_size=dcp_world_size,
            blocks_per_peer=blocks_per_peer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
        )
        if source_stride < rows or source_stride <= 0:
            raise RuntimeError(
                "Owner-local slot cache source stride is smaller than its "
                f"active rows: rows={rows}, stride={source_stride}."
            )
        if not 0 <= owner_rank < dcp_world_size:
            raise RuntimeError(
                "Owner-local slot cache owner rank is outside the DCP world: "
                f"rank={owner_rank}, world={dcp_world_size}."
            )
        assert self._layout is not None
        expected_layout = _OwnerLocalSlotLayout(
            generation=self.generation,
            peer_layout=self._layout,
            rows=rows,
            source_stride=source_stride,
            owner_rank=owner_rank,
            dcp_world_size=dcp_world_size,
            blocks_per_peer=blocks_per_peer,
            block_size=block_size,
        )
        if self._owner_local_layout == expected_layout:
            assert self._owner_local_slots is not None
            assert self._owner_local_valid_counts is not None
            return self._owner_local_slots, self._owner_local_valid_counts

        padded_peer_slots = torch.full(
            (source_stride, peer_slots.shape[1]),
            -1,
            dtype=torch.int32,
            device=peer_slots.device,
        )
        if rows > 0:
            padded_peer_slots[:rows].copy_(peer_slots)
        local_slots, local_valid_counts = build(padded_peer_slots)
        expected_rows = dcp_world_size * source_stride
        expected_slots_shape = (expected_rows, peer_slots.shape[1])
        if (
            local_slots.dtype != torch.int32
            or local_slots.device != peer_slots.device
            or tuple(local_slots.shape) != expected_slots_shape
        ):
            raise RuntimeError(
                "Owner-local slot builder returned an invalid slot tensor: "
                f"expected={expected_slots_shape} int32 on {peer_slots.device}, "
                f"actual={tuple(local_slots.shape)} {local_slots.dtype} "
                f"on {local_slots.device}."
            )
        if (
            local_valid_counts.dtype != torch.int32
            or local_valid_counts.device != peer_slots.device
            or tuple(local_valid_counts.shape) != (expected_rows,)
        ):
            raise RuntimeError(
                "Owner-local slot builder returned invalid valid counts: "
                f"expected={(expected_rows,)} int32 on {peer_slots.device}, "
                f"actual={tuple(local_valid_counts.shape)} "
                f"{local_valid_counts.dtype} on {local_valid_counts.device}."
            )
        self._owner_local_layout = expected_layout
        self._owner_local_slots = local_slots
        self._owner_local_valid_counts = local_valid_counts
        return local_slots, local_valid_counts

    def get_or_build_owner_local_metadata(
        self,
        key: tuple[object, ...],
        build: Callable[[], object],
    ) -> object:
        """Reuse kernel metadata within one owner-local IndexShare epoch.

        Some sparse kernels derive scheduler metadata from the fixed routed
        shape and the owner-local valid-count tensor. Both remain identical
        across the shared attention layers that consume one Indexer result.
        The next ``refresh`` invalidates this object together with the routed
        slots, so data-dependent metadata cannot escape its selection epoch.
        """
        if self._owner_local_layout is None:
            raise RuntimeError(
                "Owner-local kernel metadata was requested before routed slots."
            )
        metadata_key = (self._owner_local_layout, *key)
        if self._owner_local_metadata_key == metadata_key:
            assert self._owner_local_metadata is not None
            return self._owner_local_metadata
        metadata = build()
        self._owner_local_metadata_key = metadata_key
        self._owner_local_metadata = metadata
        return metadata


def maybe_allocate_owner_peer_slot_cache(
    vllm_config: VllmConfig,
    topk_indices_buffer: torch.Tensor,
) -> OwnerPeerSlotCache | None:
    """Allocate only for the owner-sharded history path."""
    parallel_config = vllm_config.parallel_config
    if (
        not envs.VLLM_USE_PCP_OWNER_HISTORY
        or parallel_config.prefill_context_parallel_size <= 1
        or parallel_config.decode_context_parallel_size <= 1
    ):
        return None
    peer_slots = torch.empty_like(topk_indices_buffer)
    valid_counts = torch.empty(
        topk_indices_buffer.shape[0],
        dtype=torch.int32,
        device=topk_indices_buffer.device,
    )
    return OwnerPeerSlotCache(peer_slots, valid_counts)
