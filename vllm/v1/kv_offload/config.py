# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalized configuration consumed by native offloading backends."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactSliceConfig:
    """Per-slice accounting within one compact KV group (transport-neutral).

    Mirrors GroupCompactSlice from the CPU module but lives in the base config
    package to avoid a dependency cycle from the offloading config layer into
    the CPU manager submodule.
    """

    offset_bytes: int
    """Byte offset of this slice within the packed GPU block row."""
    real_bytes_per_gpu_block: int
    """Unpadded real KV payload bytes per GPU block per rank."""
    padded_bytes_per_gpu_block: int
    """Padded GPU block bytes consumed by this slice per rank."""
    layer_name: str
    """Model layer name attributed to this slice."""


@dataclass(frozen=True)
class CompactGroupSliceConfig:
    """Slice-level accounting for one KV group's compact layout.

    Aggregate ``compact_real_bytes_per_rank`` must equal the transported
    ``compact_bytes_per_native_block_per_worker`` on the matching
    :class:`OffloadingGroupConfig`.
    """

    group_idx: int
    slices: tuple[CompactSliceConfig, ...]
    compact_real_bytes_per_rank: int
    """Sum of ``s.real_bytes_per_gpu_block`` per GPU block per rank."""
    compact_padded_bytes_per_rank: int
    """Sum of ``s.padded_bytes_per_gpu_block`` per GPU block per rank."""

    @property
    def aggregate_bytes(self) -> int:
        """Total real bytes per GPU block per rank (alias for validation)."""
        return self.compact_real_bytes_per_rank


@dataclass(frozen=True)
class OffloadingGroupConfig:
    # Total token span covered by one block across all workers
    # (accounts for context parallelism).
    tokens_per_block: int
    # Layer names belonging to this group.
    layer_names: tuple[str, ...]
    # Compact layout: real bytes per worker per native block.
    # None when compact layout is disabled.
    compact_bytes_per_native_block_per_worker: int | None = None


@dataclass(frozen=True)
class OffloadingModelConfig:
    # Model identifier (e.g. HuggingFace model path).
    name: str
    # KV cache data type (e.g. "float16").
    dtype: str


@dataclass(frozen=True)
class OffloadingCacheConfig:
    # Tokens per block hash.
    tokens_per_hash: int
    # Blocks coalesced into one offload chunk.
    blocks_per_chunk: int


@dataclass(frozen=True)
class OffloadingParallelConfig:
    # Worker index in [0, world_size). 0 on the scheduler side.
    rank: int
    # Total number of workers.
    world_size: int
    # Tensor parallel size.
    tp_size: int
    # Pipeline parallel size.
    pp_size: int
    # Prefill context parallel size.
    pcp_size: int
    # Decode context parallel size.
    dcp_size: int
    # Data parallel replica index of this engine.
    data_parallel_index: int
    # True when concatenating a block's data across all workers yields
    # the same result regardless of the parallelism configuration.
    is_parallelism_agnostic: bool


@dataclass(frozen=True)
class OffloadingConfig:
    groups: tuple[OffloadingGroupConfig, ...]
    # KV bytes stored by one worker per block.
    worker_kv_bytes_per_block: int
    # Whether the scheduler emits KV cache events. When true,
    # the offloading backend should emit events as well.
    enable_kv_cache_events: bool
    # Offloading-specific configuration from kv_connector_extra_config.
    extra_config: Mapping[str, Any]
    # Unique identifier for this engine, distinct per DP rank.
    engine_id: str
    model: OffloadingModelConfig
    cache: OffloadingCacheConfig
    parallel: OffloadingParallelConfig
    # Compact slice accounting per group, present on worker-side config
    # when compact layout is enabled. None for legacy configs or scheduler-side
    # projected configs that lack packed tensor metadata.
    compact_slice_accounting: tuple[CompactGroupSliceConfig, ...] | None = None
