# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalized configuration consumed by native offloading backends."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OffloadingGroupConfig:
    # Total token span covered by one block across all workers
    # (accounts for context parallelism).
    tokens_per_block: int
    # Layer names belonging to this group.
    layer_names: tuple[str, ...]
    # Original KVCacheConfig group index.
    group_id: int
    # Chunks of this group the tier keeps for one request, or None when
    # attention reaches back without a bound (a full layer).
    sliding_window_size_in_chunks: int | None = None


@dataclass(frozen=True)
class OffloadingModelConfig:
    # Model identifier (e.g. HuggingFace model path).
    name: str
    # KV cache data type (e.g. "float16").
    dtype: str
    # Longest request the engine accepts, in tokens. 0 when not known.
    max_model_len: int = 0


@dataclass(frozen=True)
class OffloadingCacheConfig:
    # Tokens per block hash.
    tokens_per_hash: int
    # Blocks coalesced into one offload chunk.
    blocks_per_chunk: int


def places_workers_by_nnodes(distributed_executor_backend: object) -> bool:
    """Whether the executor places workers by the configured node count, so
    that local_world_size says which workers share a node."""
    return distributed_executor_backend in ("mp", "uni")


def workers_may_span_nodes(
    world_size: int, local_world_size: int | None, placement_known: bool
) -> bool:
    """Whether an engine's workers may be on several nodes: as the executor
    reports, or for any engine of several workers whose placement is unknown."""
    if not placement_known:
        return world_size > 1
    return local_world_size is not None and local_world_size < world_size


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
    # Number of data parallel replicas.
    data_parallel_size: int
    # Local rank of the data parallel group, set only in SPMD mode.
    data_parallel_rank_local: int | None
    # True when the bytes that will be persisted for a block are portable
    # across parallelism configurations: for the direct layout, concatenating
    # the block's data across all workers in rank order yields the same bytes
    # under any topology; for the canonical layout, the canonical page itself
    # is topology-free.
    is_parallelism_agnostic: bool
    # Workers of this engine on each node; world_size when unknown.
    local_world_size: int | None = None
    # False when the executor does not place workers by the configured node
    # count (Ray, external_launcher, executor classes): local_world_size then
    # says nothing about the nodes.
    placement_known: bool = True

    @property
    def spans_nodes(self) -> bool:
        """Whether workers may be on several nodes."""
        return workers_may_span_nodes(
            self.world_size, self.local_world_size, self.placement_known
        )


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
    # True when the offloaded bytes of every worker are expected to be
    # byte-identical per block (pure-MLA model, single-node TP-only
    # parallelism), enabling a single-copy host layout in backends that
    # support it. Aggregate layout decision; per-layer replication metadata
    # is planned for CanonicalKVCacheRef (#48408).
    replicated_layout: bool = False
    # True when the canonical per-layer host byte layout was requested via
    # kv_connector_extra_config; certified per-layer at worker registration.
    canonical_layout: bool = False
    # Resolved KVCacheLayout name of the worker KV cache.
    kv_cache_layout: str | None = None
    # Whether the engine may release and remap the GPU KV cache memory: in
    # sleep mode, or when the extensible KV cache lends it to a weight update.
    remaps_kv_cache: bool = False
