# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalized configuration consumed by native offloading backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OffloadAccessMode(Enum):
    """Direction(s) in which an offloading connector may transfer data.

    Configured through the optional `access_mode` key of a connector's
    config: `read` serves loads but never accepts stores, `write` accepts
    stores but never serves loads, and `readwrite` (the default) does both.
    """

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"

    @classmethod
    def from_config(cls, value: str | OffloadAccessMode | None) -> OffloadAccessMode:
        """Parse a configured `access_mode` value, defaulting to `readwrite`."""
        if value is None:
            return cls.READWRITE
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"access_mode must be a string, got {type(value).__name__}: {value!r}"
            )
        try:
            return cls(value.strip().lower())
        except ValueError:
            valid = ", ".join(repr(access_mode.value) for access_mode in cls)
            raise ValueError(
                f"Invalid access_mode {value!r}; expected one of {valid}"
            ) from None

    @property
    def can_read(self) -> bool:
        """Whether the connector may serve loads (read from the tier)."""
        return self is not OffloadAccessMode.WRITE

    @property
    def can_write(self) -> bool:
        """Whether the connector may accept stores (write to the tier)."""
        return self is not OffloadAccessMode.READ


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class OffloadingModelConfig:
    # Model identifier (e.g. HuggingFace model path).
    name: str
    # KV cache data type (e.g. "float16").
    dtype: str
    # Longest request the engine accepts, in tokens. 0 when not known.
    max_model_len: int = 0


@dataclass(frozen=True, slots=True)
class OffloadingCacheConfig:
    # Tokens per block hash.
    tokens_per_hash: int
    # Blocks coalesced into one offload chunk.
    blocks_per_chunk: int


@dataclass(frozen=True, slots=True)
class OffloadingParallelConfig:
    # Worker index in [0, world_size). 0 on the scheduler side.
    rank: int
    # Total number of workers.
    world_size: int
    # Number of workers on this node.
    local_world_size: int
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


@dataclass(frozen=True, slots=True)
class OffloadingConfig:
    groups: tuple[OffloadingGroupConfig, ...]
    # CPU bytes reserved for each worker per block, the same on every pipeline
    # stage: the largest stage's KV bytes per block.
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
