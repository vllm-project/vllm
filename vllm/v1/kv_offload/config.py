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
