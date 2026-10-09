# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalized configuration consumed by native offloading backends."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class OffloadingGroupConfig:
    # Total token span covered by one block across all workers
    # (accounts for context parallelism).
    tokens_per_block: int
    # Layer names belonging to this group.
    layer_names: tuple[str, ...]
    # Original KVCacheConfig group index.
    group_id: int


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
    # Unified number of CPU offload blocks across all workers, if precomputed.
    num_cpu_blocks: int | None = None


def unify_cpu_offload_num_chunks(
    vllm_config: "VllmConfig", kv_cache_configs: "list[KVCacheConfig]"
) -> int | None:
    """Smallest CPU-offload chunk count any worker can hold, or None.

    Under pipeline parallelism workers own different layers, so each derives a
    different capacity from its own KV cache tensors while the scheduler would
    otherwise size its block ids from a single worker. Taking the minimum keeps
    every scheduler-allocated chunk id addressable on every worker.

    Sizing is delegated to the CPU backend's own ``cpu_offload_layout`` so a
    worker's view and the scheduler's unified count cannot be derived
    differently.
    """
    # Imported here to avoid a cycle: the offloading connector config imports
    # this module, and the CPU spec imports the connector's group selection.
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
        get_offloading_group_ids,
        uses_cpu_offloading_spec,
    )
    from vllm.v1.kv_offload.cpu.spec import cpu_offload_layout

    if not uses_cpu_offloading_spec(vllm_config):
        return None

    def _num_chunks(kv_cache_config: "KVCacheConfig") -> int:
        # A stage owning no offloadable group contributes no capacity of its own.
        if not get_offloading_group_ids(kv_cache_config):
            return 0
        layout = cpu_offload_layout(
            build_offloading_config(vllm_config, kv_cache_config)
        )
        return layout.num_chunks

    return min(_num_chunks(c) for c in kv_cache_configs)
