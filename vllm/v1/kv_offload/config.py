# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalized configuration consumed by native offloading backends."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OffloadingGroupConfig:
    """Normalized configuration for one KV cache group."""

    block_size: int
    gpu_block_size: int
    layer_names: tuple[str, ...]
    is_non_mla_full_attention: bool


@dataclass(frozen=True)
class OffloadingConfig:
    """Model-level configuration for a native offloading backend."""

    groups: tuple[OffloadingGroupConfig, ...]
    hash_block_size: int
    block_size_factor: int
    num_gpu_blocks: int
    worker_kv_bytes_per_gpu_block: int
    world_size: int
    enable_kv_cache_events: bool
    extra_config: Mapping[str, Any]
    model_name: str
    kv_cache_dtype: str
    namespace_block_size: int
    tp_size: int
    pp_size: int
    pcp_size: int
    dcp_size: int
    rank: int
    use_v2_model_runner: bool
    engine_id: str | None
