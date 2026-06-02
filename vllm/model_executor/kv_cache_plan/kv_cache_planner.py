# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planner interface."""

from abc import ABC, abstractmethod

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)


class KVCachePlanner(ABC):
    """Plan model-specific KV cache layouts."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config

    @abstractmethod
    def get_kv_cache_configs(
        self,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        """Return per-worker KV cache configs."""
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Return planned KV cache groups for a worker spec map."""
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """Return one KV cache config from already-planned groups."""
        raise NotImplementedError

    @abstractmethod
    def get_max_model_len_capacity(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> int:
        """Return the maximum model length supported by a KV cache layout."""
        raise NotImplementedError
