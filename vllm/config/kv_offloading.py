# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV Cache Offloading Configuration Module.

This module provides a flexible interface for configuring KV cache offloading
to CPU or other tier-2 storage backends. Contributors can add new offloading
backends by implementing KVOffloadingConfigParser and adding them to the
_OFFLOADING_BACKEND_REGISTRY.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.config.parallel import ParallelConfig

logger = init_logger(__name__)


class KVOffloadingConfigParser(ABC):
    """Abstract base class for KV offloading backend handlers.

    Contributors can implement this interface to add support for new
    offloading backends. The handler is responsible for configuring
    the KVTransferConfig based on the offloading size and any additional
    backend-specific parameters.
    """

    @abstractmethod
    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        offloading_size_gib: float,
        cache_config: "CacheConfig",
        parallel_config: "ParallelConfig",
    ) -> None:
        """Configure the KVTransferConfig for this offloading backend.

        Args:
            kv_transfer_config: The KVTransferConfig to modify.
            offloading_size_gib: The total offloading buffer size in GiB
                (summed across all TP ranks when TP > 1).
            cache_config: The CacheConfig containing additional cache settings.
            parallel_config: The ParallelConfig containing parallelism settings.
        """
        raise NotImplementedError

    def get_num_kv_ranks(self, parallel_config: "ParallelConfig"):
        """
        Get number of worker processes that will 'split' the KV caches
        """
        return (
            parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
        )


class NativeOffloadingParser(KVOffloadingConfigParser):
    """Handler for vLLM native CPU offloading backend."""

    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        offloading_size_gib: float,
        cache_config: "CacheConfig",
        parallel_config: "ParallelConfig",
    ) -> None:
        kv_transfer_config.kv_connector = "OffloadingConnector"
        kv_transfer_config.kv_role = "kv_both"

        # Parse the rank information
        num_kv_ranks = super().get_num_kv_ranks(parallel_config)
        kv_bytes_per_rank = offloading_size_gib * (1 << 30) / num_kv_ranks

        if kv_transfer_config.kv_connector_extra_config is None:
            kv_transfer_config.kv_connector_extra_config = {}

        # NOTE(ApostaC): the actual calculation for num_cpu_blocks should be
        # done after the model's KV cache is initialized
        kv_transfer_config.kv_connector_extra_config["kv_bytes_per_rank"] = (
            kv_bytes_per_rank
        )
        kv_transfer_config.kv_connector_extra_config["num_cpu_blocks"] = 0


class LMCacheOffloadingParser(KVOffloadingConfigParser):
    def _get_lmcache_config_dict(self, offloading_gib: float):
        return {"local_cpu": True, "max_local_cpu_size": offloading_gib}

    """Handler for LMCache CPU offloading backend"""

    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        offloading_size_gib: float,
        cache_config: "CacheConfig",
        parallel_config: "ParallelConfig",
    ) -> None:
        kv_transfer_config.kv_connector = "LMCacheConnectorV1"
        kv_transfer_config.kv_role = "kv_both"

        num_kv_ranks = super().get_num_kv_ranks(parallel_config)
        kv_gb_per_rank = offloading_size_gib / num_kv_ranks

        kv_transfer_config.kv_connector_extra_config = self._get_lmcache_config_dict(
            kv_gb_per_rank
        )


# Registry for offloading backend handlers
# Contributors can add new handlers here by implementing KVOffloadingConfigParser
_OFFLOADING_BACKEND_REGISTRY: dict[str, KVOffloadingConfigParser] = {
    "native": NativeOffloadingParser(),
    "lmcache": LMCacheOffloadingParser(),
}


def apply_kv_offloading_config(
    cache_config: "CacheConfig",
    kv_transfer_config: "KVTransferConfig",
    parallel_config: "ParallelConfig",
) -> None:
    """Apply KV offloading configuration to KVTransferConfig.

    This function reads the offloading settings from CacheConfig and
    configures the KVTransferConfig accordingly using the registered
    backend handler.

    Args:
        cache_config: The CacheConfig containing offloading settings.
        kv_transfer_config: The KVTransferConfig to modify.
        parallel_config: The ParallelConfig containing parallelism settings.

    Raises:
        ValueError: If offloading configuration is invalid or backend
            is not registered.
    """
    # Check if offloading is enabled
    if cache_config.kv_offloading_size is None:
        # No offloading configured
        return

    # Validate configuration
    if cache_config.kv_offloading_backend is None:
        available_backends = ", ".join(_OFFLOADING_BACKEND_REGISTRY.keys())
        raise ValueError(
            "kv_offloading_backend must be specified when "
            f"kv_offloading_size is set. Available backends: {available_backends}"
        )

    if cache_config.kv_offloading_size <= 0:
        raise ValueError(
            f"kv_offloading_size must be positive, got "
            f"{cache_config.kv_offloading_size}"
        )

    backend_name = cache_config.kv_offloading_backend

    # Get the handler for this backend
    if backend_name not in _OFFLOADING_BACKEND_REGISTRY:
        available_backends = ", ".join(_OFFLOADING_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown offloading backend: '{backend_name}'. "
            f"Available backends: {available_backends}"
        )

    handler = _OFFLOADING_BACKEND_REGISTRY[backend_name]

    # Apply the configuration
    handler.configure_kv_transfer(
        kv_transfer_config=kv_transfer_config,
        offloading_size_gib=cache_config.kv_offloading_size,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )

    logger.info(
        "KV offloading configured: backend=%s, size=%.2f GiB",
        backend_name,
        cache_config.kv_offloading_size,
    )
