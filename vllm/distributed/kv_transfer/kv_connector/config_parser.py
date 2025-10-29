# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV Cache Offloading Configuration Module.

This module provides a flexible interface for configuring KV cache offloading
to CPU or other tier-2 storage backends. Contributors can add new offloading
backends by implementing KVOffloadingConfigParser and adding them to the
_CONFIG_PARSER_REGISTRY.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.config.kv_transfer import KVTransferConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.vllm import VllmConfig

logger = init_logger(__name__)


class KVConnectorConfigParser(ABC):
    """Abstract base class to parse update the KVTransferConfig based
    on the VllmConfig.

    Contributors can implement this interface to add support for new
    connector backends. The handler is responsible for configuring
    the KVTransferConfig based on the other configurations in the
    VllmConfig.
    """

    @abstractmethod
    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        vllm_config: "VllmConfig",
    ) -> None:
        """Configure the KVTransferConfig for this offloading backend.

        Args:
            kv_transfer_config: The KVTransferConfig to modify.
            vllm_config: The VllmConfig containing overall configuration.

        Note:
            This method should modify the kv_transfer_config in place.
        """
        raise NotImplementedError

    # Helper functions
    def get_num_kv_ranks(self, vllm_config: "VllmConfig"):
        """
        Get number of worker processes that will 'split' the KV caches
        """
        parallel_config = vllm_config.parallel_config
        return (
            parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
        )


class NativeOffloadingParser(KVConnectorConfigParser):
    """Handler for vLLM native CPU offloading backend."""

    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        vllm_config: "VllmConfig",
    ) -> None:
        assert vllm_config.cache_config.kv_offloading_size is not None
        offloading_size_gib = vllm_config.cache_config.kv_offloading_size

        kv_transfer_config.kv_connector = "OffloadingConnector"
        kv_transfer_config.kv_role = "kv_both"

        # Parse the rank information
        num_kv_ranks = super().get_num_kv_ranks(vllm_config)
        kv_bytes_per_rank = offloading_size_gib * (1 << 30) / num_kv_ranks

        if kv_transfer_config.kv_connector_extra_config is None:
            kv_transfer_config.kv_connector_extra_config = {}

        # NOTE(ApostaC): the actual calculation for num_cpu_blocks should be
        # done after the model's KV cache is initialized
        kv_transfer_config.kv_connector_extra_config["kv_bytes_per_rank"] = (
            kv_bytes_per_rank
        )
        kv_transfer_config.kv_connector_extra_config["num_cpu_blocks"] = 0


class LMCacheOffloadingParser(KVConnectorConfigParser):
    """Handler for LMCache CPU offloading backend"""

    def _get_lmcache_config_dict(self, offloading_gib: float):
        return {"lmcache.local_cpu": True, "lmcache.max_local_cpu_size": offloading_gib}

    def configure_kv_transfer(
        self,
        kv_transfer_config: "KVTransferConfig",
        vllm_config: "VllmConfig",
    ) -> None:
        assert vllm_config.cache_config.kv_offloading_size is not None
        offloading_size_gib = vllm_config.cache_config.kv_offloading_size

        kv_transfer_config.kv_connector = "LMCacheConnectorV1"
        kv_transfer_config.kv_role = "kv_both"

        num_kv_ranks = super().get_num_kv_ranks(vllm_config)
        kv_gb_per_rank = offloading_size_gib / num_kv_ranks

        kv_transfer_config.kv_connector_extra_config = self._get_lmcache_config_dict(
            kv_gb_per_rank
        )


# Registry for offloading backend handlers
_CONFIG_PARSER_REGISTRY: dict[str, KVConnectorConfigParser] = {
    "native": NativeOffloadingParser(),
    "lmcache": LMCacheOffloadingParser(),
}


def get_connector_config_parser(
    vllm_config: "VllmConfig",
) -> KVConnectorConfigParser | None:
    """Get the KVConnectorConfigParser based on the VllmConfig.

    Right now, this function only looks at
    `vllm_config.cache_config.kv_offloading_backend` to determine
    which connector to use for CPU offloading.
    We can extend it for more general purposes in the future.

    Args:
        vllm_config: The VllmConfig containing overall configuration.

    Returns:
        The KVConnectorConfigParser for the specified backend.
        If no backend is specified, returns None.

    Raises:
        ValueError: If the specified backend is not registered.
    """
    cache_config = vllm_config.cache_config
    backend_name = cache_config.kv_offloading_backend

    if backend_name is None:
        return None

    if backend_name not in _CONFIG_PARSER_REGISTRY:
        available_backends = ", ".join(_CONFIG_PARSER_REGISTRY.keys())
        raise ValueError(
            f"Unknown offloading backend: '{backend_name}'. "
            f"Available backends: {available_backends}"
        )

    return _CONFIG_PARSER_REGISTRY[backend_name]


def apply_extra_kv_connector_config(
    vllm_config: "VllmConfig",
    kv_transfer_config: "KVTransferConfig" | None,
) -> KVTransferConfig | None:
    """Apply KV offloading configuration to KVTransferConfig.

    This function reads the offloading settings from CacheConfig and
    configures the KVTransferConfig accordingly using the registered
    backend handler.

    Args:
        cache_config: The CacheConfig containing offloading settings.
        kv_transfer_config: The KVTransferConfig to modify.
        parallel_config: The ParallelConfig containing parallelism settings.

    Returns:
        The modified KVTransferConfig with offloading settings applied.

    Raises:
        ValueError: If offloading configuration is invalid or backend
            is not registered.
    """
    config_parser = get_connector_config_parser(vllm_config)
    if config_parser is None:
        # No connector is configured, return the original
        return kv_transfer_config

    # Initialize KVTransferConfig if not provided
    if kv_transfer_config is None:
        kv_transfer_config = KVTransferConfig()

    # Apply the configuration
    config_parser.configure_kv_transfer(
        kv_transfer_config=kv_transfer_config,
        vllm_config=vllm_config,
    )

    backend_name = vllm_config.cache_config.kv_offloading_backend
    logger.info(
        "KV connector configured with backend: %s",
        backend_name,
    )

    return kv_transfer_config
