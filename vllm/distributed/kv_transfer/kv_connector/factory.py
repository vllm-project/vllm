# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, cast

from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase,
    KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole,
    supports_hma,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> KVConnectorBase:
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls, compat_sig = cls._get_connector_class_with_compat(
            kv_transfer_config
        )

        # check if the connector supports HMA
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not supports_hma(connector_cls):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

        logger.info(
            "Creating v1 connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            kv_transfer_config.engine_id,
        )
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        if compat_sig:
            # Old signature: __init__(self, vllm_config, role)
            return connector_cls(config, role)
        else:
            # New signature: __init__(self, vllm_config, role, kv_cache_config)
            return connector_cls(config, role, kv_cache_config)

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.

        Args:
            connector_name: Name of the registered connector.

        Returns:
            The connector class.
        """
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    @classmethod
    def _get_connector_class_with_compat(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> tuple[type[KVConnectorBaseType], bool]:
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        compat_sig = False
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            connector_module_path = kv_transfer_config.kv_connector_module_path
            if connector_module_path is None:
                raise ValueError(f"Unsupported connector type: {connector_name}")
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            connector_cls = cast(type[KVConnectorBaseType], connector_cls)
            if not supports_kw(connector_cls, "kv_cache_config"):
                compat_sig = True
                logger.warning(
                    "Connector %s uses deprecated signature with 2 required arguments. "
                    "Please update to include kv_cache_config as the second argument.",
                    connector_cls.__name__,
                )
        return connector_cls, compat_sig

    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        """Get the connector class by name."""
        connector_cls, _ = cls._get_connector_class_with_compat(kv_transfer_config)
        return connector_cls


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector",
)

KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
    "P2pNcclConnector",
)

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1",
)

KVConnectorFactory.register_connector(
    "LMCacheMPConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
    "LMCacheMPConnector",
)

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "NixlConnector",
)

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector",
)

KVConnectorFactory.register_connector(
    "OffloadingConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    "OffloadingConnector",
)

KVConnectorFactory.register_connector(
    "DecodeBenchConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector",
    "DecodeBenchConnector",
)
