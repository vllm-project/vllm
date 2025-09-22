# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import TYPE_CHECKING, Callable

# yapf: disable
from vllm.distributed.kv_transfer.kv_connector import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger

# yapf: enable

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_transfer import KVTransferConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
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
    ) -> KVConnectorBase:
        kv_transfer_config = config.kv_transfer_config
        connector_cls = cls.get_connector_class(kv_transfer_config)
        logger.info("Creating KV connector with name: %s and engine_id: %s",
                    connector_cls.__name__, kv_transfer_config.engine_id)
        # NOTE(Kuntai): KV connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role)

    @classmethod
    def get_connector_class(
            cls,
            kv_transfer_config: "KVTransferConfig") -> type[KVConnectorBase]:
        """Get the connector class by name."""
        connector_name = kv_transfer_config.kv_connector
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            connector_module_path = kv_transfer_config.kv_connector_module_path
            if connector_module_path is None:
                raise ValueError(
                    f"Unsupported connector type: {connector_name}")
            connector_module = importlib.import_module(connector_module_path)
            connector_cls = getattr(connector_module, connector_name)
        return connector_cls


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "vllm.distributed.kv_transfer.kv_connector.shared_storage_connector",
    "SharedStorageConnector")

KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.p2p.p2p_nccl_connector",
    "P2pNcclConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnectorV1")

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.nixl_connector",
    "NixlConnector")

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.multi_connector",
    "MultiConnector")

KVConnectorFactory.register_connector(
    "OffloadingConnector",
    "vllm.distributed.kv_transfer.kv_connector.offloading_connector",
    "OffloadingConnector")
