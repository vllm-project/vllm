# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING, Callable, Dict, Type

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:
    _registry: Dict[str, Callable[[], Type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> Type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, rank: int, local_rank: int,
                         config: "VllmConfig") -> KVConnectorBase:
        connector_name = config.kv_transfer_config.kv_connector
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
            return connector_cls(rank, local_rank, config)
        if ":" in connector_name:
            # style 2: module_path:class_name
            module_path, class_name = connector_name.split(":")
        else:
            # style 1: connector_name
            module_path = ("vllm.distributed.kv_transfer.kv_connector."
                           f"{connector_name.lower()}_connector")
            class_name = f"{connector_name}Connector"

        try:
            # dynamic import and instantiation
            module = importlib.import_module(module_path)
            connector_cls = getattr(module, class_name)

            # create instance
            return connector_cls(rank, local_rank, config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"cannot create connector '{connector_name}'."
                             f"reason: {str(e)}") from e


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.
KVConnectorFactory.register_connector(
    "PyNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnector",
    "vllm.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnector")

KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.mooncake_store_connector",
    "MooncakeStoreConnector")
