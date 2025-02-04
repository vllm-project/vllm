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
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()
        return connector_cls(rank, local_rank, config)


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
