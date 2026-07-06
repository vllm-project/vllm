# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from vllm.config.artifact_transfer import ArtifactTransferConfig
from vllm.distributed.artifact_transfer.artifact_connector.base import (
    ArtifactConnectorBase,
    ArtifactConnectorBaseType,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1 import (
    ArtifactConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ArtifactConnectorFactory:
    _registry: dict[str, Callable[[], type[ArtifactConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[ArtifactConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: ArtifactConnectorRole,
    ) -> ArtifactConnectorBase:
        artifact_transfer_config = config.artifact_transfer_config
        if artifact_transfer_config is None:
            raise ValueError(
                "artifact_transfer_config must be set to create a connector"
            )
        connector_cls = cls.get_connector_class(artifact_transfer_config)
        logger.info(
            "Creating artifact connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            artifact_transfer_config.engine_id,
        )
        return connector_cls(config, role)

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[ArtifactConnectorBaseType]:
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    @classmethod
    def get_connector_class(
        cls, artifact_transfer_config: ArtifactTransferConfig
    ) -> type[ArtifactConnectorBaseType]:
        connector_name = artifact_transfer_config.artifact_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in ArtifactTransferConfig")

        connector_module_path = artifact_transfer_config.artifact_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError(
                "artifact_connector_module_path cannot be an empty string."
            )

        if connector_module_path:
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            connector_cls = cast(type[ArtifactConnectorBaseType], connector_cls)
            if not supports_kw(connector_cls, "role"):
                msg = (
                    f"Connector {connector_cls.__name__} must accept role as "
                    "a constructor argument and pass it to super().__init__()."
                )
                logger.error(msg)
                raise ValueError(msg)
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls


ArtifactConnectorFactory.register_connector(
    "DummyArtifactConnector",
    "vllm.distributed.artifact_transfer.artifact_connector.v1.dummy_connector",
    "DummyArtifactConnector",
)

ArtifactConnectorFactory.register_connector(
    "TransferQueueArtifactConnector",
    "vllm.distributed.artifact_transfer.artifact_connector.v1.transfer_queue_connector",
    "TransferQueueArtifactConnector",
)
