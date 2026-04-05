# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ECTransferConfig, VllmConfig

logger = init_logger(__name__)


class ECConnectorFactory:
    _registry: dict[str, Callable[[], type[ECConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[ECConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: ECConnectorRole,
    ) -> ECConnectorBase:
        ec_transfer_config = config.ec_transfer_config
        if ec_transfer_config is None:
            raise ValueError("ec_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class(ec_transfer_config)
        logger.info(
            "Creating connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            ec_transfer_config.engine_id,
        )
        # Connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        return connector_cls(config, role)

    @classmethod
    def get_connector_class(
        cls, ec_transfer_config: "ECTransferConfig"
    ) -> type[ECConnectorBase]:
        """Get the connector class by name."""
        connector_name = ec_transfer_config.ec_connector
        if connector_name is None:
            raise ValueError("EC connect must not be None")
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            connector_module_path = ec_transfer_config.ec_connector_module_path
            if connector_module_path is None:
                raise ValueError(f"Unsupported connector type: {connector_name}")
            connector_module = importlib.import_module(connector_module_path)
            connector_cls = getattr(connector_module, connector_name)
        return connector_cls


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

ECConnectorFactory.register_connector(
    "ECExampleConnector",
    "vllm.distributed.ec_transfer.ec_connector.example_connector",
    "ECExampleConnector",
)
ECConnectorFactory.register_connector(
    "SHMConnector",
    "vllm.distributed.ec_transfer.ec_connector.shm_connector",
    "SHMConnector",
)
