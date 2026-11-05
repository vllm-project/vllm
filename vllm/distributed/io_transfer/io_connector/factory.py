# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.io_transfer.io_connector.base import (
    IOConnectorBase,
    IOConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import IOTransferConfig, VllmConfig

logger = init_logger(__name__)


class IOConnectorFactory:
    """Registry + factory for :class:`IOConnectorBase` implementations.

    Connectors are registered lazily so their modules are only imported
    when the connector is actually instantiated.
    """

    _registry: dict[str, Callable[[], type[IOConnectorBase]]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"IO connector '{name}' is already registered.")

        def loader() -> type[IOConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: IOConnectorRole,
    ) -> IOConnectorBase:
        """Instantiate the configured connector."""
        io_transfer_config = config.io_transfer_config
        if io_transfer_config is None:
            raise ValueError("io_transfer_config must be set to create an IO connector")
        connector_cls = cls.get_connector_class(io_transfer_config)
        logger.info(
            "Creating IO connector %s (engine_id=%s)",
            connector_cls.__name__,
            io_transfer_config.engine_id,
        )
        return connector_cls(config, role)

    @classmethod
    def get_connector_class(
        cls,
        io_transfer_config: "IOTransferConfig",
    ) -> type[IOConnectorBase]:
        """Resolve the connector class by name."""
        connector_name = io_transfer_config.io_connector
        if connector_name is None:
            raise ValueError("io_connector must not be None")
        if connector_name in cls._registry:
            return cls._registry[connector_name]()
        # Fall back to dynamic import via explicit module path.
        module_path = io_transfer_config.io_connector_module_path
        if module_path is None:
            raise ValueError(
                f"Unsupported IO connector type: {connector_name!r}. "
                "Set io_connector_module_path to load a custom connector."
            )
        module = importlib.import_module(module_path)
        return getattr(module, connector_name)


# ---------------------------------------------------------------------------
# Built-in connector registrations
# ---------------------------------------------------------------------------

IOConnectorFactory.register_connector(
    "IOExampleConnector",
    "vllm.distributed.io_transfer.io_connector.example_connector",
    "IOExampleConnector",
)
IOConnectorFactory.register_connector(
    "IOP2PConnector",
    "vllm.distributed.io_transfer.io_connector.p2p_connector",
    "IOP2PConnector",
)
