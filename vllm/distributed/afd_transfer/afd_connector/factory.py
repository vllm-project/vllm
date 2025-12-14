# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for creating AFD connectors based on configuration."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from .base import AFDConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class AFDConnectorFactory:
    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls, rank: int, local_rank: int, config: "VllmConfig"
    ) -> AFDConnectorBase:
        """Create an AFD connector based on the configuration.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFDConfig

        Returns:
            AFDConnectorBase: The created connector instance

        Raises:
            ValueError: If the transport backend is not supported
            ImportError: If required dependencies are not available
        """
        afd_config = config.afd_config
        connector_name = afd_config.afd_connector

        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()
        assert issubclass(connector_cls, AFDConnectorBase)
        return connector_cls(rank, local_rank, config)

    @classmethod
    def get_connector_class(cls, connector_name: str) -> type[AFDConnectorBase]:
        """Get the connector class for a given connector name.

        Args:
            connector_name: The connector name

        Returns:
            type[AFDConnectorBase]: The connector class

        Raises:
            ValueError: If the connector name is not supported
        """
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        return cls._registry[connector_name]()


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

AFDConnectorFactory.register_connector(
    "dummy",
    "vllm.distributed.afd_transfer.afd_connector.dummy_connector",
    "DummyAFDConnector",
)

AFDConnectorFactory.register_connector(
    "p2pconnector",
    "vllm.distributed.afd_transfer.afd_connector.p2p_connector",
    "P2PAFDConnector",
)
