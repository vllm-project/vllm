# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class AzureConnectorAdapter(ConnectorAdapter):
    """Adapter for native Azure Blob Storage connectors.

    URL format:
        azure://<container_name>

    Auth and account settings are read from ``config.extra_config``:
        - azure_account_url     (https://<account>.blob.core.windows.net)
        - azure_connection_string
        - azure_account_key
        - azure_sas_token
    If none of the credential fields are set, DefaultAzureCredential is used.
    """

    def __init__(self) -> None:
        super().__init__("azure://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create an :class:`AzureConnector` from the given context.

        Args:
            context: The connector context carrying the URL, event loop,
                local CPU backend, config, and metadata.

        Returns:
            A configured :class:`AzureConnector`.

        Raises:
            ValueError: If ``context.config`` is missing, ``context.metadata``
                is missing, or the URL does not include a container name.
        """
        # Local
        from .azure_connector import AzureConnector

        config = context.config
        if config is None:
            raise ValueError("config is required for AzureConnectorAdapter")
        extra_config = config.extra_config if config.extra_config is not None else {}

        if context.metadata is None:
            raise ValueError("metadata is required for AzureConnector")

        container = context.url.removeprefix("azure://").strip("/")
        if not container:
            raise ValueError(
                "Azure url must include a container: azure://<container_name>"
            )

        logger.info(f"Creating Azure connector for container: {container}")

        return AzureConnector(
            container=container,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            account_url=extra_config.get("azure_account_url"),
            connection_string=extra_config.get("azure_connection_string"),
            account_key=extra_config.get("azure_account_key"),
            sas_token=extra_config.get("azure_sas_token"),
        )
