# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    CreateConnector,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class AuditConnectorAdapter(ConnectorAdapter):
    """Adapter for Audit connectors (for debugging and verification)."""

    def __init__(self) -> None:
        super().__init__("audit://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .audit_connector import AuditConnector

        """
        Create an Audit connector. This connector wraps another connector
        and audits all operations.
        
        extra_config:
        - audit_actual_remote_url: The actual remote URL to connect to.
        - audit_calc_checksum: Whether to calculate checksums.
        - audit_verify_checksum: Whether to verify checksums.

        URL format:
        - audit://host:port[?verify=true|false]

        Examples:
        - audit://localhost:8080
        - audit://audit-server.example.com:8080?verify=true
        - audit://127.0.0.1:8080?verify=false
        """
        logger.info(f"Creating Audit connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for audit connector, but got {hosts}"
            )
        if not context.config:
            raise ValueError("Config is not set")

        parse_url = parse_remote_url(context.url)
        # (Deprecated) verify URL parameter will be removed in future versions
        # Use the extra config instead
        verify_param = parse_url.query_params.get("verify", ["false"])[0]
        verify_checksum = verify_param.lower() in ("true", "1", "yes")
        # Get the actual remote URL from the extra config first to keep consistency
        real_url = context.config.extra_config.get(
            "audit_actual_remote_url", context.config.audit_actual_remote_url
        )
        if not real_url:
            raise ValueError(
                "audit_actual_remote_url is not set in the config or extra_config"
            )
        # Store verify_checksum in extra_config if not already set
        if context.config.extra_config is None:
            context.config.extra_config = {}
        if "audit_verify_checksum" not in context.config.extra_config:
            context.config.extra_config["audit_verify_checksum"] = verify_checksum
        connector = CreateConnector(
            real_url,
            context.loop,
            context.local_cpu_backend,
            context.config,
            context.metadata,
        )
        # Metaclass dynamically implements all abstract methods at runtime
        return AuditConnector(connector.getWrappedConnector(), context.config)  # type: ignore[abstract]
