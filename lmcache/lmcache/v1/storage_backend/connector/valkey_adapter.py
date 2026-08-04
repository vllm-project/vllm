# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config_base import _to_bool
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class ValkeyConnectorAdapter(ConnectorAdapter):
    """Adapter for the ValkeyConnector (``valkey://`` scheme).

    Uses the GLIDE sync client with a ThreadPoolExecutor for
    high-throughput KV cache transfer.  Supports both standalone
    (default) and cluster modes via ``valkey_mode`` config.

    Optionally sets a per-key TTL (``valkey_enable_ttl`` + ``valkey_ttl_sec``) so
    Valkey/Redis ``volatile-*`` eviction policies can reclaim L2 cache keys
    under memory pressure.

    Like ``RESPConnectorAdapter``, this adapter uses single-key, fixed-size
    storage with no per-chunk metadata, so partial/unfull chunks are not
    supported: ``save_chunk_meta`` and ``save_unfull_chunk`` must both be
    ``False``.

    Requires ``valkey-glide`` 2.3+.
    """

    def __init__(self) -> None:
        super().__init__("valkey://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a ValkeyConnector from the given context.

        Args:
            context: Connector creation context containing URL, config,
                event loop, and local CPU backend.

        Returns:
            A configured ValkeyConnector instance.
        """
        # Local
        from .valkey_connector import (
            DEFAULT_CONNECTION_TIMEOUT_SECS,
            DEFAULT_REQUEST_TIMEOUT_SECS,
            DEFAULT_TTL_SECS,
            ValkeyConnector,
        )

        config = context.config
        extra_config = (
            config.extra_config
            if config is not None and config.extra_config is not None
            else {}
        )

        # Single-key fixed-size storage (like RESP) carries no per-chunk
        # metadata, so partial/unfull chunks are unsupported. save_chunk_meta
        # comes from free-form extra_config, so coerce it (a string like
        # "false" would otherwise be truthy).
        if _to_bool(extra_config.get("save_chunk_meta", False)):
            raise ValueError("save_chunk_meta must be False for Valkey glide connector")
        if config is not None and config.save_unfull_chunk:
            raise ValueError(
                "save_unfull_chunk must be False for Valkey glide connector"
            )

        num_workers = int(
            extra_config.get(
                "valkey_num_workers",
                extra_config.get("valkey_sync_num_workers", 8),
            )
        )
        username = str(extra_config.get("valkey_username", ""))
        password = str(extra_config.get("valkey_password", ""))
        tls_enable = _to_bool(extra_config.get("tls_enable", False))

        # Timeouts
        request_timeout = float(
            extra_config.get(
                "request_timeout",
                config.blocking_timeout_secs
                if config is not None and config.blocking_timeout_secs is not None
                else DEFAULT_REQUEST_TIMEOUT_SECS,
            )
        )
        connection_timeout = float(
            extra_config.get("connection_timeout", DEFAULT_CONNECTION_TIMEOUT_SECS)
        )

        # Mode: "standalone" (default) or "cluster"
        valkey_mode = str(extra_config.get("valkey_mode", "standalone"))
        cluster_mode = valkey_mode == "cluster"

        # Database ID (standalone only — cluster always uses DB 0)
        database_id: Optional[int] = None
        raw_db = extra_config.get("valkey_database", None)
        if raw_db is not None:
            database_id = int(raw_db)
            if cluster_mode:
                logger.warning(
                    "valkey_database=%s is ignored in cluster mode "
                    "(Valkey cluster always uses DB 0).",
                    database_id,
                )
                database_id = None

        # TTL feature flag (volatile-* eviction support).
        #
        # Without a TTL, keys are persisted indefinitely; a Valkey/Redis node
        # using a ``volatile-lru``/``volatile-lfu`` eviction policy will never
        # evict them and the L2 remote cache chokes once ``maxmemory`` is hit.
        # Enabling ``valkey_enable_ttl`` makes every key expire after
        # ``valkey_ttl_sec`` seconds so the eviction policy can reclaim memory.
        ttl_seconds: Optional[int] = None
        if _to_bool(extra_config.get("valkey_enable_ttl", False)):
            raw_ttl = extra_config.get("valkey_ttl_sec", None)
            if raw_ttl is None:
                ttl_seconds = DEFAULT_TTL_SECS
                logger.warning(
                    "valkey_enable_ttl is enabled but valkey_ttl_sec is not set; "
                    "defaulting key TTL to %d seconds.",
                    ttl_seconds,
                )
            else:
                # bool is a subclass of int, so a boolean (e.g. ``True``)
                # would otherwise coerce to a 1-second TTL — reject it.
                if isinstance(raw_ttl, bool):
                    raise ValueError(
                        f"valkey_ttl_sec must be a positive integer number of "
                        f"seconds, got {raw_ttl!r}."
                    )
                try:
                    ttl_seconds = int(float(raw_ttl))
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"valkey_ttl_sec must be a positive integer number of "
                        f"seconds, got {raw_ttl!r}."
                    ) from e
                if ttl_seconds <= 0:
                    raise ValueError(
                        f"valkey_ttl_sec must be a positive number of seconds, "
                        f"got {ttl_seconds}."
                    )

        parsed_url = parse_remote_url(context.url)
        logger.info(
            "Creating Valkey connector for %s:%d (mode=%s, ttl_seconds=%s)",
            parsed_url.host,
            parsed_url.port,
            valkey_mode,
            ttl_seconds,
        )
        return ValkeyConnector(
            host=parsed_url.host,
            port=parsed_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            num_workers=num_workers,
            username=username,
            password=password,
            request_timeout=request_timeout,
            connection_timeout=connection_timeout,
            tls_enable=tls_enable,
            cluster_mode=cluster_mode,
            database_id=database_id,
            ttl_seconds=ttl_seconds,
        )
