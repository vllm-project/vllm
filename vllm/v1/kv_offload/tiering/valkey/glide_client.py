# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ValkeyClient implemented with valkey-glide-sync."""

from typing import Any

from glide_sync import (
    Batch,
    ClusterBatch,
    ExpirySet,
    ExpiryType,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    NodeAddress,
)


def parse_startup_nodes(startup_nodes: str) -> list[NodeAddress]:
    """Parse ``"host:port,host:port"`` into glide node addresses."""
    addresses = []
    for node in startup_nodes.split(","):
        host, _, port = node.strip().rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"startup node {node!r} is not host:port")
        addresses.append(NodeAddress(host, int(port)))
    return addresses


class GlideValkeyClient:
    """Thread-safe: one instance is shared by every pool thread."""

    def __init__(
        self,
        startup_nodes: str,
        cluster_mode: bool,
        timeout_ms: int,
        **configuration_options: Any,
    ) -> None:
        """Args:
        startup_nodes: Comma-separated ``host:port`` seed nodes.
        cluster_mode: Connect as a cluster client.
        timeout_ms: Glide request timeout.
        configuration_options: Extra glide client configuration fields.
        """
        addresses = parse_startup_nodes(startup_nodes)
        self._client: GlideClient | GlideClusterClient
        self._batch_type: type[Batch] | type[ClusterBatch]
        if cluster_mode:
            self._client = GlideClusterClient.create(
                GlideClusterClientConfiguration(
                    addresses, request_timeout=timeout_ms, **configuration_options
                )
            )
            self._batch_type = ClusterBatch
        else:
            self._client = GlideClient.create(
                GlideClientConfiguration(
                    addresses, request_timeout=timeout_ms, **configuration_options
                )
            )
            self._batch_type = Batch

    def set(self, key: str, value: memoryview, ttl_seconds: int | None) -> None:
        expiry = None if ttl_seconds is None else ExpirySet(ExpiryType.SEC, ttl_seconds)
        self._client.set(key, value, expiry=expiry)

    def get_into(self, key: str, buffer: memoryview) -> int | None:
        # With a buffer, glide returns the byte count as a decimal string.
        written = self._client.get(key, buffer=buffer)
        return None if written is None else int(written)

    def exists_each(self, keys: list[str]) -> list[bool]:
        # One EXISTS per key: a multi-key EXISTS returns only a count.
        batch = self._batch_type(is_atomic=False)
        for key in keys:
            batch.exists([key])
        results = self._client.exec(batch, raise_on_error=True)  # type: ignore[arg-type]
        assert results is not None
        return [bool(result) for result in results]

    def close(self) -> None:
        self._client.close()
