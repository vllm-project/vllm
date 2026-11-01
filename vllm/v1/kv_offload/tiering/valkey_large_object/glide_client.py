# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LargeObjectClient implemented with an RDMA-enabled valkey-glide-sync."""

from collections.abc import Iterator
from contextlib import contextmanager

from glide_sync import (
    ClosingError,
    ConnectionError,
    GlideClient,
    RdmaConfiguration,
    RdmaError,
    RdmaProvider,
    RdmaRegion,
    RdmaWindow,
    TimeoutError,
)

from vllm.v1.kv_offload.tiering.valkey.glide_client import GlideValkeyClient
from vllm.v1.kv_offload.tiering.valkey_large_object.client import (
    ValkeyTransportError,
)

_RDMA_PROVIDERS = {
    "efa-direct": RdmaProvider.EfaDirect,
    "tcp": RdmaProvider.Tcp,
}


# Glide's messages for a region an earlier unanswered transfer closed.
_DEAD_REGION_MESSAGES = ("revoked", "region was closed")


@contextmanager
def _transport_errors() -> Iterator[None]:
    """Map failures that revoke a region to ValkeyTransportError.

    Glide revokes the region on any failure the server did not report, and
    exposes no revoked flag, so the message is the only signal for a region
    revoked by an earlier failure.
    """
    try:
        yield
    except (ClosingError, ConnectionError, TimeoutError) as error:
        raise ValkeyTransportError(str(error)) from error
    except RdmaError as error:
        if any(message in str(error) for message in _DEAD_REGION_MESSAGES):
            raise ValkeyTransportError(str(error)) from error
        raise


class GlideLargeObjectClient(GlideValkeyClient):
    def __init__(
        self,
        startup_nodes: str,
        cluster_mode: bool,
        timeout_ms: int,
        rdma_provider: str,
        rdma_interface: str | None,
    ) -> None:
        """Args:
        startup_nodes: Comma-separated ``host:port`` seed nodes.
        cluster_mode: Connect as a cluster client.
        timeout_ms: Glide request timeout. RDMA transfers ignore it.
        rdma_provider: ``"efa-direct"`` or ``"tcp"`` (emulated, for testing).
        rdma_interface: Fabric domain to pin to, or None for the first.
        """
        if rdma_provider not in _RDMA_PROVIDERS:
            raise ValueError(
                f"rdma_provider must be one of {list(_RDMA_PROVIDERS)}, "
                f"got {rdma_provider!r}"
            )
        if not GlideClient.rdma_usable():
            raise RuntimeError(
                "valkey-glide-sync has no RDMA support or libfabric is missing"
            )
        super().__init__(
            startup_nodes,
            cluster_mode,
            timeout_ms,
            rdma=RdmaConfiguration(
                _RDMA_PROVIDERS[rdma_provider](), interface=rdma_interface
            ),
        )
        self._regions: list[RdmaRegion] = []

    def register_region(self, memory: memoryview) -> int:
        self._regions.append(self._client.register_rdma_region(memory))
        return len(self._regions) - 1

    def _window(self, region: int, offset: int, length: int) -> RdmaWindow:
        return self._regions[region].window(offset, length)

    def rdma_set(self, key: str, region: int, offset: int, length: int) -> None:
        with _transport_errors():
            self._client.rdma_set(key, self._window(region, offset, length))

    def rdma_get(self, key: str, region: int, offset: int, length: int) -> int | None:
        with _transport_errors():
            receipt = self._client.rdma_get(key, self._window(region, offset, length))
        return None if receipt is None else receipt.bytes_written

    def expire(self, key: str, ttl_seconds: int) -> None:
        self._client.expire(key, ttl_seconds)

    def release_regions(self) -> None:
        for region in self._regions:
            region.close()
