# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDMA client surface the valkey-large-object tier depends on.

Kept free of glide imports so the tier and its tests load without an
RDMA-enabled valkey-glide-sync installed.
"""

from typing import Protocol

from vllm.v1.kv_offload.tiering.valkey.client import ValkeyClient


class ValkeyTransportError(Exception):
    """The transport failed and the registered region can no longer be used.

    Raised for a lost connection, a cancelled transfer, or a revoked region.
    The client must be closed and replaced.
    """


class LargeObjectClient(ValkeyClient, Protocol):
    """A ValkeyClient that moves values by RDMA through registered regions.

    A region serves one transfer at a time; callers serialize per region.
    """

    def register_region(self, memory: memoryview) -> int:
        """Register ``memory`` as a region.

        Returns:
            The region's index, passed to every transfer that windows into it.

        """
        ...

    def rdma_set(self, key: str, region: int, offset: int, length: int) -> None:
        """Store ``[offset, offset + length)`` of ``region`` at ``key``."""
        ...

    def rdma_get(self, key: str, region: int, offset: int, length: int) -> int | None:
        """Read ``key`` into ``[offset, offset + length)`` of ``region``.

        Returns:
            Bytes written, or None if the key does not exist.

        """
        ...

    def expire(self, key: str, ttl_seconds: int) -> None: ...

    def release_regions(self) -> None:
        """Deregister every region. Only after every transfer has returned."""
        ...
