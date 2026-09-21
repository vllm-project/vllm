# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for the EC data-plane transport (NIXL/UCX)."""

from abc import ABC, abstractmethod
from typing import Any


class DataTransport(ABC):
    """Interface wrapping the NIXL data-plane for EC block transfers.

    Peer lifecycle:
      add_remote_peer()  — register a peer and store its dlist handle internally
      remove_remote_peer() — deregister and release the stored handle
      post_read()        — issue a READ using the stored handle for agent_name

    The dlist handle (prepped xfer descriptor list) is an implementation detail
    of the transport and is never exposed to callers.
    """

    @abstractmethod
    def get_agent_metadata(self) -> bytes:
        """Return this agent's serialized NIXL metadata blob."""

    @abstractmethod
    def get_mem_descriptor(self) -> bytes:
        """Return the msgpack-encoded block descriptor list for this agent."""

    @abstractmethod
    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> str:
        """Register a remote peer, prep its READ-source dlist, and return agent_name.

        The dlist handle is stored internally and used by post_read().
        """

    @abstractmethod
    def remove_remote_peer(self, agent_name: str) -> None:
        """Deregister a previously registered remote peer and release its handle."""

    @abstractmethod
    def post_read(
        self,
        local_indices: list[int],
        agent_name: str,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        """Issue a consumer-initiated READ using the stored dlist for agent_name.

        Returns a transfer handle for use with check_xfer_state / release_xfer_handle.
        """

    @abstractmethod
    def get_new_notifs(self) -> dict[str, list[bytes]]:
        """Drain NIXL completion notifications addressed to this agent."""

    @abstractmethod
    def check_xfer_state(self, handle: Any) -> str:
        """Return the NIXL state string: 'DONE', 'PROC', or an error string."""

    @abstractmethod
    def release_xfer_handle(self, handle: Any) -> None:
        """Release a transfer handle returned by post_read."""

    @abstractmethod
    def deregister(self) -> None:
        """Deregister all memory and tear down the NIXL agent."""
