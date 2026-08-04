# SPDX-License-Identifier: Apache-2.0
"""Abstract base classes for the transfer channel abstraction.

Main classes:
* ``TransferChannelContext`` -- the global singleton that owns the underlying
  transfer engine, maintains the registered L1 memory, and translates L1 addresses
  into transfer-channel-specific addresses.
* ``TransferChannelServer`` -- listens for client connections and exchanges the
  metadata during the handshake.
* ``TransferChannelClient`` -- performs the read operations against a peer's
  memory. Only reads are supported for P2P for now.
"""

# Standard
import abc

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)


class TransferChannelServer(metaclass=abc.ABCMeta):
    """Listens for incoming client connections and exchanges transfer metadata."""

    @abc.abstractmethod
    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        """Creates the server listening on ``listen_url``.

        Args:
            listen_url: The ``host:port`` to bind and listen on (e.g.
                ``0.0.0.0:7600``).
            advertise_url: The ``host:port`` this peer is reachable at and
                announces in its outgoing handshakes.
            l1_memory_desc: Describes the L1 memory region to register.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Free the resources created by the server."""
        raise NotImplementedError


class TransferChannelClient(metaclass=abc.ABCMeta):
    """Performs read operations from a remote peer's memory."""

    @abc.abstractmethod
    def __init__(self, transfer_channel_server_url: str) -> None:
        """Connect to the server at ``transfer_channel_server_url``."""
        raise NotImplementedError

    @abc.abstractmethod
    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Read data from the remote addresses into the local addresses.

        Args:
            local_addresses: addresses in the local L1 buffer.
            remote_addresses: addresses in the remote peer's L1 buffer.
                Must have the same length as ``local_addresses``.

        Returns:
            A task id, to be passed to ``query_read_status``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Query the status of a previously submitted read.

        Args:
            task_id: The task id returned by ``submit_read``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Terminate in-flight requests, deregister from the server, free resources."""
        raise NotImplementedError


class TransferChannelContext(metaclass=abc.ABCMeta):
    """Global singleton that owns the server singleton, and all clients."""

    @abc.abstractmethod
    def get_transfer_channel_server(self) -> TransferChannelServer:
        """Return the transfer channel server singleton."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> TransferChannelClient:
        """Get or create a client that is connected to the peer identified by
        ``peer_advertise_url``.

        Args:
            peer_advertise_url: The ``host:port`` url that can connects to the
                peer.

        Notes:
            For some bi-directional transport libraries (like NIXL), the client
            might be created passively when the current context (server) is
            connected by the remote peer.
            This behavior is per implementation and transparent to the caller.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Discard the client for ``peer_advertise_url`` and free its resources.

        Call this when reads from the peer are no longer needed (e.g. the
        owning L2 adapter is being removed). Any task ids previously returned by
        that client become invalid. A later ``get_transfer_channel_client`` for
        the same peer returns a fresh client.

        Calling this for a peer with no current client does nothing.

        Args:
            peer_advertise_url: The ``host:port`` of the peer, as passed to
                ``get_transfer_channel_client``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        """Translate ``(offset, size)`` L1 objects into the transfer-channel-specific
        addresses.

        Args:
            lmcache_addresses: A list of tuples of the form ``(offset, size)``, where
                ``offset`` is the offset into the L1 buffer, and ``size`` is the
                size of the region.

        Returns:
            A list of transfer-channel-specific addresses that can be used in the
                client read operations. Must have the same length as
                ``lmcache_addresses``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_connected_clients(self) -> int:
        """Return the number of currently connected clients."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Stop and clean up all servers and clients; free context resources."""
        raise NotImplementedError
