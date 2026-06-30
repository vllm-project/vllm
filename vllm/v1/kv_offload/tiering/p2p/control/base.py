# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base classes for the P2P control-plane transport.

The control plane handles peer discovery, connection lifecycle, and
message routing. It is message-content agnostic — it moves opaque
dicts between peers without interpreting them.

Architecture
------------

    ControlTransport (one per node)
        ├── listen for inbound connections
        ├── connect() to outbound peers
        └── poll() → new ControlConnections

    ControlConnection (one per peer)
        ├── send(msg)   — enqueue a message to the peer
        ├── recv()      — drain buffered inbound messages
        ├── mark_dead() — signal that the peer is gone
        └── close()     — tear down the connection

Threading model: all I/O is driven by the caller invoking poll().
No background threads. poll() must be called periodically to:
  - receive messages (buffered per-connection)
  - accept new inbound peers
  - detect disconnections

Implementor contracts
---------------------

- ControlConnection.send() must not block. Messages are serialized
  and queued for the next I/O pass.
- ControlConnection.recv() returns all messages received since the
  last call (may be empty). Messages are dicts (already deserialized).
- ControlConnection.alive returns False after mark_dead() or close().
- ControlTransport.poll() returns newly accepted connections only —
  not previously returned ones. Each connection appears exactly once.
- ControlTransport.connect() creates an outbound connection to a peer
  identified by peer_id (format: "host:port"). Raises on failure.
- Messages may arrive from unknown peers (new inbound connections).
  The transport creates a ControlConnection and returns it from poll()
  with the first message(s) already in its recv() buffer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class ControlConnection(ABC):
    """Bidirectional message channel to a single remote peer.

    Lifecycle:
        1. Created by ControlTransport (connect() or poll())
        2. Used for send/recv by sessions
        3. Marked dead on disconnect (mark_dead())
        4. Cleaned up with close()

    Once mark_dead() is called, alive becomes False and the owning
    session should stop using this connection. close() releases
    underlying resources (sockets, monitors).
    """

    def __init__(self, peer_id: str) -> None:
        self.peer_id = peer_id

    @property
    @abstractmethod
    def alive(self) -> bool:
        """True if the connection is usable. False after mark_dead/close."""
        ...

    @abstractmethod
    def send(self, msg: dict) -> None:
        """Enqueue a message for delivery to the peer.

        Must not block. Serialization happens internally.
        Raises on closed connection.
        """
        ...

    @abstractmethod
    def recv(self) -> Sequence[dict]:
        """Drain and return all messages received since the last call.

        Returns an empty sequence if no messages are pending.
        The returned sequence is read-only — callers must not mutate it.
        Messages are dicts deserialized from the wire format.
        """
        ...

    @abstractmethod
    def mark_dead(self) -> None:
        """Mark this connection as dead (peer disconnected).

        After this call, alive returns False. The session should
        stop using this connection and the transport will clean it up.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources (sockets, monitors).

        Idempotent. After close(), alive returns False.
        """
        ...


class ControlTransport(ABC):
    """Manages peer connections and drives all control-plane I/O.

    Owns the listening socket and all active connections.
    The caller must invoke poll() periodically to process I/O.

    Lifecycle:
        1. Constructed with a local identity and listen address
        2. connect() to reach remote peers
        3. poll() to accept inbound peers and process messages
        4. close() to shut down
    """

    @abstractmethod
    def connect(self, peer_id: str) -> ControlConnection:
        """Create an outbound connection to a remote peer.

        Args:
            peer_id: Remote peer identity (format: "host:port").

        Returns:
            A new ControlConnection ready for send/recv.

        The connection's send queue is live immediately — messages
        sent before the remote peer's poll() will be buffered.
        """
        ...

    @abstractmethod
    def poll(self) -> Sequence[ControlConnection]:
        """Process all pending I/O and return newly accepted connections.

        This is the main I/O driver. Each call:
          - Receives messages from all connected peers (buffered in
            each connection's recv() queue)
          - Accepts new inbound peers and creates connections for them
            (first message already in recv() buffer)
          - Detects disconnections and marks connections dead

        Returns:
            Newly accepted inbound connections (not previously returned).
            The returned sequence is read-only — callers must not mutate
            it. The caller is responsible for creating sessions for
            these connections.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Shut down the transport and all connections.

        Closes the listening socket and all active connections.
        Idempotent.
        """
        ...
