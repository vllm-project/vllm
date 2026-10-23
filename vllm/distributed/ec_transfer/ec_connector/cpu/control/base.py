# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base classes for the EC control plane.

ControlConnection wraps a single peer channel (one DEALER socket).
"""

from abc import ABC, abstractmethod


class ControlConnection(ABC):
    """Bidirectional message channel to a single remote peer.

    All I/O is non-blocking: send enqueues without blocking; recv drains
    all buffered inbound messages since the last call.
    """

    @property
    @abstractmethod
    def alive(self) -> bool:
        """True if the connection is still usable."""

    @abstractmethod
    def send(self, msg: bytes) -> None:
        """Enqueue msg for delivery. Must not block."""

    @abstractmethod
    def recv(self) -> list[bytes]:
        """Drain and return all buffered inbound messages."""

    @abstractmethod
    def mark_dead(self) -> None:
        """Signal that the peer disconnected. Sets alive to False."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources. Idempotent."""
