# SPDX-License-Identifier: Apache-2.0
"""Abstract transport interfaces for RPC communication.

These interfaces decouple the lookup client/server business logic
from the underlying communication mechanism (e.g., ZMQ, gRPC).

The transport layer is responsible for serialization/deserialization
of structured data, so that upper-layer business logic only works
with Python objects (no raw bytes).

TODO: Implement async transport interfaces for
LMCacheAsyncLookupClient/Server.
"""

# Standard
from typing import Any
import abc


class RpcClientTransport(abc.ABC):
    """Abstract transport for RPC client-side communication.

    Handles sending requests to multiple server ranks and
    collecting responses. The transport is responsible for
    connection management, retries, and error recovery.
    """

    @abc.abstractmethod
    def send_and_recv_all(
        self,
        msg: list[Any],
    ) -> list[bytes]:
        """Send structured data to all ranks and collect
        responses.

        The transport is responsible for serializing each
        element of msg before sending and deserializing
        responses.

        Args:
            msg: List of Python objects to send as
                message frames. Each element will be
                serialized by the transport's codec.

        Returns:
            List of raw response bytes, one per rank.

        Raises:
            RpcTransportError: If communication fails
                after retries.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def world_size(self) -> int:
        """Return the number of server ranks."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the transport and release resources."""
        raise NotImplementedError


class RpcServerTransport(abc.ABC):
    """Abstract transport for RPC server-side communication.

    Handles receiving requests and sending responses back.
    """

    @abc.abstractmethod
    def recv_request(
        self,
    ) -> tuple[bytes, list[Any]] | None:
        """Receive a request from a client.

        The transport deserializes each data frame into
        a Python object before returning.

        Returns:
            A tuple of (identity, data_frames) on success
            where data_frames contains deserialized Python
            objects, or None on timeout / no data available.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def send_response(
        self,
        identity: bytes,
        response: bytes,
    ) -> None:
        """Send a response back to the client.

        Args:
            identity: The client identity from recv_request.
            response: The raw response bytes to send.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the transport and release resources."""
        raise NotImplementedError
