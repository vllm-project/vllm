# SPDX-License-Identifier: Apache-2.0
"""ZMQ-based transport implementations for RPC communication.

TODO: Implement ZmqPushTransport and ZmqPullTransport for
LMCacheAsyncLookupClient/Server.
"""

# Standard
from collections import namedtuple
from typing import Any

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc.transport import (
    RpcClientTransport,
    RpcServerTransport,
)
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)

SocketParams = namedtuple("SocketParams", ["socket_path", "rank"])


class ZmqReqRepClientTransport(RpcClientTransport):
    """ZMQ REQ socket transport for synchronous RPC clients.

    Manages multiple REQ sockets (one per server rank) and
    provides send/recv with timeout + automatic socket
    recreation on failure.
    """

    def __init__(
        self,
        socket_params: list[SocketParams],
        timeout_ms: int,
    ):
        self.ctx = get_zmq_context(use_asyncio=False)
        self.socket_params = socket_params
        self.timeout_ms = timeout_ms
        self._world_size = len(socket_params)
        self.encoder = msgspec.msgpack.Encoder()

        self.sockets: list[zmq.Socket] = []
        for params in self.socket_params:
            logger.info(
                "Transport connecting to rank %s with socket path %s",
                params.rank,
                params.socket_path,
            )
            socket = self._create_socket(params)
            self.sockets.append(socket)

    def _create_socket(self, params: SocketParams) -> zmq.Socket:
        """Create and configure a REQ socket."""
        socket = get_zmq_socket(
            self.ctx,
            params.socket_path,
            "ipc",
            zmq.REQ,
            "connect",
        )
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        return socket

    def _recreate_all_sockets(self) -> None:
        """Recreate all sockets after a failure."""
        for rank_idx in range(self._world_size):
            old_socket = self.sockets[rank_idx]
            if old_socket is not None:
                try:
                    old_socket.close(linger=0)
                except zmq.ZMQError as e:
                    logger.warning(
                        "ZMQ error closing old socket for rank %s: %s",
                        rank_idx,
                        e,
                    )
                except AttributeError:
                    pass

            params = self.socket_params[rank_idx]
            logger.info(
                "Recreating socket for rank %s with path %s",
                params.rank,
                params.socket_path,
            )
            self.sockets[rank_idx] = self._create_socket(params)

    def send_and_recv_all(
        self,
        msg: list[Any],
    ) -> list[bytes]:
        """Send msg to all ranks and collect responses.

        Each element of msg is serialized via msgpack before
        sending. On timeout or ZMQ error, recreates all
        sockets and returns an empty list.
        """
        encoded = [self.encoder.encode(m) for m in msg]
        results: list[bytes] = []
        failed_rank = -1
        try:
            for i in range(self._world_size):
                failed_rank = i
                self.sockets[i].send_multipart(encoded, copy=False)

            for i in range(self._world_size):
                failed_rank = i
                resp = self.sockets[i].recv()
                results.append(resp)
        except zmq.Again as e:
            logger.error(
                "Timeout occurred for rank %s, recreating all sockets. Error: %s",
                failed_rank,
                e,
            )
            self._recreate_all_sockets()
            return []
        except zmq.ZMQError as e:
            logger.error(
                "ZMQ error for rank %s: %s, recreating all sockets",
                failed_rank,
                e,
            )
            self._recreate_all_sockets()
            return []

        return results

    @property
    def world_size(self) -> int:
        return self._world_size

    def close(self) -> None:
        for socket in self.sockets:
            try:
                socket.close(linger=0)
            except Exception as e:
                logger.warning("Error closing socket: %s", e)
        try:
            if self.ctx:
                self.ctx.term()
        except Exception as e:
            logger.warning("Error terminating ZMQ context: %s", e)


class ZmqRouterServerTransport(RpcServerTransport):
    """ZMQ ROUTER socket transport for synchronous RPC servers.

    Listens for incoming requests and sends responses back
    using ROUTER socket identity-based routing.
    """

    def __init__(
        self,
        socket_path: str,
        recv_timeout_ms: int = 1000,
    ):
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            "ipc",
            zmq.ROUTER,  # type: ignore[attr-defined]
            "bind",
        )
        self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.socket_path = socket_path

    def recv_request(
        self,
    ) -> tuple[bytes, list[Any]] | None:
        """Receive a request.

        Returns (identity, data_frames) or None on timeout.
        Each data frame is deserialized via msgpack.
        ROUTER socket frames:
          [0] = identity, [1] = empty delimiter, [2:] = data
        """
        try:
            frames = self.socket.recv_multipart(copy=False)
        except zmq.Again:
            return None

        identity = frames[0].bytes
        # frames[1] is the empty delimiter from REQ socket
        raw_frames = frames[2:]
        if len(raw_frames) < 3:
            logger.warning("Malformed request received: not enough frames.")
            return None
        data_frames = [self.decoder.decode(f) for f in raw_frames]
        return (identity, data_frames)

    def send_response(
        self,
        identity: bytes,
        response: bytes,
    ) -> None:
        """Send response back via ROUTER socket."""
        self.socket.send_multipart([identity, b"", response])

    def close(self) -> None:
        self.socket.close(linger=0)
