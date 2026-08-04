# SPDX-License-Identifier: Apache-2.0

# TODO(baoloongmao): This module contains the control plane implementation
# for socket-based channels. Currently, it provides the base class with
# ZMQ-based handshake and initialization logic. In the future, this can
# be extended to a complete channel with data plane using native sockets.

# Standard
from typing import Optional, Union
import asyncio
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

logger = init_logger(__name__)


class PySocketMsgBase(msgspec.Struct, tag=True):
    """Base class for all py-socket-related messages"""

    pass


class PySocketInitRequest(PySocketMsgBase):
    """Initialization request, peer_init_url is used as peer identifier"""

    peer_init_url: str


class PySocketInitResponse(PySocketMsgBase):
    """Initialization response"""

    status: str


PySocketMsg = Union[
    PySocketInitRequest,
    PySocketInitResponse,
]


class PySocketChannel(BaseTransferChannel):
    """
    Base class for socket-based transfer channels.

    Control plane: Uses ZMQ sockets for handshake and initialization.
    Data plane: To be implemented by subclasses.

    Provides common control plane logic for different channel types.
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        self.role = kwargs["role"]
        self.buffer_ptr = kwargs["buffer_ptr"]
        self.buffer_size = kwargs["buffer_size"]
        self.align_bytes = kwargs["align_bytes"]
        self.tp_rank = kwargs["tp_rank"]

        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.remote_connections: dict[str, dict] = {}

        self.side_channel: Optional[zmq.Socket] = None
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self._init_side_channels()

    ############################################################
    # Control plane: Initialization functions
    ############################################################

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        raise NotImplementedError("Sync mode not supported in PySocketChannel")

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Initialize connection to a peer using ZMQ sockets for handshake.

        Note: peer_id is expected to be peer_init_url in this implementation.
        """
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        init_req = PySocketInitRequest(peer_init_url=self.peer_init_url)
        await init_tmp_socket.send(msgspec.msgpack.encode(init_req))

        init_resp_bytes = await init_tmp_socket.recv()
        _ = msgspec.msgpack.decode(init_resp_bytes, type=PySocketMsg)

        self.remote_connections[peer_id] = {
            "peer_init_url": peer_init_url,
        }

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def _init_side_channels(self):
        """Initialize side channel for handling incoming connections"""
        if self.peer_init_url is None:
            return

        self.side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[PySocketMsg, InitSideMsgBase]
    ) -> Union[PySocketMsg, InitSideRetMsgBase]:
        """Handle initialization messages from peers"""
        resp: Union[PySocketMsg, InitSideRetMsgBase]
        if isinstance(req, PySocketInitRequest):
            peer_url = req.peer_init_url

            self.remote_connections[peer_url] = {
                "peer_init_url": peer_url,
            }

            self._on_peer_connected(peer_url)

            resp = PySocketInitResponse(status="ok")
            logger.info("Replying initialization response")

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _on_peer_connected(self, peer_url: str):
        """Hook for subclasses to perform additional setup when a peer connects"""
        pass

    def _init_loop(self):
        """Synchronous initialization loop for handling incoming connections"""
        while self.running:
            try:
                req_bytes = self.side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                self.side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        """Asynchronous initialization loop for handling incoming connections"""
        logger.info("Starting async initialization loop")

        while self.running:
            try:
                req_bytes = await self.side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                await self.side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    await asyncio.sleep(0.01)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        """Close all sockets and cleanup resources"""
        self.running = False
        for thread in self.running_threads:
            thread.join()

        if self.side_channel is not None:
            self.side_channel.close()

        self.zmq_context.term()
