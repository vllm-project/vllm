# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, no_type_check
import asyncio
import socket

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.

        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        # loop.sock_recv_into(sock, buf)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.async_socket_lock = asyncio.Lock()

    # TODO(Jiayi): This should be an async function
    def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
        received = 0
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.local_cpu_backend.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        # logger.debug("Call to exists()!")

        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.EXIST,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            response = self.client_socket.recv(ServerMetaMessage.packlength())

        return ServerMetaMessage.deserialize(response).code == ServerReturnCode.SUCCESS

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            logger.warning(f"lm connector failed in exists: {e}")
            return False

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        # logger.debug("Async call to put()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async with self.async_socket_lock:
            await self.loop.sock_sendall(
                self.client_socket,
                ClientMetaMessage(
                    ClientCommand.PUT,
                    key,
                    len(kv_bytes),
                    memory_format,
                    kv_dtype,
                    kv_shape,
                ).serialize(),
            )

            await self.loop.sock_sendall(self.client_socket, kv_bytes)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # NOTE(Jiayi): Not using any await in the following as
        # we don't want to yield control to other tasks which could
        # sacrifice the performance loading to trade the performance of
        # saving
        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            data = self.client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        if meta.code != ServerReturnCode.SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all(meta)

        return memory_obj

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        async with self.async_socket_lock:
            self.client_socket.close()
        logger.info("Closed the lmserver connection")
