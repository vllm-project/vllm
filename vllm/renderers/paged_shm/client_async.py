# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Asynchronous paged shared memory client.

Provides an asyncio-based client for the PagedShmServer.  It uses a pool
of async ZMQ REQ sockets for concurrent access and exposes the same
high-level API as the synchronous PagedShmClient, including context
managers for write/read locks and iterators that hold the read lock.
"""

import asyncio
import contextlib
import json
import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import zmq
import zmq.asyncio

from vllm.utils.async_utils import make_async

from .client import PagedShmClient
from .constant import (
    CLOSE_READ,
    DELETE,
    EMPTY,
    ERROR,
    GET_INFO,
    GET_MANAGER_STATE,
    GET_STORAGE_INFO,
    OK,
    OPEN_READ,
    OPEN_WRITE,
    PIN,
    UNPIN,
)

logger = logging.getLogger(__name__)


class _AsyncBaseClient:
    """Base class for async ZMQ REQ‑socket communication with the server."""

    def _build_frames(self, command: bytes, payload: str | None = None) -> list[bytes]:
        """Build multipart message frames for a REQ socket."""
        frames = [command]
        if payload is not None:
            frames.append(payload.encode("utf-8"))
        return frames

    async def _parse_response(self, response: list[bytes]) -> str:
        """Parse the server response (async version)."""
        if not response:
            raise ConnectionError("Empty response from server")
        status = response[0]
        data = response[1] if len(response) > 1 else EMPTY
        if status == ERROR:
            raise RuntimeError(f"Server error: {data.decode('utf-8')}")
        if status != OK:
            raise RuntimeError(f"Unknown server status: {status!r}")
        return data.decode("utf-8")


class _AsyncWriteContext:
    """
    Async context manager for a write operation.
    On enter, blocks are allocated via open_write.
    On normal exit, the write is committed (close_write).
    On exception, the write lock is released and the item is deleted.
    """

    def __init__(
        self, client: "AsyncPagedShmClient", uuid: str, size: int, use_cache: bool
    ):
        self._client = client
        self._uuid = uuid
        self._size = size
        self._use_cache = use_cache
        self.blocks: list[int] = []

    async def __aenter__(self) -> "_AsyncWriteContext":
        item_spec = {
            "uuid": self._uuid,
            "size": self._size,
            "use_cache": self._use_cache,
        }
        alloc = await self._client.open_write([item_spec])
        self.blocks = alloc[0]["blocks"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            await self._client.close_write(self._uuid)
        else:
            # Rollback: release write lock, then delete item
            try:
                await self._client.close_write(self._uuid)
            except Exception as e:
                logger.error(
                    "Failed to close_write during rollback for %s: %s",
                    self._uuid,
                    e,
                )
            try:
                await self._client.delete(self._uuid)
            except Exception as e:
                logger.error(
                    "Failed to clean up blocks for uuid %s after error: %s",
                    self._uuid,
                    e,
                )
        return False


class _AsyncReadContext:
    """
    Async context manager for a read operation.
    Holds a read lock while active and provides ``size`` and ``blocks``.
    """

    def __init__(self, client: "AsyncPagedShmClient", uuid: str):
        self._client = client
        self._uuid = uuid
        self.size: int = 0
        self.blocks: list[int] = []

    async def __aenter__(self) -> "_AsyncReadContext":
        items = await self._client.open_read(self._uuid)
        self.size = items["size"]
        self.blocks = items["blocks"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self._client.close_read(self._uuid)
        return False


# ---------------------------------------------------------------------------
# Public async client class
# ---------------------------------------------------------------------------


class AsyncPagedShmClient(_AsyncBaseClient):
    """
    Asynchronous client for the paged shared‑memory storage server.

    Maintains a pool of async ZMQ REQ sockets for concurrent access.
    All public operations are ``async`` and internally manage read/write
    locks via async context managers.

    Parameters
    ----------
    address : str
        IPC address of the server.
    pin : bool
        If True, the client‑side shared memory is pinned for GPU direct
        transfers (requires ``PIN_MEMORY`` support).
    pool_size : int
        Maximum number of concurrent ZMQ sockets to keep in the pool.
    """

    def __init__(
        self, address: str, pin: bool = False, pool_size: int = 4, pool_workers: int = 1
    ):
        self._pin = pin
        self._address = address
        self._ctx = zmq.asyncio.Context()
        self._pool: asyncio.Queue = asyncio.Queue()
        self._pool_size = pool_size

        for _ in range(pool_size):
            sock = self._init_sock()
            self._pool.put_nowait(sock)

        self.sync_client = PagedShmClient(address, pin, pool_size)

        self._storage = self.sync_client._storage
        self._executor: Executor = ThreadPoolExecutor(max_workers=pool_workers)

        self.write = make_async(self.sync_client.write, executor=self._executor)
        self.read = make_async(self.sync_client.read, executor=self._executor)

    # ------------------------------------------------------------------
    # Context manager factories
    # ------------------------------------------------------------------

    def write_context(
        self, uuid: str, size: int, use_cache: bool = True
    ) -> _AsyncWriteContext:
        """Create an async context manager for a write operation."""
        return _AsyncWriteContext(self, uuid, size, use_cache)

    def read_context(self, uuid: str) -> _AsyncReadContext:
        """Create an async context manager for a read operation."""
        return _AsyncReadContext(self, uuid)

    # ------------------------------------------------------------------
    # Async iterators with read‑lock protection
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def get_iterator_numpy(self, uuid: str):
        """
        Async context manager providing a NumPy iterator over the blocks.
        The read lock is held during the whole iteration.
        """
        async with self.read_context(uuid) as ctx:
            it = self._storage.get_iterator_numpy(ctx.size, ctx.blocks)()
            yield it

    @asynccontextmanager
    async def get_iterator_tensor(self, uuid: str):
        """
        Async context manager providing a tensor iterator over the blocks.
        The read lock is held during the whole iteration.
        """
        async with self.read_context(uuid) as ctx:
            it = self._storage.get_iterator_tensor(ctx.size, ctx.blocks)()
            yield it

    # ------------------------------------------------------------------
    # Async version of the public API
    # ------------------------------------------------------------------

    async def open_write(self, items: list) -> list[dict[str, Any]]:
        """Allocate blocks for a batch of items to be written."""
        payload = json.dumps(items)
        resp = await self._request(OPEN_WRITE, payload)
        return json.loads(resp)

    async def close_write(self, uuid: str) -> None:
        """Finalise a write operation."""
        await self._request(b"close_write", uuid)

    async def open_read(self, uuid: str) -> dict[str, Any]:
        """Acquire a read reference and return block list."""
        resp = await self._request(OPEN_READ, uuid)
        return json.loads(resp)

    async def close_read(self, uuid: str) -> None:
        """Release a read reference."""
        await self._request(CLOSE_READ, uuid)

    async def pin(self, uuid: str) -> None:
        """Pin an item so it is not evicted."""
        await self._request(PIN, uuid)

    async def unpin(self, uuid: str) -> None:
        """Unpin an item."""
        await self._request(UNPIN, uuid)

    async def delete(self, uuid: str) -> None:
        """Delete an item and free its blocks."""
        await self._request(DELETE, uuid)

    async def get_storage_info(self) -> dict[str, Any]:
        """Return storage metadata."""
        resp = await self._request(GET_STORAGE_INFO)
        return json.loads(resp)

    async def get_manager_state(self) -> dict[str, Any]:
        """Return manager statistics."""
        resp = await self._request(GET_MANAGER_STATE)
        return json.loads(resp)

    async def get_shm_name(self) -> str:
        """Return the shared memory name."""
        info = await self.get_storage_info()
        return info["name"]

    async def get_info(self, uuid: str) -> dict[str, Any]:
        """Return object info."""
        resp = await self._request(GET_INFO, uuid)
        return json.loads(resp)

    async def close(self) -> None:
        """Close all async sockets, terminate context,
        and close sync client."""

        # 1. Close all pooled sockets
        while not self._pool.empty():
            try:
                sock = self._pool.get_nowait()
                sock.close()
            except asyncio.QueueEmpty:
                break

        # 2. Destroy context without blocking the event loop
        self._ctx.destroy(linger=0)

        # 3. close sync client
        self.sync_client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sock(self) -> zmq.asyncio.Socket:
        """Create and connect a new async REQ socket."""
        sock = self._ctx.socket(zmq.REQ)
        sock.connect(self._address)
        return sock

    async def _request(self, command: bytes, payload: str | None = None):
        """
        Send a command to the server and return the decoded response.
        Uses an async socket from the pool; creates a new one if pool is
        empty (up to ``pool_size``).  Corrupted sockets are replaced.
        """
        try:
            sock = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Pool exhausted – create a new socket if within limits
            sock = self._init_sock()

        try:
            frames = self._build_frames(command, payload)
            await sock.send_multipart(frames)
            response = await sock.recv_multipart()
            return await self._parse_response(response)
        except Exception:
            with contextlib.suppress(Exception):
                sock.close()
            raise
        else:
            self._pool.put_nowait(sock)
