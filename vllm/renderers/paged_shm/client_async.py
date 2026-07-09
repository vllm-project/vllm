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
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import zmq
import zmq.asyncio
from torch._prims_common import DeviceLikeType

from .storage import PagedShmStorage

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
        data = response[1] if len(response) > 1 else b""
        if status == b"ERROR":
            raise RuntimeError(f"Server error: {data.decode('utf-8')}")
        if status != b"OK":
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

    def __init__(self, address: str, pin: bool = False, pool_size: int = 4):
        self._pin = pin
        self._address = address
        self._ctx = zmq.asyncio.Context()
        self._pool: asyncio.Queue = asyncio.Queue()
        self._pool_size = pool_size

        for _ in range(pool_size):
            sock = self._init_sock()
            self._pool.put_nowait(sock)

        # Synchronously fetch storage info (one‑off during init)
        info = self._fetch_storage_info_sync(address)
        self._storage = PagedShmStorage(
            size=info["size"],
            block_size=info["block_size"],
            name=info["name"],
            pin=self._pin,
        )

    @staticmethod
    def _fetch_storage_info_sync(address: str) -> dict[str, Any]:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        try:
            sock.connect(address)
            sock.send_multipart([b"get_storage_info"])
            response = sock.recv_multipart()
            if response[0] != b"OK":
                raise RuntimeError(
                    f"Failed to get storage info: server returned {response[0]!r}"
                )
            return json.loads(response[1].decode("utf-8"))
        finally:
            sock.close()
            ctx.term()

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
    # High‑level async methods
    # ------------------------------------------------------------------

    async def write(
        self,
        uuid: str,
        data: bytes | np.ndarray | torch.Tensor,
        use_cache: bool = True,
    ) -> None:
        """Asynchronously write an item to shared memory."""
        if isinstance(data, torch.Tensor):
            size = data.numel() * data.element_size()
        elif isinstance(data, np.ndarray):
            size = data.nbytes
        elif isinstance(data, bytes):
            size = len(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        async with self.write_context(uuid, size, use_cache) as ctx:
            self._storage.write(data, ctx.blocks)

    async def read(
        self, uuid: str, device: DeviceLikeType = "cpu"
    ) -> np.ndarray | torch.Tensor:
        """Asynchronously read an item from shared memory."""
        async with self.read_context(uuid) as ctx:
            if not ctx.blocks:
                raise ValueError(f"Server returned empty block list for uuid '{uuid}'")
            if device == "cpu":
                result = self._storage.read_to_numpy(ctx.size, ctx.blocks)
            else:
                result = self._storage.read_to_tensor(ctx.size, ctx.blocks, device)
        return result

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
        resp = await self._request(b"open_write", payload)
        return json.loads(resp)

    async def close_write(self, uuid: str) -> None:
        """Finalise a write operation."""
        await self._request(b"close_write", uuid)

    async def open_read(self, uuid: str) -> dict[str, Any]:
        """Acquire a read reference and return block list."""
        resp = await self._request(b"open_read", uuid)
        return json.loads(resp)

    async def close_read(self, uuid: str) -> None:
        """Release a read reference."""
        await self._request(b"close_read", uuid)

    async def pin(self, uuid: str) -> None:
        """Pin an item so it is not evicted."""
        await self._request(b"pin", uuid)

    async def unpin(self, uuid: str) -> None:
        """Unpin an item."""
        await self._request(b"unpin", uuid)

    async def delete(self, uuid: str) -> None:
        """Delete an item and free its blocks."""
        await self._request(b"delete", uuid)

    async def get_storage_info(self) -> dict[str, Any]:
        """Return storage metadata."""
        resp = await self._request(b"get_storage_info")
        return json.loads(resp)

    async def get_manager_state(self) -> dict[str, Any]:
        """Return manager statistics."""
        resp = await self._request(b"get_manager_state")
        return json.loads(resp)

    async def get_shm_name(self) -> str:
        """Return the shared memory name."""
        info = await self.get_storage_info()
        return info["name"]

    async def close(self) -> None:
        """Close all async sockets, terminate context,
        and detach from shared memory."""

        # 1. Close all pooled sockets
        while not self._pool.empty():
            try:
                sock = self._pool.get_nowait()
                sock.close()
            except asyncio.QueueEmpty:
                break

        # 2. Destroy context without blocking the event loop
        self._ctx.destroy(linger=0)

        # 3. Detach shared memory
        if hasattr(self, "_storage"):
            self._storage.close()
            logger.debug("Shared memory storage closed.")

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
            # Discard faulty socket
            try:
                sock.close()
            except Exception:
                pass
            raise
        else:
            self._pool.put_nowait(sock)
