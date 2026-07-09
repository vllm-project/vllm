# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory client.

Provides a ZMQ-based client for the PagedShmServer that manages a pool
of REQ sockets for thread-safe concurrent access.  The client attaches
to a shared memory segment and offers high-level read/write operations
with automatic block allocation and lock management via context managers.
"""

import json
import logging
import queue
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import zmq
from torch._prims_common import DeviceLikeType

from .storage import PagedShmStorage

logger = logging.getLogger(__name__)


class _BaseClient:
    """Base class for ZMQ REQ‑socket communication with the PagedShmServer."""

    def _build_frames(self, command: bytes, payload: str | None = None) -> list[bytes]:
        """
        Build the multipart message frames for a REQ socket.

        The REQ socket automatically prepends an empty delimiter frame,
        so we only supply the command and an optional payload.
        """
        frames = [command]
        if payload is not None:
            frames.append(payload.encode("utf-8"))
        return frames

    def _parse_response(self, response: list[bytes]) -> str:
        """
        Parse the response from the server.

        The server (ROUTER) replies with:
            [identity, b"", status, data_bytes]

        The REQ socket automatically strips the identity and delimiter frames,
        leaving exactly [status, data_bytes] (or just [status] if no data).
        """
        if not response:
            raise ConnectionError("Empty response from server")

        status = response[0]
        data = response[1] if len(response) > 1 else b""

        if status == b"ERROR":
            raise RuntimeError(f"Server error: {data.decode('utf-8')}")
        if status != b"OK":
            raise RuntimeError(f"Unknown server status: {status!r}")

        return data.decode("utf-8")


# ---------------------------------------------------------------------------
# Internal context managers for write/read locks
# ---------------------------------------------------------------------------


class _WriteContext:
    """
    Context manager that acquires a write lock (allocates blocks) on enter
    and commits (close_write) or rolls back (delete) on exit.
    """

    def __init__(self, client: "PagedShmClient", uuid: str, size: int, use_cache: bool):
        self._client = client
        self._uuid = uuid
        self._size = size
        self._use_cache = use_cache
        self.blocks: list[int] = []

    def __enter__(self) -> "_WriteContext":
        item_spec = {
            "uuid": self._uuid,
            "size": self._size,
            "use_cache": self._use_cache,
        }
        alloc = self._client.open_write([item_spec])
        self.blocks = alloc[0]["blocks"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._client.close_write(self._uuid)
        else:
            # Rollback: release the write lock first, then delete the item
            try:
                self._client.close_write(self._uuid)
            except Exception as e:
                logger.error(
                    "Failed to close_write during rollback for %s: %s",
                    self._uuid,
                    e,
                )
            try:
                self._client.delete(self._uuid)
            except Exception as e:
                logger.error(
                    "Failed to clean up blocks for uuid %s after error: %s",
                    self._uuid,
                    e,
                )
        return False


class _ReadContext:
    """
    Context manager that acquires a read lock on enter and releases it on exit.
    Exposes ``size`` and ``blocks`` attributes for the duration of the block.
    """

    def __init__(self, client: "PagedShmClient", uuid: str):
        self._client = client
        self._uuid = uuid
        self.size: int = 0
        self.blocks: list[int] = []

    def __enter__(self) -> "_ReadContext":
        items = self._client.open_read(self._uuid)
        self.size = items["size"]
        self.blocks = items["blocks"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._client.close_read(self._uuid)
        return False


# ---------------------------------------------------------------------------
# Public client class
# ---------------------------------------------------------------------------


class PagedShmClient(_BaseClient):
    """
    Client for the paged shared‑memory storage server.

    Maintains a pool of ZMQ REQ sockets for thread‑safe concurrent access
    (one socket per thread at a time).  All public operations that require
    read or write locks are now exposed through high‑level methods that
    internally use context managers, guaranteeing correct lock release.

    Parameters
    ----------
    address : str
        IPC address of the server (e.g., ``"ipc:///tmp/xxx"``).
    pin : bool
        If True, the client‑side shared memory will be pinned for
        fast GPU direct transfers (requires ``PIN_MEMORY`` support).
    init_pool_size : int
        Initial number of ZMQ sockets to pre‑allocate.
    """

    def __init__(self, address: str, pin: bool = False, init_pool_size: int = 4):
        self._pin = pin
        self._address = address
        self._ctx = zmq.Context()
        self._pool: queue.Queue = queue.Queue()

        for _ in range(init_pool_size):
            sock = self._init_sock()
            self._pool.put(sock)

        # Retrieve storage metadata and attach to the shared memory segment
        info = json.loads(self._request(b"get_storage_info"))
        self._storage = PagedShmStorage(
            size=info["size"],
            block_size=info["block_size"],
            name=info["name"],
            pin=self._pin,
        )

    # ------------------------------------------------------------------
    # Context manager factories
    # ------------------------------------------------------------------

    def write_context(
        self, uuid: str, size: int, use_cache: bool = True
    ) -> _WriteContext:
        """
        Create a context manager for a write operation.

        The returned context manager allocates blocks on the server upon
        entry and either commits (``close_write``) on normal exit or
        rolls back (``delete``) if an exception occurs.
        """
        return _WriteContext(self, uuid, size, use_cache)

    def read_context(self, uuid: str) -> _ReadContext:
        """
        Create a context manager for a read operation.

        The context manager acquires a read lock on the server upon entry
        and releases it on exit, exposing the data size and block list.
        """
        return _ReadContext(self, uuid)

    # ------------------------------------------------------------------
    # High‑level convenience methods
    # ------------------------------------------------------------------

    def write(
        self,
        uuid: str,
        data: bytes | np.ndarray | torch.Tensor,
        use_cache: bool = True,
    ) -> None:
        """
        Write an item to the shared memory store.

        Uses a write context manager to ensure blocks are allocated, data is
        transferred directly into shared memory, and the write is finalised
        or rolled back atomically.
        """
        # Determine size in bytes
        if isinstance(data, torch.Tensor):
            size = data.numel() * data.element_size()
        elif isinstance(data, np.ndarray):
            size = data.nbytes
        elif isinstance(data, bytes):
            size = len(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        with self.write_context(uuid, size, use_cache) as ctx:
            self._storage.write(data, ctx.blocks)

    def read(
        self, uuid: str, device: DeviceLikeType = "cpu"
    ) -> np.ndarray | torch.Tensor:
        """
        Read an item from the shared memory store.

        Returns a numpy array if ``device="cpu"``, or a torch tensor
        if a GPU device is specified.  The read lock is held for the
        duration of the data copy.
        """
        with self.read_context(uuid) as ctx:
            if not ctx.blocks:
                raise ValueError(f"Server returned empty block list for uuid '{uuid}'")
            if device == "cpu":
                result = self._storage.read_to_numpy(ctx.size, ctx.blocks)
            else:
                result = self._storage.read_to_tensor(ctx.size, ctx.blocks, device)
        return result

    # ------------------------------------------------------------------
    # Iterators with read‑lock protection
    # ------------------------------------------------------------------

    @contextmanager
    def get_iterator_numpy(self, uuid: str, size: int):
        """
        Provide a NumPy iterator over the blocks of an item while holding
        a read lock.

        Usage::

            with client.get_iterator_numpy(uuid, data_size) as it:
                for block_array, valid_length in it:
                    # process block_array (np.ndarray) of length valid_length
        """
        with self.read_context(uuid) as ctx:
            if ctx.size < size:
                raise ValueError(
                    f"Requested size {size} exceeds available data size {ctx.size}"
                )
            it = self._storage.get_iterator_numpy(size, ctx.blocks)()
            yield it

    @contextmanager
    def get_iterator_tensor(self, uuid: str, size: int):
        """
        Provide a PyTorch tensor iterator over the blocks of an item while
        holding a read lock.

        Usage::

            with client.get_iterator_tensor(uuid, data_size) as it:
                for block_tensor, valid_length in it:
                    # process block_tensor (torch.Tensor) of length valid_length
        """
        with self.read_context(uuid) as ctx:
            if ctx.size < size:
                raise ValueError(
                    f"Requested size {size} exceeds available data size {ctx.size}"
                )
            it = self._storage.get_iterator_tensor(size, ctx.blocks)()
            yield it

    # ------------------------------------------------------------------
    # Public API – each method maps 1:1 to a server command
    # ------------------------------------------------------------------

    def open_write(self, items: list) -> list[dict[str, Any]]:
        """Allocate blocks for a batch of items to be written."""
        payload = json.dumps(items)
        resp = self._request(b"open_write", payload)
        return json.loads(resp)

    def close_write(self, uuid: str) -> None:
        """Finalise a write operation for the given UUID."""
        self._request(b"close_write", uuid)

    def open_read(self, uuid: str) -> dict[str, Any]:
        """Acquire a read reference to an item and return its block list."""
        resp = self._request(b"open_read", uuid)
        return json.loads(resp)

    def close_read(self, uuid: str) -> None:
        """Release a read reference for the given UUID."""
        self._request(b"close_read", uuid)

    def pin(self, uuid: str) -> None:
        """Pin an item so it is not evicted from the LRU cache."""
        self._request(b"pin", uuid)

    def unpin(self, uuid: str) -> None:
        """Unpin an item, allowing it to be evicted if idle."""
        self._request(b"unpin", uuid)

    def delete(self, uuid: str) -> None:
        """Delete an item and free its blocks immediately."""
        self._request(b"delete", uuid)

    def get_storage_info(self) -> dict[str, Any]:
        """Return storage metadata (name, size, block_size, n_block)."""
        resp = self._request(b"get_storage_info")
        return json.loads(resp)

    def get_manager_state(self) -> dict[str, Any]:
        """Return manager statistics (allocations, cache state, etc.)."""
        resp = self._request(b"get_manager_state")
        return json.loads(resp)

    def get_shm_name(self) -> str:
        """Return only the shared memory name."""
        return self.get_storage_info()["name"]

    def close(self) -> None:
        """
        Close all ZMQ sockets, terminate the context, and detach from
        the shared memory segment.
        """
        # Close all pooled sockets
        while not self._pool.empty():
            try:
                sock = self._pool.get_nowait()
                sock.close()
            except queue.Empty:
                break
        self._ctx.term()

        # Detach from the shared memory segment
        if hasattr(self, "_storage"):
            self._storage.close()
            logger.debug("Shared memory storage closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sock(self) -> zmq.Socket:
        """Create and connect a new REQ socket to the server."""
        sock = self._ctx.socket(zmq.REQ)
        sock.connect(self._address)
        return sock

    def _request(self, command: bytes, payload: str | None = None) -> str:
        """
        Send a command to the server and return the decoded response string.

        Uses a socket from the pool to allow limited concurrency.
        If the pool is empty, a new socket is created on‑the‑fly.
        """
        try:
            sock = self._pool.get_nowait()
        except queue.Empty:
            # Pool exhausted – create an additional socket
            sock = self._init_sock()

        try:
            frames = self._build_frames(command, payload)
            sock.send_multipart(frames)
            response = sock.recv_multipart()
            return self._parse_response(response)
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            raise
        else:
            self._pool.put(sock)
