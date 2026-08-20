# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory client.

Provides a ZMQ-based client for the PagedShmServer that manages a pool
of REQ sockets for thread-safe concurrent access.  The client attaches
to a shared memory segment and offers high-level read/write operations
with automatic block allocation and lock management via context managers.
"""

import contextlib
import json
import logging
import queue
import weakref
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import zmq

from vllm.config import ModelConfig
from vllm.utils.torch_utils import DeviceLikeType

from .constant import (
    CLOSE_READ,
    CLOSE_WRITE,
    DEBUG_CLEAN,
    DELETE,
    ERROR,
    GET_INFO,
    GET_MANAGER_STATE,
    GET_STORAGE_INFO,
    OK,
    OPEN_READ,
    OPEN_WRITE,
    PIN,
    UNPIN,
    WAIT_WRITE,
)
from .storage import PagedShmStorage
from .types import ShmAllocation, ShmWriteRequest

logger = logging.getLogger(__name__)

MAX_POOL_SIZE = 64


class _BaseClient:
    """Base class for ZMQ REQ‑socket communication with the PagedShmServer."""

    def _build_frames(self, command: bytes, payload: str | None = None) -> list[bytes]:
        """Build multipart message for a REQ socket."""
        frames = [command]
        if payload is not None:
            frames.append(payload.encode("utf-8"))
        return frames

    def _parse_response(self, response: list[bytes]) -> str:
        """
        Parse server response.

        REQ socket receives only application frames, stripped of routing envelope.
        Expected: [status, data_bytes] (data_bytes may be empty).
        """
        if not response:
            raise ConnectionError("Empty response from server")
        if len(response) < 2:
            raise RuntimeError(f"Malformed response: {response}")

        status = response[0]
        data = response[1] if len(response) > 1 else b""

        if status == ERROR:
            error_msg = data.decode("utf-8") if data else "unknown error"
            raise RuntimeError(f"Server error: {error_msg}")
        if status != OK:
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

    def __init__(
        self,
        client: "PagedShmClient",
        uuid: str,
        size: int,
        use_cache: bool,
        open_read: bool = False,
        blocks: list[int] | None = None,
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ):
        self._client = client
        self._uuid = uuid
        self._size = size
        self._use_cache = use_cache
        self.open_read = open_read
        self._timeout = timeout
        self._generate_read_token = generate_read_token
        self.blocks = blocks or []
        self.read_token: str | None = None

    def __enter__(self) -> "_WriteContext":
        if self.blocks:
            return self

        item_spec = ShmWriteRequest(
            uuid=self._uuid,
            size=self._size,
            use_cache=self._use_cache,
            generate_read_token=self._generate_read_token,
        )
        alloc = self._client.open_write([item_spec], timeout=self._timeout)
        self.blocks = alloc[0].blocks
        self.read_token = alloc[0].read_token
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._client.close_write(self._uuid, self.open_read)
        else:
            # Rollback: delete the item immediately; do not close_write first.
            try:
                self._client.delete(self._uuid)
            except Exception as e:
                logger.error(
                    "Failed to clean up blocks for uuid %s after error: %s",
                    self._uuid,
                    e,
                )


class _ReadContext:
    """
    Context manager that acquires a read lock on enter and releases it on exit.
    Exposes ``size`` and ``blocks`` attributes.

    The `uuid` passed to the constructor may be a real UUID or a read token.
    The context manager will preserve the original identifier for use in
    `close_read`, ensuring that tokens are properly consumed on the server.
    """

    def __init__(
        self,
        client: "PagedShmClient",
        uuid: str,
        size: int | None = None,
        blocks: list[int] | None = None,
        timeout: float = 0.0,
    ):
        self._client = client
        self._uuid = uuid  # original identifier (token or UUID)
        self._real_uuid: str | None = None
        self._timeout = timeout
        self.size = size or 0
        self.blocks = blocks or []

    def __enter__(self) -> "_ReadContext":
        if self.blocks and self.size > 0:
            # User provided blocks/size; assume _uuid is the real UUID.
            self._real_uuid = self._uuid
            return self
        items = self._client.open_read(self._uuid, timeout=self._timeout)
        self._real_uuid = items.uuid
        self.size = items.size
        self.blocks = items.blocks
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Use the original identifier so that if it was a token,
        # the server can consume it.
        self._client.close_read(self._uuid)


# ---------------------------------------------------------------------------
# Public client class
# ---------------------------------------------------------------------------


class PagedShmClient(_BaseClient):
    """
    Client for the paged shared‑memory storage server.

    Maintains a pool of ZMQ REQ sockets for thread‑safe concurrent access.
    All public operations that require read/write locks are exposed through
    high‑level methods that internally use context managers.

    Parameters
    ----------
    address : str
        IPC address of the server (e.g., ``"ipc:///tmp/xxx"``).
    pin : bool
        If True, the client‑side shared memory will be pinned for fast GPU transfers.
    init_pool_size : int
        Initial number of ZMQ sockets to pre‑allocate.
    pool_workers : int
        Number of threads in the internal thread pool for asynchronous writes.
    max_pool_size : int
        Maximum number of ZMQ sockets in the pool.
    socket_timeout_ms : int
        Send/receive timeout for ZMQ sockets in milliseconds.
    """

    def __init__(
        self,
        address: str,
        pin: bool = False,
        init_pool_size: int = 4,
        pool_workers: int = 1,
        max_pool_size: int = MAX_POOL_SIZE,
        socket_timeout_ms: int = 5000,
    ):
        self._pin = pin
        self._address = address
        self._max_pool_size = max_pool_size
        self._socket_timeout_ms = socket_timeout_ms

        self._resources = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, self._resources.close)

        self._ctx = zmq.Context()
        self._resources.callback(self._ctx.destroy, linger=0)

        self._pool: queue.Queue = queue.Queue()
        for _ in range(init_pool_size):
            sock = self._init_sock()
            self._pool.put(sock)
        self._resources.callback(_close_sock_pool, self._pool)

        # Thread pool for asynchronous writes
        self._executor: Executor = ThreadPoolExecutor(max_workers=pool_workers)
        self._resources.callback(self._executor.shutdown, wait=True)

        # Retrieve and cache storage metadata
        info = json.loads(self._request(GET_STORAGE_INFO))["data"]
        self._storage = PagedShmStorage(
            size=info["size"],
            block_size=info["block_size"],
            name=info["name"],
            pin=self._pin,
        )
        self._resources.callback(self._storage.close)

        # Cache commonly used metadata
        self._shm_name = info["name"]
        self._storage_size = info["size"]
        self._block_size = info["block_size"]
        self._n_block = info["n_block"]

    @classmethod
    def from_model_config(
        cls, model_config: ModelConfig | None, pin: bool = False
    ) -> "PagedShmClient | None":
        if model_config is None:
            return None
        multimodal_config = model_config.multimodal_config
        if multimodal_config is None or not multimodal_config.is_paged_shm_enabled():
            return None
        return cls(address=multimodal_config.paged_shm_server_address, pin=pin)

    # ------------------------------------------------------------------
    # Context manager factories
    # ------------------------------------------------------------------

    def write_context(
        self,
        uuid: str,
        size: int,
        use_cache: bool = True,
        blocks: list[int] | None = None,
        open_read: bool = False,
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ) -> _WriteContext:
        """Create a context manager for a synchronous write operation."""
        return _WriteContext(
            self,
            uuid,
            size,
            use_cache=use_cache,
            open_read=open_read,
            blocks=blocks,
            timeout=timeout,
            generate_read_token=generate_read_token,
        )

    def read_context(
        self,
        uuid: str,
        size: int | None = None,
        blocks: list[int] | None = None,
        timeout: float = 0.0,
    ) -> _ReadContext:
        """
        Create a context manager for a read operation.

        The `uuid` may be a real UUID or a read token.  The context manager
        will automatically release the read lock on exit; if a token was used,
        it will be consumed on the server.
        """
        return _ReadContext(self, uuid, size, blocks, timeout)

    # ------------------------------------------------------------------
    # High‑level convenience methods
    # ------------------------------------------------------------------

    def write(
        self,
        uuid: str,
        data: bytes | np.ndarray | torch.Tensor,
        use_cache: bool = True,
        blocks: list[int] | None = None,
        open_read: bool = False,
        async_write: bool = False,
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ):
        """
        Write an item to the shared memory store.

        If `generate_read_token` is True, the server will generate a read token
        that can be used to read the item without knowing the UUID.  The token
        is returned as part of the result.

        Returns:
            - If `async_write` is False and `generate_read_token` is False:
                int (size in bytes)
            - If `async_write` is False and `generate_read_token` is True:
                tuple[int, str] (size, token)
            - If `async_write` is True:
                tuple[int, Future, str | None] (size, future, token)
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

        if async_write:
            token = None
            if blocks is None:
                item_spec = ShmWriteRequest(
                    uuid=uuid,
                    size=size,
                    use_cache=use_cache,
                    generate_read_token=generate_read_token,
                )
                alloc = self.open_write([item_spec], timeout=timeout)
                blocks = alloc[0].blocks
                if generate_read_token:
                    token = alloc[0].read_token

            future = self._executor.submit(
                self._async_write_task, uuid, data, blocks, open_read
            )
            return size, future, token

        # Synchronous path
        with self.write_context(
            uuid,
            size,
            use_cache,
            blocks,
            open_read,
            timeout=timeout,
            generate_read_token=generate_read_token,
        ) as ctx:
            self._storage.write(data, ctx.blocks)
            if generate_read_token:
                return size, ctx.read_token
            return size

    def _async_write_task(self, uuid: str, data, blocks: list[int], open_read: bool):
        """Background task for asynchronous writes."""
        try:
            self._storage.write(data, blocks)
            self.close_write(uuid, open_read)
        except Exception:
            # Rollback: delete the item to free blocks
            try:
                self.delete(uuid)
            except Exception as e:
                logger.error(
                    "Failed to clean up blocks for async write uuid %s: %s", uuid, e
                )
            raise  # re-raise original exception

    def read(
        self,
        uuid_or_token: str,
        size: int | None = None,
        blocks: list[int] | None = None,
        device: DeviceLikeType = "cpu",
        timeout: float = 0.0,
    ) -> np.ndarray | torch.Tensor:
        """Read an item from shared memory. Accepts UUID or read token."""
        with self.read_context(uuid_or_token, size, blocks, timeout=timeout) as ctx:
            if ctx.size == 0:
                if device == "cpu":
                    return np.array([])
                else:
                    return torch.tensor([], device=device)

            if not ctx.blocks:
                raise ValueError(
                    f"Server returned empty block list for '{uuid_or_token}'"
                )

            if device == "cpu":
                return self._storage.read_to_numpy(ctx.size, ctx.blocks)
            return self._storage.read_to_tensor(ctx.size, ctx.blocks, device)

    # ------------------------------------------------------------------
    # Iterators with read‑lock protection
    # ------------------------------------------------------------------

    @contextmanager
    def get_iterator_numpy(self, uuid: str, timeout: float = 0.0):
        with self.read_context(uuid, timeout=timeout) as ctx:
            it = self._storage.get_iterator_numpy(ctx.size, ctx.blocks)()
            yield it

    @contextmanager
    def get_iterator_tensor(self, uuid: str, timeout: float = 0.0):
        with self.read_context(uuid, timeout=timeout) as ctx:
            it = self._storage.get_iterator_tensor(ctx.size, ctx.blocks)()
            yield it

    # ------------------------------------------------------------------
    # Public API – each method maps 1:1 to a server command
    # ------------------------------------------------------------------

    def open_write(
        self, items: list[ShmWriteRequest], timeout: float = 0.0
    ) -> list[ShmAllocation]:
        """Allocate blocks for a batch of items to be written."""
        payload = json.dumps(
            {
                "items": [asdict(item) for item in items],
                "timeout": timeout,
            }
        )
        resp = self._request(OPEN_WRITE, payload)
        resp_dict = json.loads(resp)
        return [ShmAllocation(**a) for a in resp_dict["data"]]

    def close_write(self, uuid: str, open_read: bool = False) -> None:
        """Finalise a write operation for the given UUID."""
        payload = json.dumps({"uuid": uuid, "open_read": open_read})
        self._request(CLOSE_WRITE, payload)

    def open_read(self, uuid_or_token: str, timeout: float = 0.0) -> ShmAllocation:
        """
        Acquire a read reference to an item and return its block list.
        Accepts UUID or read token. If a token is used, it is marked as used
        on the server and must be paired with a subsequent close_read.
        """
        payload = json.dumps({"uuid": uuid_or_token, "timeout": timeout})
        resp = self._request(OPEN_READ, payload)
        resp_dict = json.loads(resp)
        return ShmAllocation(**resp_dict["data"])

    def close_read(self, uuid_or_token: str) -> None:
        """
        Release a read reference. Accepts UUID or read token.

        If a read token is provided, the token will be consumed on the server
        (i.e., removed from the token mapping).  The token must have been used
        in a prior open_read call.
        """
        self._request(CLOSE_READ, uuid_or_token)

    def wait_write(self, uuid_or_token: str, timeout: float = 0.0) -> None:
        """Wait for an item to become readable. Does NOT acquire a read lock."""
        payload = json.dumps({"uuid": uuid_or_token, "timeout": timeout})
        self._request(WAIT_WRITE, payload)

    def pin(self, uuid: str) -> None:
        """Pin an item so it is not evicted from the LRU cache."""
        self._request(PIN, uuid)

    def unpin(self, uuid: str) -> None:
        """Unpin an item, allowing it to be evicted if idle."""
        self._request(UNPIN, uuid)

    def delete(self, uuid: str) -> None:
        """Delete an item and free its blocks immediately."""
        self._request(DELETE, uuid)

    def get_storage_info(self) -> dict[str, Any]:
        """Return storage metadata (name, size, block_size, n_block)."""
        resp = self._request(GET_STORAGE_INFO)
        resp_dict = json.loads(resp)
        return resp_dict["data"]

    def get_manager_state(self) -> dict[str, Any]:
        """Return manager statistics (allocations, cache state, etc.)."""
        resp = self._request(GET_MANAGER_STATE)
        return json.loads(resp)

    def get_shm_name(self) -> str:
        """Return the shared memory name (cached from initial handshake)."""
        return self._shm_name

    def get_info(self, uuid: str) -> dict[str, Any]:
        """
        Return object info for the given UUID.
        Also accepts read tokens (they are resolved on the server).
        """
        resp = self._request(GET_INFO, uuid)
        return json.loads(resp)

    def debug_cleanup(self) -> None:
        """
        Send a DEBUG_CLEAN command to the server to forcibly clean up all
        pending waiters and purge tokens. This is only effective if the server
        was started with debug=True; otherwise it raises RuntimeError.
        """
        self._request(DEBUG_CLEAN)

    def close(self) -> None:
        self._resources.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sock(self) -> zmq.Socket:
        """Create and connect a new REQ socket with configured timeouts."""
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._socket_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._socket_timeout_ms)
        sock.connect(self._address)
        return sock

    def _request(self, command: bytes, payload: str | None = None) -> str:
        """
        Send a command to the server and return the decoded response.

        Uses a socket from the pool; if pool is empty, creates a new one.
        Sockets are returned to the pool if the total pool size is below
        `_max_pool_size`, otherwise they are closed.
        """
        try:
            sock = self._pool.get_nowait()
        except queue.Empty:
            sock = self._init_sock()

        try:
            frames = self._build_frames(command, payload)
            sock.send_multipart(frames)
            response = sock.recv_multipart()
            return self._parse_response(response)
        except zmq.ZMQError as e:
            logger.debug("ZMQ error during request: %s", e)
            sock.close(linger=0)
            raise
        except Exception:
            sock.close(linger=0)
            raise
        else:
            if self._pool.qsize() < self._max_pool_size:
                self._pool.put(sock)
            else:
                sock.close(linger=0)


def _close_sock_pool(pool: queue.Queue):
    while not pool.empty():
        try:
            sock = pool.get_nowait()
            sock.close(linger=0)
        except queue.Empty:
            break
