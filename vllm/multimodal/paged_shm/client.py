# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory client.

This module provides a ZMQ-based client for the PagedShmServer, enabling
remote processes to read and write large binary data (bytes, NumPy arrays,
PyTorch tensors) to a shared memory pool managed by the server.  The client
automatically handles block allocation, reference counting, and LRU caching
on the server side, while offering a convenient Pythonic interface.

Key Features:
  - **Write and Read**: Transparently write Python objects to shared memory
    and read them back as NumPy arrays or PyTorch tensors (CPU/GPU).
  - **Context Managers**: Simplified write/read contexts that handle allocation,
    commit/rollback, and lock release automatically.
  - **Thread‑Safe**: Maintains a pool of ZMQ REQ sockets for concurrent access
    from multiple threads.
  - **Asynchronous Writes**: Offload data copy and finalization (close_write)
    to a thread pool, overlapping shared memory transfer with ZMQ IPC
    round‑trip latency, improving throughput for large data.
  - **Read Tokens**: The server can generate an opaque token during write
    (`generate_read_token=True`).  This token provides a safe way to read data.
    Alternatively, a client may call `open_read` with a raw UUID, which causes
    the server to generate a new token, increment the global reference count,
    and return both data and the new token.  This is useful for bootstrapping
    the first reader.  Both the UUID and the token paths are supported and
    fully functional.
    Key properties:
      * The server automatically reserves a **read reference** for each token
        at `close_write` time, effectively **pinning** the data in the cache
        until the token is released via `close_read(token)`.
      * The token can be used in `open_read` multiple times (e.g., by
        multiple Tensor Parallelism workers) **without consuming it**; each
        call returns the data blocks.
      * The token **must** be passed to `close_read` exactly once to release
        the reserved read reference and destroy the token.
  - **Atomic Write‑or‑Read**: The `open_write_or_read` method and its context
    manager allow a client to atomically handle a batch of items: missing
    UUIDs are allocated for writing, existing readable items are opened for
    reading, and if any item is still being written, the request can wait
    (with timeout) until all such items become readable.  This is useful for
    scenarios where the client does not know in advance whether an item exists
    and wants to either read or create it in one atomic operation.
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

from .constants import (
    CLOSE_READ,
    CLOSE_WRITE,
    DEBUG_CLEAN,
    DELETE,
    ERROR,
    GET_INFO,
    GET_MANAGER_STATES,
    GET_STORAGE_INFO,
    OK,
    OPEN_READ,
    OPEN_WRITE,
    OPEN_WRITE_OR_READ,
    WAIT_FOR_READABLE,
)
from .storage import PagedShmStorage
from .types import ShmAllocation, ShmWriteRequest

logger = logging.getLogger(__name__)

MAX_POOL_SIZE = 64


class _BaseClient:
    """Base class for ZMQ REQ‑socket communication with the PagedShmServer."""

    _ctx: zmq.Context
    _address: str
    _socket_timeout_ms: int
    _pool: queue.Queue
    _max_pool_size: int

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
            # Convert server error messages to appropriate Python exceptions
            if error_msg.startswith("TimeoutError:"):
                raise TimeoutError(error_msg)
            if error_msg.startswith("MemoryError:"):
                raise MemoryError(error_msg)
            raise RuntimeError(f"Server error: {error_msg}")
        if status != OK:
            raise RuntimeError(f"Unknown server status: {status!r}")

        return data.decode("utf-8")

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
        blocks: list[int] | None = None,
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ):
        self._client = client
        self._uuid = uuid
        self._size = size
        self._use_cache = use_cache
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
            self._client.close_write(self._uuid)
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

    The `uuid_or_token` parameter can be either:
      - A **raw UUID** (the server will generate a new token, increment the
        reference count, and return the token along with the data).
      - A **read token** (the server returns the data without modifying the
        reference count; the token holds a reserved reference).

    The context manager automatically stores the actual read token returned by
    the server and uses it for `close_read` on exit.

    **Important**: (For now, consider it a feature, not a bug.)
    If you provide pre‑obtained `blocks` and `size` (i.e., you
    already have the block list from a previous call), the context manager
    will **not** call `open_read` and therefore will **not** automatically
    call `close_read` on exit.  In such cases, you must manage the read
    reference yourself (e.g., by explicitly calling `client.close_read(token)`
    after the context).
    """

    def __init__(
        self,
        client: "PagedShmClient",
        uuid_or_token: str,
        size: int | None = None,
        blocks: list[int] | None = None,
        timeout: float = 0.0,
    ):
        self._client = client
        self._uuid_or_token = uuid_or_token
        self._timeout = timeout
        self.size = size or 0
        self.blocks = blocks or []
        # This will be set to the actual token after open_read
        self._token_for_close: str | None = None

    def __enter__(self) -> "_ReadContext":
        if self.blocks and self.size > 0:
            # User provided blocks/size – skip open_read; no token to close.
            return self
        items = self._client.open_read(self._uuid_or_token, timeout=self._timeout)
        self.size = items.size
        self.blocks = items.blocks
        # Store the actual read token returned by the server
        self._token_for_close = items.read_token
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close using the actual token (only if we acquired it in __enter__)
        # **Important**:
        # This implies that if the user passes in 'size' and 'blocks',
        # the token will not be closed.
        # For now, consider it a feature, not a bug.
        if self._token_for_close is not None:
            self._client.close_read(self._token_for_close)


# ---------------------------------------------------------------------------
# Public client class
# ---------------------------------------------------------------------------


class PagedShmClientWithoutStorage(_BaseClient):
    """
    ZMQ client without local storage (does not attach to shared memory).
    Useful for administration or light-weight clients that only need to
    send commands.
    """

    def __init__(
        self,
        address: str,
        init_pool_size: int = 4,
        max_pool_size: int = MAX_POOL_SIZE,
        socket_timeout_ms: int = 5000,
    ):
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

    def open_write_or_read(
        self, items: list[ShmWriteRequest], timeout: float = 0.0
    ) -> list[ShmAllocation]:
        """
        Atomically open for reading or writing a batch of items.

        For each item:
          - If the UUID does not exist, it is allocated for writing.
          - If the UUID exists and is readable, it is opened for reading.
          - If the UUID exists but is being written, a read token is generated
            immediately without waiting; the token will be counted in close_write.

        If memory is insufficient for allocating new items, the request is queued
        (if timeout > 0) and retried when space becomes available.

        Returns a list of ShmAllocation objects in the same order as the input
        items.  For newly allocated items, `is_new` is True and the allocation
        includes the blocks and possibly a read token.  For existing items,
        `is_new` is False and a read token is generated if requested.

        The caller is responsible for calling `close_write` on newly allocated
        items (to commit the write) and `close_read` on any returned read tokens
        to release references.  The `is_new` field can be used to distinguish
        the two cases.

        The `timeout` parameter specifies how long to wait for memory to become
        available if allocation fails.  If timeout=0, the call fails immediately
        with MemoryError.  If timeout<0, it waits indefinitely (capped by server
        configuration).
        """
        payload = json.dumps(
            {
                "items": [asdict(item) for item in items],
                "timeout": timeout,
            }
        )
        resp = self._request(OPEN_WRITE_OR_READ, payload)
        resp_dict = json.loads(resp)
        return [ShmAllocation(**a) for a in resp_dict["data"]]

    def close_write(self, uuid: str) -> None:
        """
        Finalise a write operation for the given UUID.
        The server will automatically reserve one read reference for each
        generated read token associated with this UUID, effectively pinning
        the item in the cache until each token is closed.
        """
        payload = json.dumps({"uuid": uuid})
        self._request(CLOSE_WRITE, payload)

    def open_read(self, uuid_or_token: str, timeout: float = 0.0) -> ShmAllocation:
        """
        Acquire a read lock (if UUID) or return data (if token) for an item.

        - If `uuid_or_token` is a **real UUID**, the server will:
            * Generate a new read token.
            * Increment the global reference count (via `manager.open_read`).
            * Return the data blocks along with the newly created token.
        - If `uuid_or_token` is a **read token** (previously obtained), the server:
            * Returns the data blocks without modifying the reference count
              (the token already holds a reserved reference).
            * The token is NOT consumed; it can be reused by multiple readers.

        In both cases, `is_new` is False because the item already existed.
        """
        payload = json.dumps({"uuid": uuid_or_token, "timeout": timeout})
        resp = self._request(OPEN_READ, payload)
        resp_dict = json.loads(resp)
        return ShmAllocation(**resp_dict["data"])

    def close_read(self, token: str) -> None:
        """
        Release a read reference. **Must be called with a read token**.
        The token is destroyed on the server and its reserved read reference
        is released.
        """
        self._request(CLOSE_READ, token)

    def wait_for_readable(self, uuid_or_token: str, timeout: float = 0.0) -> None:
        """Wait for an item to become readable. Does NOT acquire a read lock."""
        payload = json.dumps({"uuid": uuid_or_token, "timeout": timeout})
        self._request(WAIT_FOR_READABLE, payload)

    def delete(self, uuid: str) -> None:
        """Delete an item and free its blocks immediately."""
        self._request(DELETE, uuid)

    def get_storage_info(self) -> dict[str, Any]:
        """Return storage metadata (name, size, block_size, n_block)."""
        resp = self._request(GET_STORAGE_INFO)
        resp_dict = json.loads(resp)
        return resp_dict["data"]

    def get_manager_states(self) -> dict[str, Any]:
        """Return manager statistics (allocations, cache state, etc.)."""
        resp = self._request(GET_MANAGER_STATES)
        return json.loads(resp)

    def get_info(self, uuid_or_token: str) -> dict[str, Any]:
        """
        Return object info for the given UUID or read token.
        (Resolved on the server.)
        """
        resp = self._request(GET_INFO, uuid_or_token)
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


class PagedShmClient(PagedShmClientWithoutStorage):
    """
    Client for the paged shared‑memory storage server.

    Maintains a pool of ZMQ REQ sockets for thread‑safe concurrent access.
    All public operations that require read/write locks are exposed through
    high‑level methods that internally use context managers.
    """

    def __init__(
        self,
        address: str,
        init_pool_size: int = 4,
        max_pool_size: int = MAX_POOL_SIZE,
        socket_timeout_ms: int = 5000,
        pin: bool = False,
        pool_workers: int = 1,
    ):
        super().__init__(
            address=address,
            init_pool_size=init_pool_size,
            max_pool_size=max_pool_size,
            socket_timeout_ms=socket_timeout_ms,
        )

        self._pin = pin

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
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ) -> _WriteContext:
        """Create a context manager for a synchronous write operation."""
        return _WriteContext(
            self,
            uuid,
            size,
            use_cache=use_cache,
            blocks=blocks,
            timeout=timeout,
            generate_read_token=generate_read_token,
        )

    def read_context(
        self,
        uuid_or_token: str,
        size: int | None = None,
        blocks: list[int] | None = None,
        timeout: float = 0.0,
    ) -> _ReadContext:
        """
        Create a context manager for a read operation.

        `uuid_or_token` can be either a raw UUID or a read token.
        - If UUID, the server generates a new token and increments ref_count.
        - If token, the token is reused without changing ref_count.

        The context manager automatically obtains the correct token from the
        server and uses it for the final `close_read` call.

        **Important**: If you provide pre‑obtained `blocks` and `size`, no
        `open_read` is performed and therefore no `close_read` is called
        on exit; you must manage the read reference manually.
        For now, consider it a feature, not a bug.
        """
        return _ReadContext(self, uuid_or_token, size, blocks, timeout)

    # ------------------------------------------------------------------
    # High‑level convenience methods
    # ------------------------------------------------------------------

    def write(
        self,
        uuid: str,
        data: bytes | np.ndarray | torch.Tensor,
        use_cache: bool = True,
        blocks: list[int] | None = None,
        async_write: bool = False,
        timeout: float = 0.0,
        generate_read_token: bool = False,
    ):
        """
        Write an item to the shared memory store.

        If `generate_read_token` is True, the server generates a read token
        and reserves a read reference for it at `close_write` time.
        This token is returned and can be used for reading.

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

            future = self._executor.submit(self._async_write_task, uuid, data, blocks)
            return size, future, token

        # Synchronous path
        with self.write_context(
            uuid,
            size,
            use_cache,
            blocks,
            timeout=timeout,
            generate_read_token=generate_read_token,
        ) as ctx:
            self._storage.write(data, ctx.blocks)
            if generate_read_token:
                return size, ctx.read_token
            return size

    def _async_write_task(self, uuid: str, data, blocks: list[int]):
        """Background task for asynchronous writes."""
        try:
            self._storage.write(data, blocks)
            self.close_write(uuid)
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
        """
        Read an item from shared memory.

        `uuid_or_token` can be either a raw UUID or a read token.
        The method uses `read_context` internally, so it automatically
        handles token generation and cleanup.

        **Important**:
        Unless you provide pre‑obtained `blocks` and `size`, in which
        case you must manage the read reference manually.
        For now, consider it a feature, not a bug.
        """
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
    def get_iterator_numpy(self, uuid_or_token: str, timeout: float = 0.0):
        with self.read_context(uuid_or_token, timeout=timeout) as ctx:
            it = self._storage.get_iterator_numpy(ctx.size, ctx.blocks)()
            yield it

    @contextmanager
    def get_iterator_tensor(self, uuid_or_token: str, timeout: float = 0.0):
        with self.read_context(uuid_or_token, timeout=timeout) as ctx:
            it = self._storage.get_iterator_tensor(ctx.size, ctx.blocks)()
            yield it

    def get_shm_name(self) -> str:
        """Return the shared memory name (cached from initial handshake)."""
        return self._shm_name


def _close_sock_pool(pool: queue.Queue):
    while not pool.empty():
        try:
            sock = pool.get_nowait()
            sock.close(linger=0)
        except queue.Empty:
            break
