# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import queue
import logging
from typing import Union, Optional, Dict, Any, List

import numpy as np
import torch
import zmq
from torch._prims_common import DeviceLikeType

from .storage import PagedShmStorage

logger = logging.getLogger(__name__)


class _BaseClient:
    """Base class for ZMQ REQ‑socket communication with the PagedShmServer."""

    def _build_frames(self, command: bytes, payload: Optional[str] = None) -> List[bytes]:
        """
        Build the multipart message frames for a REQ socket.

        The REQ socket automatically prepends an empty delimiter frame,
        so we only supply the command and an optional payload.
        """
        frames = [command]
        if payload is not None:
            frames.append(payload.encode("utf-8"))
        return frames

    def _parse_response(self, response: List[bytes]) -> str:
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


class PagedShmClient(_BaseClient):
    """
    Client for the paged shared‑memory storage server.

    Internally maintains a pool of ZMQ REQ sockets for thread‑safe
    concurrent access (one socket per thread at a time).

    Parameters
    ----------
    address : str
        IPC address of the server (e.g., "ipc:///tmp/xxx").
    pin : bool
        If True, the client‑side shared memory will be pinned for
        fast GPU direct transfers (requires PIN_MEMORY support).
    init_pool_size : int
        Initial number of ZMQ sockets to pre‑allocate.
    """

    def __init__(self, address: str, pin: bool = False, init_pool_size: int = 4):
        self._pin = pin
        self._address = address
        self._ctx = zmq.Context()
        self._pool = queue.Queue()

        # Pre‑populate the socket pool
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
    # High‑level convenience methods
    # ------------------------------------------------------------------

    def write(
        self,
        uuid: str,
        data: Union[bytes, np.ndarray, torch.Tensor],
        use_cache: bool = True,
    ) -> None:
        """
        Write an item to the shared memory store.

        Steps:
        1. Allocate blocks on the server.
        2. Transfer data directly into the shared memory.
        3. Finalise the write on the server.

        If any step after allocation fails, the allocated blocks are
        automatically cleaned up via a ``delete`` call, preventing
        resource leaks.
        """
        # Calculate data size in bytes
        if isinstance(data, bytes):
            size = len(data)
        elif isinstance(data, np.ndarray):
            size = data.nbytes
        elif isinstance(data, torch.Tensor):
            size = data.numel() * data.element_size()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # 1. Allocate blocks remotely (payload is the item specification dict)
        item_spec = {"uuid": uuid, "size": size, "use_cache": use_cache}
        alloc = self.open_write([item_spec])
        blocks = alloc[0]["blocks"]

        try:
            # 2. Perform the actual data transfer directly into shared memory
            self._storage.write(data, blocks)

            # 3. Finalise the write on the server
            self.close_write(uuid)
        except Exception:
            # If anything goes wrong after allocation, clean up the blocks
            # to avoid leaving the item stuck in an open‑write state.
            try:
                self.delete(uuid)
            except Exception as cleanup_err:
                logger.error(
                    "Failed to clean up blocks for uuid %s after write error: %s",
                    uuid,
                    cleanup_err,
                )
            raise   # Re‑raise the original exception to inform the caller

    def read(
            self,
            uuid: str,
            device: DeviceLikeType = "cpu",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Read an item from the shared memory store.

        Returns a numpy array if ``device="cpu"``, or a torch tensor
        if a GPU device is specified.

        The read lock acquired by ``open_read`` is **always** released
        via ``close_read`` in a ``finally`` block.
        """
        items = self.open_read(uuid)
        try:
            blocks = items.get("blocks")
            if not blocks:
                raise ValueError(f"Server returned empty block list for uuid '{uuid}'")

            size = items["size"]

            if device == "cpu":
                result = self._storage.read_to_numpy(size, blocks)
            else:
                result = self._storage.read_to_tensor(size, blocks, device)
        finally:
            self.close_read(uuid)

        return result

    # ------------------------------------------------------------------
    # Public API – each method maps 1:1 to a server command
    # ------------------------------------------------------------------

    def open_write(self, items: list) -> List[Dict[str, Any]]:
        """Allocate blocks for a batch of items to be written."""
        payload = json.dumps(items)
        resp = self._request(b"open_write", payload)
        return json.loads(resp)

    def close_write(self, uuid: str) -> None:
        """Finalise a write operation for the given UUID."""
        # The server expects the raw UUID string, *not* a JSON‑encoded one
        self._request(b"close_write", uuid)

    def open_read(self, uuid: str) -> Dict[str, Any]:
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

    def get_storage_info(self) -> Dict[str, Any]:
        """Return storage metadata (name, size, block_size, n_block)."""
        resp = self._request(b"get_storage_info")
        return json.loads(resp)

    def get_manager_state(self) -> Dict[str, Any]:
        """Return manager statistics (allocations, cache state, etc.)."""
        resp = self._request(b"get_manager_state")
        return json.loads(resp)

    def get_shm_name(self) -> str:
        """Convenience method that returns only the shared memory name."""
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

    def _request(self, command: bytes, payload: Optional[str] = None) -> str:
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
            # If any communication error occurs, close the faulty socket
            # and do NOT return it to the pool. A new one will be created
            # next time if needed.
            try:
                sock.close()
            except Exception:
                pass
            raise
        else:
            # Only healthy sockets are returned to the pool
            self._pool.put(sock)