# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import multiprocessing as mp
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict
from queue import PriorityQueue

import zmq

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.network_utils import get_open_zmq_ipc_path

from .constant import (
    CLOSE_READ,
    CLOSE_WRITE,
    DEBUG_CLEAN,
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
    POLL_INTERVAL,
    SHUTDOWN,
    UNPIN,
    WAIT_WRITE,
)
from .manager import PagedShmManager
from .storage import PagedShmStorage
from .types import ShmAllocation, ShmWriteRequest

logger = init_logger(__name__)

# Pre‑coded responses
_OK_RESPONSE = '{"status":"ok"}'
_ERROR_RESPONSE_TEMPLATE = '{"status":"error","message":"%s"}'


class PagedShmServer:
    """Server‑side wrapper that exposes PagedShmManager over ZMQ."""

    def __init__(
        self,
        size: int,
        block_size: int,
        max_timeout: float = 3600.0,
        debug: bool = False,
    ):
        self._resources = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, self._resources.close)

        self.storage = PagedShmStorage(size, block_size, pin=False)
        self._resources.callback(self.storage.close)

        self.manager = PagedShmManager(size, block_size)

        self.size = self.storage.size
        self.shm_name = self.storage.name
        self.n_block = self.storage.n_block
        self.block_size = self.storage.block_size

        # Maximum timeout for requests that ask for infinite waiting
        self.max_timeout = max_timeout

        # Debug mode flag
        self.debug = debug

        # Priority queue for pending open_write requests.
        # Elements: (deadline, identity, list_of_ShmItem)
        self.wait_for_open_write: PriorityQueue = PriorityQueue()

        # Per‑uuid priority queues for pending open_read requests.
        # Keys are item UUIDs; values are PriorityQueues with elements
        # (deadline, identity, token_or_none).
        self.wait_for_open_read: dict[str, PriorityQueue] = {}
        # Set of UUIDs that have at least one pending open_read waiter
        self._open_read_pending: set[str] = set()

        # Per‑uuid priority queues for pending WAIT_WRITE requests.
        self.wait_for_write_completion: dict[str, PriorityQueue] = {}
        # Set of UUIDs that have at least one pending wait_write waiter
        self._wait_write_pending: set[str] = set()

        # Read token storage: token -> real_uuid
        self._read_tokens: dict[str, str] = {}
        # Reverse mapping: real_uuid -> set of tokens
        self._item_to_read_tokens: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Internal helper to get blocks without changing ref_count
    # ------------------------------------------------------------------
    def _get_item_blocks_no_ref(self, uuid: str) -> tuple[int, list[int]]:
        """
        Retrieve size and block list for an item without modifying its
        reference count. This is used for token-based open_read.
        """
        item = self.manager._all_items.get(uuid)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count == -1:  # REF_WRITING
            raise RuntimeError(f"Item {uuid} is still being written")
        return item.size, item.blocks.copy()

    # ------------------------------------------------------------------
    # Generic queue cleanup helper
    # ------------------------------------------------------------------
    def _clean_expired_queue(
        self,
        queue: PriorityQueue,
        socket: zmq.Socket,
        error_msg: str,
    ) -> bool:
        """
        Remove expired entries from a priority queue and send error responses.
        """
        now = time.monotonic()
        while not queue.empty():
            item = queue.queue[0]
            deadline = item[0]
            if deadline > now:
                break
            queue.get()
            identity = item[1]
            self._send_response(socket, identity, ERROR, error_msg)
        return queue.empty()

    # ------------------------------------------------------------------
    # Expiration cleanup methods
    # ------------------------------------------------------------------
    def clean_expired_open_write(self, socket: zmq.Socket) -> None:
        """Clean expired open_write waiters."""
        self._clean_expired_queue(
            self.wait_for_open_write,
            socket,
            "TimeoutError: open_write timed out",
        )

    def clean_expired_open_read(
        self, socket: zmq.Socket, uuid: str | None = None
    ) -> None:
        """
        Clean expired entries from open_read wait queues.
        If uuid is given, only that queue is cleaned; otherwise all pending queues.
        """
        if uuid is not None:
            q = self.wait_for_open_read.get(uuid)
            if q is not None:
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: open_read timed out",
                )
                if empty:
                    self.wait_for_open_read.pop(uuid, None)
                    self._open_read_pending.discard(uuid)
        else:
            # Iterate over a snapshot of pending UUIDs
            for u in list(self._open_read_pending):
                q = self.wait_for_open_read.get(u)
                if q is None:
                    self._open_read_pending.discard(u)
                    continue
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: open_read timed out",
                )
                if empty:
                    self.wait_for_open_read.pop(u, None)
                    self._open_read_pending.discard(u)

    def clean_expired_wait_write(
        self, socket: zmq.Socket, uuid: str | None = None
    ) -> None:
        """
        Clean expired entries from wait_write queues.
        If uuid is given, only that queue is cleaned; otherwise all pending queues.
        """
        if uuid is not None:
            q = self.wait_for_write_completion.get(uuid)
            if q is not None:
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: wait_write timed out",
                )
                if empty:
                    self.wait_for_write_completion.pop(uuid, None)
                    self._wait_write_pending.discard(uuid)
        else:
            for u in list(self._wait_write_pending):
                q = self.wait_for_write_completion.get(u)
                if q is None:
                    self._wait_write_pending.discard(u)
                    continue
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: wait_write timed out",
                )
                if empty:
                    self.wait_for_write_completion.pop(u, None)
                    self._wait_write_pending.discard(u)

    # ------------------------------------------------------------------
    # Debug-only cleanup – forcibly cleans all pending queues and purges
    # tokens. Only available when self.debug is True.
    # ------------------------------------------------------------------
    def debug_cleanup(self, socket: zmq.Socket) -> None:
        """Force‑clean all wait queues and purge tokens (debug only)."""
        if not self.debug:
            raise RuntimeError("debug_cleanup called but debug mode is disabled")

        # 1. Clean all pending open_write waiters
        while not self.wait_for_open_write.empty():
            _, identity, _ = self.wait_for_open_write.get()
            self._send_response(socket, identity, ERROR, "Cleaned by debug_cleanup")

        # 2. Clean all pending open_read waiters
        for u in list(self._open_read_pending):
            q = self.wait_for_open_read.pop(u, None)
            if q is not None:
                while not q.empty():
                    _, identity, _ = q.get()
                    self._send_response(
                        socket, identity, ERROR, "Cleaned by debug_cleanup"
                    )
        self._open_read_pending.clear()

        # 3. Clean all pending wait_write waiters
        for u in list(self._wait_write_pending):
            q = self.wait_for_write_completion.pop(u, None)
            if q is not None:
                while not q.empty():
                    _, identity = q.get()
                    self._send_response(
                        socket, identity, ERROR, "Cleaned by debug_cleanup"
                    )
        self._wait_write_pending.clear()

        # 4. Purge all tokens (forcefully)
        if self._read_tokens:
            logger.warning(
                "debug_cleanup: clearing %d read tokens", len(self._read_tokens)
            )
            self._read_tokens.clear()
        if self._item_to_read_tokens:
            self._item_to_read_tokens.clear()

        # 5. reinit manager
        self.manager = PagedShmManager(self.size, self.block_size)

        logger.info("debug_cleanup completed")

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------
    def open_write(self, data: bytes, identity: bytes) -> str | None:
        """
        Allocate blocks for a batch of items to be written.

        Returns JSON response if satisfied immediately, None if queued.
        Raises MemoryError if timeout=0 or if requested size exceeds total storage.
        """
        write_request = json.loads(data)
        items = write_request["items"]
        timeout = float(write_request.get("timeout", 0.0))

        item_objs = [ShmWriteRequest(**item) for item in items]
        total_required = sum(item.size for item in item_objs)

        # Hard reject if total request exceeds total capacity
        if total_required > self.storage.size:
            raise MemoryError(
                f"Requested {total_required} bytes exceeds total "
                f"storage size {self.storage.size}"
            )

        try:
            allocated = self.manager.open_write(item_objs)
        except MemoryError:
            if timeout == 0.0:
                raise
            # Apply max timeout for infinite waits
            if timeout < 0:
                timeout = self.max_timeout
            deadline = time.monotonic() + timeout
            self.wait_for_open_write.put((deadline, identity, item_objs))
            return None

        return self._build_write_response(allocated, item_objs)

    def open_read(self, data: bytes, identity: bytes) -> str | None:
        """
        Acquire a read reference to an item, returning its block list and size.
        Supports both real UUIDs and read tokens.
        - For UUIDs: increases ref_count via manager.open_read.
        - For tokens: does NOT increase ref_count; returns blocks directly.
        """
        read_request = json.loads(data)
        uuid_or_token = read_request["uuid"]
        timeout = float(read_request.get("timeout", 0.0))

        real_uuid, is_token = self._resolve_read_token(uuid_or_token)

        if is_token:
            # Token must exist and not have been consumed by close_read
            if uuid_or_token not in self._read_tokens:
                raise ValueError(
                    f"Read token '{uuid_or_token}' not found or already consumed"
                )
            # Get the item info without modifying ref_count
            try:
                size, blocks = self._get_item_blocks_no_ref(real_uuid)
            except ValueError as e:
                # If item not found or still being written, delegate to waiting logic
                if "still being written" in str(e):
                    # Wait if timeout allows
                    if timeout == 0.0:
                        raise RuntimeError(
                            f"Item {real_uuid} is still being written"
                        ) from None
                    if timeout < 0:
                        timeout = self.max_timeout
                    deadline = time.monotonic() + timeout
                    q = self.wait_for_open_read.setdefault(real_uuid, PriorityQueue())
                    self._open_read_pending.add(real_uuid)
                    q.put((deadline, identity, uuid_or_token))
                    return None
                else:
                    raise
            # Build response directly
            resp = ShmAllocation(
                uuid=real_uuid, size=size, blocks=blocks, use_cache=True
            )
            return json.dumps({"status": "ok", "data": asdict(resp)})
        else:
            # Normal UUID path – use manager.open_read (increases ref_count)
            try:
                info = self.manager.get_info(real_uuid)
            except ValueError:
                raise
            if info["ref_count"] < 0:  # being written
                if timeout == 0.0:
                    raise RuntimeError(f"Item {real_uuid} is still being written")
                if timeout < 0:
                    timeout = self.max_timeout
                deadline = time.monotonic() + timeout
                q = self.wait_for_open_read.setdefault(real_uuid, PriorityQueue())
                self._open_read_pending.add(real_uuid)
                q.put((deadline, identity, uuid_or_token))
                return None
            return self._open_read(real_uuid)

    def wait_write(self, data: bytes, identity: bytes) -> str | None:
        """
        Wait for an item to become readable. Supports read tokens.
        """
        req = json.loads(data)
        uuid_or_token = req["uuid"]
        timeout = float(req.get("timeout", 0.0))

        real_uuid, _ = self._resolve_read_token(uuid_or_token)

        try:
            info = self.manager.get_info(real_uuid)
        except ValueError:
            raise ValueError(f"UUID {uuid_or_token} not found") from None

        if info["ref_count"] >= 0:
            return _OK_RESPONSE

        # Still being written
        if timeout == 0.0:
            raise RuntimeError(f"Item {uuid_or_token} is still being written")
        if timeout < 0:
            timeout = self.max_timeout
        deadline = time.monotonic() + timeout
        q = self.wait_for_write_completion.setdefault(real_uuid, PriorityQueue())
        self._wait_write_pending.add(real_uuid)
        q.put((deadline, identity))
        return None

    # ------------------------------------------------------------------
    # Deferred request processing (FCFS)
    # ------------------------------------------------------------------
    def defer_open_write(self, socket: zmq.Socket) -> None:
        """
        Attempt to satisfy the first pending open_write request (FCFS).
        If it fails due to MemoryError, leave it for the next iteration.
        """
        if self.wait_for_open_write.empty():
            return

        # Check head without popping
        deadline, identity, item_objs = self.wait_for_open_write.queue[0]
        now = time.monotonic()
        if deadline <= now:
            # Expired – should have been cleaned, but handle defensively
            self.wait_for_open_write.get()
            self._send_response(
                socket, identity, ERROR, "TimeoutError: open_write timed out"
            )
            return

        try:
            allocated = self.manager.open_write(item_objs)
            # Success: pop and respond
            self.wait_for_open_write.get()
            response = self._build_write_response(allocated, item_objs)
            self._send_response(socket, identity, OK, response)
        except MemoryError:
            # Not enough space – keep head and try again later
            pass
        except Exception as e:
            # Other errors (duplicate UUID, etc.) – pop and report
            self.wait_for_open_write.get()
            self._send_response(socket, identity, ERROR, f"{type(e).__name__}: {e}")
            logger.warning("open_write deferred request failed: %s", e)

    def defer_open_read(self, socket: zmq.Socket, uuid: str) -> None:
        """
        Wake up pending open_read waiters for the given UUID.
        Assumes expired entries have already been cleaned.
        """
        q = self.wait_for_open_read.get(uuid)
        if q is None or q.empty():
            self._open_read_pending.discard(uuid)
            return

        # Check if item is now readable
        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            # Item disappeared – send error to all waiters
            while not q.empty():
                _, ident, original_id = q.get()
                self._send_response(socket, ident, ERROR, "UUID not found")
            self.wait_for_open_read.pop(uuid, None)
            self._open_read_pending.discard(uuid)
            return

        if info["ref_count"] >= 0:
            # Satisfy all waiters
            while not q.empty():
                _, identity, original_id = q.get()
                real_uuid, is_token = self._resolve_read_token(original_id)
                try:
                    if is_token:
                        # Token path: do not increase ref_count
                        size, blocks = self._get_item_blocks_no_ref(real_uuid)
                        resp = ShmAllocation(
                            uuid=real_uuid, size=size, blocks=blocks, use_cache=True
                        )
                        self._send_response(
                            socket,
                            identity,
                            OK,
                            json.dumps({"status": "ok", "data": asdict(resp)}),
                        )
                    else:
                        result = self._open_read(real_uuid)
                        self._send_response(socket, identity, OK, result)
                except Exception as e:
                    self._send_response(
                        socket, identity, ERROR, f"{type(e).__name__}: {e}"
                    )
            self.wait_for_open_read.pop(uuid, None)
            self._open_read_pending.discard(uuid)
        # else still being written – keep queue

    def defer_wait_write(self, socket: zmq.Socket, uuid: str) -> None:
        """
        Wake up pending WAIT_WRITE waiters for the given UUID.
        """
        q = self.wait_for_write_completion.get(uuid)
        if q is None or q.empty():
            self._wait_write_pending.discard(uuid)
            return

        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            while not q.empty():
                _, ident = q.get()
                self._send_response(socket, ident, ERROR, "UUID not found")
            self.wait_for_write_completion.pop(uuid, None)
            self._wait_write_pending.discard(uuid)
            return

        if info["ref_count"] >= 0:
            while not q.empty():
                _, identity = q.get()
                self._send_response(socket, identity, OK, _OK_RESPONSE)
            self.wait_for_write_completion.pop(uuid, None)
            self._wait_write_pending.discard(uuid)
        # else still being written

    # ------------------------------------------------------------------
    # Helpers for token resolution and management
    # ------------------------------------------------------------------
    def _resolve_read_token(self, uuid: str) -> tuple[str, bool]:
        """Return (real_uuid, is_token) without modifying state."""
        if uuid in self._read_tokens:
            return self._read_tokens[uuid], True
        return uuid, False

    def _destroy_token(self, token: str) -> None:
        """
        Permanently destroy a read token by removing it from both mappings.
        After this, any further use of the token will be invalid.
        """
        real_uuid = self._read_tokens.pop(token, None)
        if real_uuid is not None:
            token_set = self._item_to_read_tokens.get(real_uuid)
            if token_set is not None:
                token_set.discard(token)
                if not token_set:
                    self._item_to_read_tokens.pop(real_uuid, None)

    def _open_read(self, uuid: str) -> str:
        """Internal helper to open a read reference and build response."""
        item = self.manager.open_read(uuid)
        resp = ShmAllocation(
            uuid=item.uuid,
            size=item.size,
            blocks=item.blocks,
            use_cache=item.use_cache,
        )
        return json.dumps({"status": "ok", "data": asdict(resp)})

    def _build_write_response(
        self, allocated: list, requests: list[ShmWriteRequest]
    ) -> str:
        """Build JSON response for open_write, including read tokens if requested."""
        data = []
        for a, req in zip(allocated, requests):
            token = None
            if req.generate_read_token:
                token = random_uuid()
                self._read_tokens[token] = a.uuid
                self._item_to_read_tokens.setdefault(a.uuid, set()).add(token)
            data.append(
                asdict(
                    ShmAllocation(
                        uuid=a.uuid,
                        size=a.size,
                        blocks=a.blocks,
                        use_cache=a.use_cache,
                        read_token=token,
                    )
                )
            )
        return json.dumps({"status": "ok", "data": data})

    # ------------------------------------------------------------------
    # Other command handlers
    # ------------------------------------------------------------------
    def close_write(self, items_data: bytes) -> str:
        """
        Finish writing an item, making it readable and cacheable.

        Automatically gives one read reference for each generated read token
        associated with the UUID, ensuring the item remains cached as long as
        any token is alive.  The token can be used for multiple open_read calls
        until it is closed via close_read.
        """
        items = json.loads(items_data)
        uuid = items["uuid"]
        tokens = self._item_to_read_tokens.get(uuid, set())
        open_n_reads = len(tokens)
        self.manager.close_write(uuid, open_n_reads)
        return _OK_RESPONSE

    def close_read(self, uuid: str) -> str:
        """
        Release a read reference. Accepts UUID or read token.
        If a token is given, it is destroyed (removed) after releasing the reference.
        If the token has no pre‑incremented reference (ref_count==0), we only destroy
        the token without calling manager.close_read to avoid ValueError.
        """
        real_uuid, is_token = self._resolve_read_token(uuid)
        if is_token:
            # Check if the item exists and has a positive ref_count
            try:
                info = self.manager.get_info(real_uuid)
                if info["ref_count"] > 0:
                    self.manager.close_read(real_uuid)
                # else ref_count == 0, do not call close_read
            except ValueError:
                # Item may have been deleted; just destroy token
                pass
            self._destroy_token(uuid)
        else:
            self.manager.close_read(real_uuid)
        return _OK_RESPONSE

    def pin(self, uuid: str) -> str:
        """Pin an item so it is not evicted from the LRU cache."""
        self.manager.pin(uuid)
        return _OK_RESPONSE

    def unpin(self, uuid: str) -> str:
        """Unpin an item, allowing it to be evicted if idle."""
        self.manager.unpin(uuid)
        return _OK_RESPONSE

    def delete(self, uuid: str) -> str:
        """Delete an item and free its blocks (forcefully)."""
        # Destroy all read tokens associated with this item
        if uuid in self._item_to_read_tokens:
            for token in list(self._item_to_read_tokens[uuid]):
                self._destroy_token(token)
        self.manager.delete(uuid, force=True)
        return _OK_RESPONSE

    def get_manager_state(self) -> str:
        """Return manager statistics as a JSON string."""
        return json.dumps(self.manager.get_manager_state())

    def get_storage_info(self) -> str:
        """Return storage metadata (name, size, block info) as a JSON string."""
        info = {
            "name": self.shm_name,
            "size": self.size,
            "block_size": self.block_size,
            "n_block": self.n_block,
        }
        return json.dumps({"status": "ok", "data": info})

    def get_info(self, uuid: str) -> str:
        """Return object info as a JSON string."""
        real_uuid, _ = self._resolve_read_token(uuid)
        info = self.manager.get_info(real_uuid)
        info.pop("use_cache", None)
        info.pop("blocks", None)
        return json.dumps(info)

    def close(self):
        """Close the shared memory storage."""
        self._resources.close()

    # ------------------------------------------------------------------
    # ZMQ communication helper
    # ------------------------------------------------------------------
    def _send_response(
        self, socket: zmq.Socket, identity: bytes, status: bytes, payload: str
    ) -> None:
        """Send a multipart response. payload is a string (will be encoded)."""
        try:
            socket.send_multipart([identity, EMPTY, status, payload.encode("utf-8")])
        except zmq.ZMQError as e:
            logger.debug("Failed to send response to %s: %s", identity, e)


# ------------------------------------------------------------------
# Server process entry point
# ------------------------------------------------------------------
def _zmq_server(
    size: int, block_size: int, conn, max_timeout: float = 3600.0, debug: bool = False
):
    context = zmq.Context()
    socket = None
    server = None

    try:
        server = PagedShmServer(size, block_size, max_timeout=max_timeout, debug=debug)

        address = get_open_zmq_ipc_path()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(address)

        conn.send(address)
        conn.close()

        # Command dispatcher for simple commands
        handlers: dict[bytes, tuple[Callable, bool]] = {
            CLOSE_WRITE: (server.close_write, True),
            CLOSE_READ: (server.close_read, True),
            PIN: (server.pin, True),
            UNPIN: (server.unpin, True),
            GET_INFO: (server.get_info, True),
            GET_MANAGER_STATE: (server.get_manager_state, False),
            GET_STORAGE_INFO: (server.get_storage_info, False),
        }

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        logger.info("PagedShmServer started at %s", address)

        while True:
            try:
                socks = dict(poller.poll(POLL_INTERVAL))
            except (zmq.ZMQError, KeyboardInterrupt, EOFError):
                break

            if socket not in socks or socks[socket] != zmq.POLLIN:
                # Periodic maintenance
                server.clean_expired_open_write(socket)
                server.clean_expired_open_read(socket)
                server.clean_expired_wait_write(socket)
                server.defer_open_write(socket)
                continue

            # Receive request
            try:
                frames = socket.recv_multipart()
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                logger.error("Error receiving message: %s", e)
                continue

            if len(frames) < 3:
                logger.warning(
                    "Malformed message with %d frames, ignoring", len(frames)
                )
                continue

            identity, delimiter, command, *payloads = frames
            if delimiter != EMPTY:
                logger.warning("Invalid delimiter from %s, ignoring", identity)
                continue

            # SHUTDOWN
            if command == SHUTDOWN:
                logger.info("Received SHUTDOWN from %s, exiting", identity)
                server._send_response(socket, identity, OK, "shutting down")
                break

            # DEBUG_CLEAN – only if debug mode is enabled
            if command == DEBUG_CLEAN:
                if not server.debug:
                    server._send_response(
                        socket, identity, ERROR, "Debug mode not enabled"
                    )
                else:
                    try:
                        server.debug_cleanup(socket)
                        server._send_response(
                            socket, identity, OK, "debug cleanup completed"
                        )
                    except Exception as e:
                        server._send_response(
                            socket, identity, ERROR, f"Debug cleanup failed: {e}"
                        )
                continue

            # OPEN_WRITE
            if command == OPEN_WRITE:
                try:
                    result = server.open_write(payloads[0].decode(), identity)
                    if result is not None:
                        server._send_response(socket, identity, OK, result)
                except MemoryError as e:
                    server._send_response(socket, identity, ERROR, f"MemoryError: {e}")
                except ValueError as e:
                    server._send_response(socket, identity, ERROR, f"ValueError: {e}")
                except Exception as e:
                    logger.exception("Unexpected error in OPEN_WRITE")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            # OPEN_READ
            if command == OPEN_READ:
                try:
                    result = server.open_read(payloads[0].decode(), identity)
                    if result is not None:
                        server._send_response(socket, identity, OK, result)
                except (ValueError, RuntimeError) as e:
                    server._send_response(
                        socket, identity, ERROR, f"{type(e).__name__}: {e}"
                    )
                except Exception as e:
                    logger.exception("Unexpected error in OPEN_READ")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            # WAIT_WRITE
            if command == WAIT_WRITE:
                try:
                    result = server.wait_write(payloads[0].decode(), identity)
                    if result is not None:
                        server._send_response(socket, identity, OK, result)
                except (ValueError, RuntimeError) as e:
                    server._send_response(
                        socket, identity, ERROR, f"{type(e).__name__}: {e}"
                    )
                except Exception as e:
                    logger.exception("Unexpected error in WAIT_WRITE")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            # DELETE (special: clears pending waiters)
            if command == DELETE:
                uuid = payloads[0].decode()
                try:
                    server.delete(uuid)
                    # Clear pending open_read waiters for this uuid
                    q = server.wait_for_open_read.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident, _ = q.get()
                            server._send_response(
                                socket, ident, ERROR, "Item deleted while waiting"
                            )
                        server._open_read_pending.discard(uuid)
                    # Clear wait_write waiters
                    q = server.wait_for_write_completion.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident = q.get()
                            server._send_response(
                                socket, ident, ERROR, "Item deleted while waiting"
                            )
                        server._wait_write_pending.discard(uuid)
                    server._send_response(socket, identity, OK, _OK_RESPONSE)
                    server.defer_open_write(socket)
                except ValueError as e:
                    server._send_response(socket, identity, ERROR, f"ValueError: {e}")
                except Exception as e:
                    logger.exception("Unexpected error in DELETE")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            # Dispatch other commands
            handler_info = handlers.get(command)
            if handler_info is None:
                server._send_response(
                    socket,
                    identity,
                    ERROR,
                    f"Unknown command: {command.decode(errors='replace')}",
                )
                continue

            handler, requires_payload = handler_info
            try:
                if requires_payload:
                    param = payloads[0].decode()
                    result = handler(param)
                else:
                    result = handler()
                if result is not None:
                    server._send_response(socket, identity, OK, result)
                else:
                    server._send_response(socket, identity, OK, _OK_RESPONSE)
            except ValueError as e:
                server._send_response(socket, identity, ERROR, f"ValueError: {e}")
            except Exception as e:
                logger.exception("Unexpected error in command %s", command)
                server._send_response(socket, identity, ERROR, f"Internal error: {e}")

            # After CLOSE_WRITE, wake up waiting readers and waiters for that UUID
            if command == CLOSE_WRITE:
                try:
                    close_req = json.loads(payloads[0].decode())
                    uuid = close_req.get("uuid")
                    if uuid:
                        server.clean_expired_open_read(socket, uuid)
                        server.clean_expired_wait_write(socket, uuid)
                        server.defer_open_read(socket, uuid)
                        server.defer_wait_write(socket, uuid)
                except Exception:
                    # Fall back to global cleanup (already done at top of loop)
                    pass

            # After CLOSE_READ, try to satisfy deferred writes
            if command == CLOSE_READ:
                server.clean_expired_open_write(socket)
                server.defer_open_write(socket)

    except Exception as e:
        logger.exception("Fatal error in zmq_server: %s", e)
    finally:
        # Resource release order: socket, context, then server storage
        if socket is not None:
            with contextlib.suppress(Exception):
                socket.close()
        if context is not None:
            with contextlib.suppress(Exception):
                context.term()
        if server is not None:
            with contextlib.suppress(Exception):
                server.close()
        logger.debug("[shutdown] PagedShmServer stopped")


# ------------------------------------------------------------------
# Process wrapper
# ------------------------------------------------------------------
class PagedShmServerProc:
    def __init__(
        self,
        size: int,
        block_size: int,
        max_timeout: float = 3600.0,
        debug: bool = False,
    ):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        proc = ctx.Process(
            target=_zmq_server,
            args=(size, block_size, child_conn, max_timeout, debug),
        )

        self.proc = proc
        self.address = ""
        self.parent_conn = parent_conn
        self._finalizer = weakref.finalize(self, self.shutdown)

    def start(self, timeout: float = 5.0):
        self.proc.start()
        if not self.parent_conn.poll(timeout):
            self.proc.terminate()
            self.proc.join()
            raise TimeoutError("Server process did not send address within timeout")
        self.address = self.parent_conn.recv()
        self.parent_conn.close()

    def shutdown(self, timeout: float = 5.0):
        if self.proc.is_alive() and self.address:
            try:
                ctx = zmq.Context()
                sock = ctx.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDTIMEO, 1000)
                sock.connect(self.address)
                sock.send(SHUTDOWN)
                sock.close()
                ctx.term()
            except Exception as e:
                logger.debug("Failed to send SHUTDOWN: %s", e)

        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()


# ------------------------------------------------------------------
# Factory function for vLLM integration
# ------------------------------------------------------------------
def maybe_start_paged_shm_server(
    model_config: ModelConfig,
    max_timeout: float = 3600.0,
    debug: bool = False,
) -> PagedShmServerProc | None:
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return None

    if not multimodal_config.is_paged_shm_enabled():
        return None

    paged_shm_server = PagedShmServerProc(
        size=multimodal_config.paged_shm_size,
        block_size=multimodal_config.paged_shm_block_size,
        max_timeout=max_timeout,
        debug=debug,
    )
    paged_shm_server.start()

    multimodal_config.paged_shm_server_address = paged_shm_server.address
    return paged_shm_server
