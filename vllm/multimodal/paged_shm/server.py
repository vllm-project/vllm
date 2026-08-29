# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import heapq
import json
import multiprocessing as mp
import time
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import zmq

from vllm import envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.network_utils import get_open_zmq_ipc_path

from .constants import (
    CLEANUP_INTERVAL,
    CLOSE_READ,
    CLOSE_WRITE,
    DEBUG_CLEAN,
    DELETE,
    EMPTY,
    ERROR,
    GET_INFO,
    GET_MANAGER_STATES,
    GET_STORAGE_INFO,
    OK,
    OPEN_READ,
    OPEN_WRITE,
    OPEN_WRITE_OR_READ,
    POLL_INTERVAL,
    SHUTDOWN,
    WAIT_FOR_READABLE,
)
from .manager import PagedShmManager
from .storage import PagedShmStorage
from .types import ShmAllocation, ShmWriteRequest

logger = init_logger(__name__)

_OK_RESPONSE = '{"status":"ok"}'
_ERROR_RESPONSE_TEMPLATE = '{"status":"error","message":"%s"}'


# ------------------------------------------------------------------
# Priority queue for single‑threaded use (heapq‑based)
# ------------------------------------------------------------------
class PriorityQueue:
    """
    Simple priority queue with min‑heap ordering.
    Not thread‑safe – designed for single‑threaded event loops.
    """

    def __init__(self):
        self._heap: list = []

    def put(self, item: Any) -> None:
        heapq.heappush(self._heap, item)

    def get(self) -> Any:
        return heapq.heappop(self._heap)

    def empty(self) -> bool:
        return len(self._heap) == 0

    def qsize(self) -> int:
        return len(self._heap)

    def peek(self) -> Any | None:
        return self._heap[0] if self._heap else None

    def clean_expired(self, now: float, on_expired: Callable[[Any], None]) -> bool:
        """
        Remove all items whose deadline <= now, calling on_expired for each.
        Returns True if the queue becomes empty.
        """
        while self._heap and self._heap[0][0] <= now:
            item = heapq.heappop(self._heap)
            on_expired(item)
        return not self._heap


# ------------------------------------------------------------------
# PagedShmServer – core server logic
# ------------------------------------------------------------------
class PagedShmServer:
    """
    Server‑side wrapper that exposes PagedShmManager over ZMQ.
    """

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

        # Maximum timeout for infinite‑wait requests
        self.max_timeout = max_timeout

        # Debug mode flag
        self.debug = debug

        # Wait queues for deferred requests
        # Memory allocation waits: (deadline, identity, req_type, payload)
        self._memory_waiters = PriorityQueue()

        # Per‑uuid open_read waits: (deadline, identity, original_id)
        self._open_read_waiters: dict[str, PriorityQueue] = defaultdict(PriorityQueue)

        # Per‑uuid wait_for_readable waits: (deadline, identity)
        self._readable_waiters: dict[str, PriorityQueue] = defaultdict(PriorityQueue)

        # Read token storage: token -> real_uuid
        self._read_tokens: dict[str, str] = {}
        # Reverse mapping: real_uuid -> set of tokens
        self._item_to_read_tokens: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Expiration cleanup
    # ------------------------------------------------------------------
    def _clean_expired_memory_waiters(self, socket: zmq.Socket) -> None:
        """Remove expired entries from the memory wait queue."""
        if self._memory_waiters.empty():
            return

        def send_error(item):
            _, identity, _, _ = item
            self._send_response(
                socket, identity, ERROR, "TimeoutError: memory allocation timed out"
            )

        self._memory_waiters.clean_expired(time.monotonic(), send_error)

    def clean_expired_open_read(
        self, socket: zmq.Socket, uuid: str | None = None
    ) -> None:
        """Clean expired open_read waiters for a specific UUID or all."""
        now = time.monotonic()

        def send_error(item):
            _, identity, _ = item
            self._send_response(
                socket, identity, ERROR, "TimeoutError: open_read timed out"
            )

        if uuid is not None:
            q = self._open_read_waiters.get(uuid)
            if q is not None and not q.empty() and q.clean_expired(now, send_error):
                del self._open_read_waiters[uuid]
        else:
            for u in list(self._open_read_waiters.keys()):
                q = self._open_read_waiters[u]
                if q.clean_expired(now, send_error):
                    del self._open_read_waiters[u]

    def clean_expired_wait_readable(
        self, socket: zmq.Socket, uuid: str | None = None
    ) -> None:
        """Clean expired wait_for_readable waiters."""
        now = time.monotonic()

        def send_error(item):
            _, identity = item
            self._send_response(
                socket, identity, ERROR, "TimeoutError: wait_for_readable timed out"
            )

        if uuid is not None:
            q = self._readable_waiters.get(uuid)
            if q is not None and not q.empty() and q.clean_expired(now, send_error):
                del self._readable_waiters[uuid]
        else:
            for u in list(self._readable_waiters.keys()):
                q = self._readable_waiters[u]
                if q.clean_expired(now, send_error):
                    del self._readable_waiters[u]

    # ------------------------------------------------------------------
    # Periodic maintenance
    # ------------------------------------------------------------------
    def _perform_maintenance(self, socket: zmq.Socket) -> None:
        """Periodic cleanup and deferred request processing."""
        self._clean_expired_memory_waiters(socket)
        self.clean_expired_open_read(socket)
        self.clean_expired_wait_readable(socket)
        self._defer_memory_requests(socket)

    # ------------------------------------------------------------------
    # Debug‑only forced cleanup
    # ------------------------------------------------------------------
    def debug_cleanup(self, socket: zmq.Socket) -> None:
        """Force‑clean all wait queues and purge tokens (debug only)."""
        if not self.debug:
            raise RuntimeError("debug_cleanup called but debug mode is disabled")

        # 1. Clear memory waiters
        def send_error(item):
            _, identity, _, _ = item
            self._send_response(socket, identity, ERROR, "Cleaned by debug_cleanup")

        self._memory_waiters.clean_expired(time.monotonic(), send_error)

        # 2. Clear open_read waiters
        for u, q in list(self._open_read_waiters.items()):

            def send_open_error(item):
                _, identity, _ = item
                self._send_response(socket, identity, ERROR, "Cleaned by debug_cleanup")

            q.clean_expired(time.monotonic(), send_open_error)
            del self._open_read_waiters[u]

        # 3. Clear readable waiters
        for u, q in list(self._readable_waiters.items()):

            def send_readable_error(item):
                _, identity = item
                self._send_response(socket, identity, ERROR, "Cleaned by debug_cleanup")

            q.clean_expired(time.monotonic(), send_readable_error)
            del self._readable_waiters[u]

        # 4. Purge read tokens
        if self._read_tokens:
            logger.warning(
                "debug_cleanup: clearing %d read tokens", len(self._read_tokens)
            )
            self._read_tokens.clear()
        if self._item_to_read_tokens:
            self._item_to_read_tokens.clear()

        # 5. Reinitialize manager
        self.manager = PagedShmManager(self.size, self.block_size)
        logger.info("debug_cleanup completed")

    # ------------------------------------------------------------------
    # Core logic for open_write_or_read
    # ------------------------------------------------------------------
    def _core_write_or_read(self, items_data: list[dict]) -> str:
        """Atomically allocate missing items and read existing ones."""
        to_allocate = []
        to_read = []
        pending_writes = []

        for item in items_data:
            uuid = item["uuid"]
            try:
                info = self.manager.get_info(uuid)
                if info["ref_count"] == -1:
                    pending_writes.append(item)
                else:
                    to_read.append(item)
            except ValueError:
                to_allocate.append(item)

        allocated_slots = []
        if to_allocate:
            write_requests = [
                ShmWriteRequest(
                    uuid=item["uuid"],
                    size=item["size"],
                    use_cache=item.get("use_cache", True),
                    generate_read_token=item.get("generate_read_token", False),
                )
                for item in to_allocate
            ]
            allocated_slots = self.manager.open_write(write_requests)

        result_map = {}

        # New items
        for slot, req_item in zip(allocated_slots, to_allocate):
            token = None
            if req_item.get("generate_read_token", False):
                token = random_uuid()
                self._read_tokens[token] = slot.uuid
                self._item_to_read_tokens.setdefault(slot.uuid, set()).add(token)
            result_map[slot.uuid] = {
                "uuid": slot.uuid,
                "size": slot.size,
                "blocks": slot.blocks,
                "use_cache": slot.use_cache,
                "read_token": token,
                "is_new": True,
            }

        # Existing readable items
        for item in to_read:
            uuid = item["uuid"]
            slot = self.manager.open_read(uuid)
            token = None
            if item.get("generate_read_token", False):
                token = random_uuid()
                self._read_tokens[token] = uuid
                self._item_to_read_tokens.setdefault(uuid, set()).add(token)
            result_map[uuid] = {
                "uuid": uuid,
                "size": slot.size,
                "blocks": slot.blocks,
                "use_cache": slot.use_cache,
                "read_token": token,
                "is_new": False,
            }

        # Pending writes (ref_count == -1)
        for item in pending_writes:
            uuid = item["uuid"]
            token = None
            if item.get("generate_read_token", False):
                token = random_uuid()
                self._read_tokens[token] = uuid
                self._item_to_read_tokens.setdefault(uuid, set()).add(token)
            result_map[uuid] = {
                "uuid": uuid,
                "size": item["size"],
                "blocks": [],
                "use_cache": item.get("use_cache", True),
                "read_token": token,
                "is_new": False,
            }

        all_data = [result_map[item["uuid"]] for item in items_data]
        return json.dumps({"status": "ok", "data": all_data})

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------
    def open_write(self, data: bytes, identity: bytes) -> str | None:
        """Allocate blocks for a batch of items to be written."""
        write_request = json.loads(data)
        items = write_request["items"]
        timeout = float(write_request.get("timeout", 0.0))

        item_objs = [ShmWriteRequest(**item) for item in items]
        total_required = sum(item.size for item in item_objs)

        if total_required > self.storage.size:
            raise MemoryError(
                f"Requested {total_required} bytes exceeds "
                f"total storage size {self.storage.size}"
            )

        try:
            allocated = self.manager.open_write(item_objs)
        except MemoryError:
            if timeout == 0.0:
                raise
            if timeout < 0:
                timeout = self.max_timeout
            deadline = time.monotonic() + timeout
            self._memory_waiters.put((deadline, identity, "write", item_objs))
            return None

        return self._build_write_response(allocated, item_objs)

    def open_read(self, data: bytes, identity: bytes) -> str | None:
        """
        Acquire a read reference to an item, supporting both UUIDs and tokens.
        For UUIDs, generates a new read token and increments ref_count.
        For tokens, returns data without modifying ref_count.
        """
        read_request = json.loads(data)
        uuid_or_token = read_request["uuid"]
        timeout = float(read_request.get("timeout", 0.0))

        real_uuid, is_token = self._resolve_read_token(uuid_or_token)

        if is_token:
            # Token path: no ref_count increment
            if uuid_or_token not in self._read_tokens:
                raise ValueError(
                    f"Read token '{uuid_or_token}' not found or already consumed"
                )
            try:
                size, blocks = self.manager._get_readable_item_blocks(real_uuid)
            except RuntimeError as e:
                if "still being written" in str(e):
                    if timeout == 0.0:
                        raise RuntimeError(
                            f"Item {real_uuid} is still being written"
                        ) from None
                    if timeout < 0:
                        timeout = self.max_timeout
                    deadline = time.monotonic() + timeout
                    self._open_read_waiters[real_uuid].put(
                        (deadline, identity, uuid_or_token)
                    )
                    return None
                else:
                    raise
            resp = ShmAllocation(
                uuid=real_uuid,
                size=size,
                blocks=blocks,
                use_cache=True,
                read_token=uuid_or_token,
                is_new=False,
            )
            return json.dumps({"status": "ok", "data": asdict(resp)})
        else:
            # UUID path: generate new token, increment ref_count
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
                self._open_read_waiters[real_uuid].put(
                    (deadline, identity, uuid_or_token)
                )
                return None
            token = random_uuid()
            self._read_tokens[token] = real_uuid
            self._item_to_read_tokens.setdefault(real_uuid, set()).add(token)
            item = self.manager.open_read(real_uuid)
            resp = ShmAllocation(
                uuid=item.uuid,
                size=item.size,
                blocks=item.blocks,
                use_cache=item.use_cache,
                read_token=token,
                is_new=False,
            )
            return json.dumps({"status": "ok", "data": asdict(resp)})

    def open_write_or_read(self, data: bytes, identity: bytes) -> str | None:
        """
        Atomically open for reading or writing a batch of items.

        For each item:
          - If the UUID does not exist, it is allocated for writing.
          - If the UUID exists and is readable, it is opened for reading.
          - If the UUID exists but is being written, a read token is generated
            immediately without waiting; the token will be counted in close_write.

        If memory is insufficient for allocating new items, the request is queued
        (if timeout > 0) and retried when space becomes available.

        Returns:
            JSON response immediately if memory is available.
            None if the request is queued (timeout > 0 and MemoryError).
        Raises:
            MemoryError: if timeout=0 and not enough memory.
            ValueError: on invalid parameters.
        """
        req = json.loads(data)
        items_data = req["items"]
        timeout = float(req.get("timeout", 0.0))

        if timeout < 0:
            timeout = self.max_timeout

        try:
            response = self._core_write_or_read(items_data)
            return response
        except MemoryError:
            if timeout == 0.0:
                raise
            deadline = time.monotonic() + timeout
            self._memory_waiters.put((deadline, identity, "write_or_read", items_data))
            return None

    def wait_for_readable(self, data: bytes, identity: bytes) -> str | None:
        """Wait for an item to become readable (write completed)."""
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
        self._readable_waiters[real_uuid].put((deadline, identity))
        return None

    # ------------------------------------------------------------------
    # Deferred request processing (FCFS with batch attempts)
    # ------------------------------------------------------------------
    def _defer_memory_requests(self, socket: zmq.Socket) -> None:
        """
        Process as many pending memory requests as possible.
        Stops when the next request cannot be satisfied.
        """
        while not self._memory_waiters.empty():
            first = self._memory_waiters.peek()
            if first is None:
                break

            deadline, identity, req_type, payload = first
            now = time.monotonic()
            if deadline <= now:
                self._memory_waiters.get()
                self._send_response(
                    socket, identity, ERROR, "TimeoutError: memory allocation timed out"
                )
                continue

            try:
                if req_type == "write":
                    allocated = self.manager.open_write(payload)
                    response = self._build_write_response(allocated, payload)
                    self._memory_waiters.get()
                    self._send_response(socket, identity, OK, response)
                elif req_type == "write_or_read":
                    response = self._execute_open_write_or_read(payload, identity)
                    self._memory_waiters.get()
                    self._send_response(socket, identity, OK, response)
                else:
                    raise ValueError(
                        f"Unknown request type in memory queue: {req_type}"
                    )
            except MemoryError:
                # Not enough memory – stop and retry later
                break
            except Exception as e:
                self._memory_waiters.get()
                self._send_response(socket, identity, ERROR, f"{type(e).__name__}: {e}")
                logger.warning("Deferred memory request failed: %s", e)

    def _execute_open_write_or_read(
        self, items_data: list[dict], identity: bytes
    ) -> str:
        """Helper for retrying deferred open_write_or_read."""
        return self._core_write_or_read(items_data)

    def defer_open_read(self, socket: zmq.Socket, uuid: str) -> None:
        """
        Wake up pending open_read waiters for the given UUID.
        Assumes expired entries have already been cleaned.
        For each waiter:
          - If the original request was a token:
            return data without changing ref_count.
          - If it was a UUID:
            generate a new token, increment ref_count, and return data+token.
        """
        q = self._open_read_waiters.get(uuid)
        if q is None or q.empty():
            return

        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            # Item no longer exists – send error to all waiters
            while not q.empty():
                _, identity, _ = q.get()
                self._send_response(socket, identity, ERROR, "UUID not found")
            del self._open_read_waiters[uuid]
            return

        if info["ref_count"] >= 0:
            while not q.empty():
                _, identity, original_id = q.get()
                if original_id in self._read_tokens:
                    # Token path
                    try:
                        size, blocks = self.manager._get_readable_item_blocks(uuid)
                        resp = ShmAllocation(
                            uuid=uuid,
                            size=size,
                            blocks=blocks,
                            use_cache=True,
                            read_token=original_id,
                            is_new=False,
                        )
                        self._send_response(
                            socket,
                            identity,
                            OK,
                            json.dumps({"status": "ok", "data": asdict(resp)}),
                        )
                    except Exception as e:
                        self._send_response(
                            socket, identity, ERROR, f"{type(e).__name__}: {e}"
                        )
                else:
                    # UUID path – generate new token
                    try:
                        token = random_uuid()
                        self._read_tokens[token] = uuid
                        self._item_to_read_tokens.setdefault(uuid, set()).add(token)
                        item = self.manager.open_read(uuid)
                        resp = ShmAllocation(
                            uuid=item.uuid,
                            size=item.size,
                            blocks=item.blocks,
                            use_cache=item.use_cache,
                            read_token=token,
                            is_new=False,
                        )
                        self._send_response(
                            socket,
                            identity,
                            OK,
                            json.dumps({"status": "ok", "data": asdict(resp)}),
                        )
                    except Exception as e:
                        self._send_response(
                            socket, identity, ERROR, f"{type(e).__name__}: {e}"
                        )
            del self._open_read_waiters[uuid]

    def defer_wait_readable(self, socket: zmq.Socket, uuid: str) -> None:
        """Wake up pending WAIT_FOR_READABLE waiters for the given UUID."""
        q = self._readable_waiters.get(uuid)
        if q is None or q.empty():
            return

        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            while not q.empty():
                _, identity = q.get()
                self._send_response(socket, identity, ERROR, "UUID not found")
            del self._readable_waiters[uuid]
            return

        if info["ref_count"] >= 0:
            while not q.empty():
                _, identity = q.get()
                self._send_response(socket, identity, OK, _OK_RESPONSE)
            del self._readable_waiters[uuid]

    # ------------------------------------------------------------------
    # Token management helpers
    # ------------------------------------------------------------------
    def _resolve_read_token(self, uuid: str) -> tuple[str, bool]:
        """Return (real_uuid, is_token) without modifying state."""
        if uuid in self._read_tokens:
            return self._read_tokens[uuid], True
        return uuid, False

    def _destroy_token(self, token: str) -> None:
        """Permanently destroy a read token."""
        real_uuid = self._read_tokens.pop(token, None)
        if real_uuid is not None:
            token_set = self._item_to_read_tokens.get(real_uuid)
            if token_set is not None:
                token_set.discard(token)
                if not token_set:
                    self._item_to_read_tokens.pop(real_uuid, None)

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
                        is_new=True,
                    )
                )
            )
        return json.dumps({"status": "ok", "data": data})

    # ------------------------------------------------------------------
    # Other command handlers
    # ------------------------------------------------------------------
    def close_write(self, items_data: bytes) -> str:
        """Finish writing an item, making it readable and cacheable."""
        items = json.loads(items_data)
        uuid = items["uuid"]
        tokens = self._item_to_read_tokens.get(uuid, set())
        open_n_reads = len(tokens)
        self.manager.close_write(uuid, open_n_reads)
        return _OK_RESPONSE

    def close_read(self, uuid: str) -> str:
        """Release a read reference. Accepts only read tokens."""
        real_uuid, is_token = self._resolve_read_token(uuid)
        if not is_token:
            try:
                self.manager.get_info(uuid)
            except ValueError:
                raise ValueError(f"Read token '{uuid}' not found") from None
            else:
                raise ValueError(
                    f"close_read only accepts read tokens, got UUID '{uuid}'"
                )

        try:
            info = self.manager.get_info(real_uuid)
            if info["ref_count"] > 0:
                self.manager.close_read(real_uuid)
        except ValueError:
            pass
        self._destroy_token(uuid)
        return _OK_RESPONSE

    def delete(self, uuid: str) -> str:
        """Delete an item and free its blocks (forcefully)."""
        if uuid in self._item_to_read_tokens:
            for token in list(self._item_to_read_tokens[uuid]):
                self._destroy_token(token)
        self.manager.delete(uuid, force=True)
        return _OK_RESPONSE

    def get_manager_states(self) -> str:
        """Return manager statistics as JSON."""
        return json.dumps(self.manager.get_manager_states())

    def get_storage_info(self) -> str:
        """Return storage metadata as JSON."""
        info = {
            "name": self.shm_name,
            "size": self.size,
            "block_size": self.block_size,
            "n_block": self.n_block,
        }
        return json.dumps({"status": "ok", "data": info})

    def get_info(self, uuid: str) -> str:
        """Return object info as JSON. Supports UUID or token."""
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
    @staticmethod
    def _send_response(
        socket: zmq.Socket, identity: bytes, status: bytes, payload: str
    ) -> None:
        """Send a multipart response."""
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
            GET_INFO: (server.get_info, True),
            GET_MANAGER_STATES: (server.get_manager_states, False),
            GET_STORAGE_INFO: (server.get_storage_info, False),
        }

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        logger.info("PagedShmServer started at %s", address)

        next_cleanup_time = time.monotonic() + CLEANUP_INTERVAL

        while True:
            try:
                socks = dict(poller.poll(POLL_INTERVAL))
            except (zmq.ZMQError, KeyboardInterrupt, EOFError):
                break

            now = time.monotonic()
            if now >= next_cleanup_time:
                server._perform_maintenance(socket)
                next_cleanup_time = now + CLEANUP_INTERVAL

            if socket not in socks or socks[socket] != zmq.POLLIN:
                continue

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

            # DEBUG_CLEAN
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

            if command == OPEN_WRITE_OR_READ:
                try:
                    result = server.open_write_or_read(payloads[0].decode(), identity)
                    if result is not None:
                        server._send_response(socket, identity, OK, result)
                except (ValueError, MemoryError) as e:
                    server._send_response(
                        socket, identity, ERROR, f"{type(e).__name__}: {e}"
                    )
                except Exception as e:
                    logger.exception("Unexpected error in OPEN_WRITE_OR_READ")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            if command == WAIT_FOR_READABLE:
                try:
                    result = server.wait_for_readable(payloads[0].decode(), identity)
                    if result is not None:
                        server._send_response(socket, identity, OK, result)
                except (ValueError, RuntimeError) as e:
                    server._send_response(
                        socket, identity, ERROR, f"{type(e).__name__}: {e}"
                    )
                except Exception as e:
                    logger.exception("Unexpected error in WAIT_FOR_READABLE")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

            if command == DELETE:
                uuid = payloads[0].decode()
                # Clear pending wait queues for this UUID before deletion
                q = server._open_read_waiters.pop(uuid, None)
                if q is not None:
                    while not q.empty():
                        _, ident, _ = q.get()
                        server._send_response(
                            socket, ident, ERROR, "Item deleted while waiting"
                        )
                q = server._readable_waiters.pop(uuid, None)
                if q is not None:
                    while not q.empty():
                        _, ident = q.get()
                        server._send_response(
                            socket, ident, ERROR, "Item deleted while waiting"
                        )

                try:
                    server.delete(uuid)
                    server._send_response(socket, identity, OK, _OK_RESPONSE)
                    # After deletion, try to satisfy any pending memory requests
                    server._defer_memory_requests(socket)
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

            # After CLOSE_WRITE, wake up waiting readers, wait_for_readable waiters,
            # and attempt to process deferred memory requests.
            if command == CLOSE_WRITE:
                try:
                    close_req = json.loads(payloads[0].decode())
                    uuid = close_req.get("uuid")
                    if uuid:
                        server.clean_expired_open_read(socket, uuid)
                        server.clean_expired_wait_readable(socket, uuid)
                        server.defer_open_read(socket, uuid)
                        server.defer_wait_readable(socket, uuid)
                        server._defer_memory_requests(socket)
                except Exception:
                    pass

            # After CLOSE_READ, attempt to process deferred memory requests.
            if command == CLOSE_READ:
                if not payloads:
                    server._send_response(
                        socket, identity, ERROR, "Missing token for close_read"
                    )
                    continue
                param = payloads[0].decode()
                try:
                    result = server.close_read(param)
                    server._send_response(socket, identity, OK, result)
                except ValueError as e:
                    server._send_response(socket, identity, ERROR, f"ValueError: {e}")
                except Exception as e:
                    logger.exception("Unexpected error in CLOSE_READ")
                    server._send_response(
                        socket, identity, ERROR, f"Internal error: {e}"
                    )
                continue

    except Exception as e:
        logger.exception("Fatal error in zmq_server: %s", e)
    finally:
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
        debug=debug or envs.VLLM_SERVER_DEV_MODE,
    )
    paged_shm_server.start()

    multimodal_config.paged_shm_server_address = paged_shm_server.address
    return paged_shm_server
