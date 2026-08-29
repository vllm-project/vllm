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

        # Unified priority queue for pending requests that are waiting for memory.
        # Elements: (deadline, identity, request_type, payload)
        # request_type:
        #     "write" (for open_write) or "write_or_read" (for open_write_or_read)
        # payload:
        #     list[ShmWriteRequest] for "write",
        #     or items_data (list[dict]) for "write_or_read"
        self.wait_for_memory: PriorityQueue = PriorityQueue()

        # Per‑uuid priority queues for pending open_read requests.
        # Keys are item UUIDs; values are PriorityQueues with elements
        # (deadline, identity, token_or_none).
        self.wait_for_open_read: dict[str, PriorityQueue] = {}
        # Set of UUIDs that have at least one pending open_read waiter
        self._open_read_pending: set[str] = set()

        # Per‑uuid priority queues for pending WAIT_FOR_READABLE requests.
        self.wait_for_readable_completion: dict[str, PriorityQueue] = {}
        # Set of UUIDs that have at least one pending wait_for_readable waiter
        self._wait_readable_pending: set[str] = set()

        # Read token storage: token -> real_uuid
        self._read_tokens: dict[str, str] = {}
        # Reverse mapping: real_uuid -> set of tokens
        self._item_to_read_tokens: dict[str, set[str]] = {}

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
        Uses get/put to peek without relying on internal `queue` attribute.
        """
        now = time.monotonic()
        expired = []
        # Peek at the first element without popping if not expired
        while not queue.empty():
            # Get the item (we'll either re-put it or discard)
            item = queue.get()
            deadline = item[0]
            if deadline > now:
                # Not expired: put it back and stop (since queue is ordered)
                queue.put(item)
                break
            else:
                # Expired: collect to send error later
                expired.append(item)
        # Now send errors for all expired items
        for _, identity, *rest in expired:
            self._send_response(socket, identity, ERROR, error_msg)
        # Return True if the queue is now empty
        return queue.empty()

    # ------------------------------------------------------------------
    # Expiration cleanup methods
    # ------------------------------------------------------------------
    def _clean_expired_memory_waiters(self, socket: zmq.Socket) -> None:
        """Clean expired entries from the unified memory wait queue."""
        self._clean_expired_queue(
            self.wait_for_memory,
            socket,
            "TimeoutError: memory allocation timed out",
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

    def clean_expired_wait_readable(
        self, socket: zmq.Socket, uuid: str | None = None
    ) -> None:
        """
        Clean expired entries from wait_for_readable queues.
        If uuid is given, only that queue is cleaned; otherwise all pending queues.
        """
        if uuid is not None:
            q = self.wait_for_readable_completion.get(uuid)
            if q is not None:
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: wait_for_readable timed out",
                )
                if empty:
                    self.wait_for_readable_completion.pop(uuid, None)
                    self._wait_readable_pending.discard(uuid)
        else:
            for u in list(self._wait_readable_pending):
                q = self.wait_for_readable_completion.get(u)
                if q is None:
                    self._wait_readable_pending.discard(u)
                    continue
                empty = self._clean_expired_queue(
                    q,
                    socket,
                    "TimeoutError: wait_for_readable timed out",
                )
                if empty:
                    self.wait_for_readable_completion.pop(u, None)
                    self._wait_readable_pending.discard(u)

    # ------------------------------------------------------------------
    # Periodic maintenance (cleanup + deferred request processing)
    # ------------------------------------------------------------------
    def _perform_maintenance(self, socket: zmq.Socket) -> None:
        """
        Perform periodic cleanup and attempt to satisfy deferred open_write requests.
        Called at fixed intervals (every CLEANUP_INTERVAL seconds).
        """
        self._clean_expired_memory_waiters(socket)
        self.clean_expired_open_read(socket)
        self.clean_expired_wait_readable(socket)
        self._defer_memory_requests(socket)

    # ------------------------------------------------------------------
    # Debug-only cleanup – forcibly cleans all pending queues and purges
    # tokens. Only available when self.debug is True.
    # ------------------------------------------------------------------
    def debug_cleanup(self, socket: zmq.Socket) -> None:
        """Force‑clean all wait queues and purge tokens (debug only)."""
        if not self.debug:
            raise RuntimeError("debug_cleanup called but debug mode is disabled")

        # 1. Clean all pending memory waiters (both write and write_or_read)
        while not self.wait_for_memory.empty():
            _, identity, _, _ = self.wait_for_memory.get()
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

        # 3. Clean all pending wait_for_readable waiters
        for u in list(self._wait_readable_pending):
            q = self.wait_for_readable_completion.pop(u, None)
            if q is not None:
                while not q.empty():
                    _, identity = q.get()
                    self._send_response(
                        socket, identity, ERROR, "Cleaned by debug_cleanup"
                    )
        self._wait_readable_pending.clear()

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
    # Core logic for open_write_or_read
    # ------------------------------------------------------------------
    def _core_write_or_read(self, items_data: list[dict]) -> str:
        """
        Core logic for open_write_or_read.
        Performs atomic allocation of missing items and reads existing ones.
        Returns JSON response string.
        Raises MemoryError if allocation fails.
        """
        # Classify items
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

        # Allocate missing items
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

        # Build result map
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
                "blocks": [],  # not yet available
                "use_cache": item.get("use_cache", True),
                "read_token": token,
                "is_new": False,
            }

        # Ensure all UUIDs are present in result_map (defensive)
        for item in items_data:
            uuid = item["uuid"]
            if uuid not in result_map:
                logger.warning(
                    "UUID %s not found in result_map, adding placeholder", uuid
                )
                result_map[uuid] = {
                    "uuid": uuid,
                    "size": item.get("size", 0),
                    "blocks": [],
                    "use_cache": item.get("use_cache", True),
                    "read_token": None,
                    "is_new": False,
                }

        # Assemble response in input order
        all_data = [result_map[item["uuid"]] for item in items_data]
        return json.dumps({"status": "ok", "data": all_data})

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
            # Queue the request in the unified memory wait queue
            self.wait_for_memory.put((deadline, identity, "write", item_objs))
            return None

        return self._build_write_response(allocated, item_objs)

    def open_read(self, data: bytes, identity: bytes) -> str | None:
        """
        Acquire a read reference to an item, returning its block list and size.
        Supports both real UUIDs and read tokens:
        - For real UUIDs: generates a new read token, increases ref_count via
          manager.open_read(), and returns the token alongside data.
        - For tokens: returns data without modifying ref_count (token already
          holds a reference).
        """
        read_request = json.loads(data)
        uuid_or_token = read_request["uuid"]
        timeout = float(read_request.get("timeout", 0.0))

        real_uuid, is_token = self._resolve_read_token(uuid_or_token)

        if is_token:
            # Token path: no ref_count increment, just return data
            if uuid_or_token not in self._read_tokens:
                raise ValueError(
                    f"Read token '{uuid_or_token}' not found or already consumed"
                )
            # Get the item info without modifying ref_count
            try:
                size, blocks = self.manager._get_readable_item_blocks(real_uuid)
            except RuntimeError as e:
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
                    q.put((deadline, identity, uuid_or_token))  # store token
                    return None
                else:
                    raise
            # Build response with the same token
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
                q = self.wait_for_open_read.setdefault(real_uuid, PriorityQueue())
                self._open_read_pending.add(real_uuid)
                q.put((deadline, identity, uuid_or_token))  # store UUID
                return None
            # Create new token and increment ref_count
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
            self.wait_for_memory.put((deadline, identity, "write_or_read", items_data))
            return None

    def wait_for_readable(self, data: bytes, identity: bytes) -> str | None:
        """
        Wait for an item to become readable (i.e., write completed).
        Supports both real UUIDs and read tokens. Returns OK response if
        immediately readable, otherwise queues the request.
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
        q = self.wait_for_readable_completion.setdefault(real_uuid, PriorityQueue())
        self._wait_readable_pending.add(real_uuid)
        q.put((deadline, identity))
        return None

    # ------------------------------------------------------------------
    # Deferred request processing (FCFS)
    # ------------------------------------------------------------------
    def _defer_memory_requests(self, socket: zmq.Socket) -> None:
        """
        Attempt to satisfy the first pending request in the memory wait queue (FCFS).
        Supports both "write" and "write_or_read" request types.
        If the request still fails due to MemoryError, re‑queue it (if not expired).
        """
        if self.wait_for_memory.empty():
            return

        item = self.wait_for_memory.get()
        deadline, identity, req_type, payload = item
        now = time.monotonic()
        if deadline <= now:
            self._send_response(
                socket,
                identity,
                ERROR,
                "TimeoutError: memory allocation timed out",
            )
            return

        try:
            if req_type == "write":
                # payload is list[ShmWriteRequest]
                allocated = self.manager.open_write(payload)
                response = self._build_write_response(allocated, payload)
                self._send_response(socket, identity, OK, response)
            elif req_type == "write_or_read":
                # payload is items_data (list[dict])
                response = self._execute_open_write_or_read(payload, identity)
                self._send_response(socket, identity, OK, response)
            else:
                # Should never happen
                raise ValueError(f"Unknown request type in memory queue: {req_type}")
        except MemoryError:
            # Still not enough memory – re‑queue with the same deadline
            self.wait_for_memory.put(item)
        except Exception as e:
            # Other errors – pop and report
            self._send_response(socket, identity, ERROR, f"{type(e).__name__}: {e}")
            logger.warning("Deferred memory request failed: %s", e)

    def _execute_open_write_or_read(
        self, items_data: list[dict], identity: bytes
    ) -> str:
        """
        Internal helper to execute the open_write_or_read logic without queuing.
        Used for retrying deferred requests.
        Raises MemoryError if allocation fails.
        """
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
        q = self.wait_for_open_read.get(uuid)
        if q is None or q.empty():
            self._open_read_pending.discard(uuid)
            return

        # Check if item is now readable
        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            while not q.empty():
                _, ident, original_id = q.get()
                self._send_response(socket, ident, ERROR, "UUID not found")
            self.wait_for_open_read.pop(uuid, None)
            self._open_read_pending.discard(uuid)
            return

        if info["ref_count"] >= 0:
            while not q.empty():
                _, identity, original_id = q.get()
                if original_id in self._read_tokens:
                    # Token path: no ref_count change
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
            self.wait_for_open_read.pop(uuid, None)
            self._open_read_pending.discard(uuid)

    def defer_wait_readable(self, socket: zmq.Socket, uuid: str) -> None:
        """
        Wake up pending WAIT_FOR_READABLE waiters for the given UUID.
        """
        q = self.wait_for_readable_completion.get(uuid)
        if q is None or q.empty():
            self._wait_readable_pending.discard(uuid)
            return

        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            while not q.empty():
                _, ident = q.get()
                self._send_response(socket, ident, ERROR, "UUID not found")
            self.wait_for_readable_completion.pop(uuid, None)
            self._wait_readable_pending.discard(uuid)
            return

        if info["ref_count"] >= 0:
            while not q.empty():
                _, identity = q.get()
                self._send_response(socket, identity, OK, _OK_RESPONSE)
            self.wait_for_readable_completion.pop(uuid, None)
            self._wait_readable_pending.discard(uuid)

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
                        is_new=True,  # Always new for open_write
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
        Release a read reference. **Only accepts read tokens**.
        If a UUID is passed, a ValueError is raised.
        """
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
        # Destroy all read tokens associated with this item
        if uuid in self._item_to_read_tokens:
            for token in list(self._item_to_read_tokens[uuid]):
                self._destroy_token(token)
        self.manager.delete(uuid, force=True)
        return _OK_RESPONSE

    def get_manager_states(self) -> str:
        """Return manager statistics as a JSON string."""
        return json.dumps(self.manager.get_manager_states())

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
        """Return object info as a JSON string. Supports UUID or token."""
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
                q = server.wait_for_open_read.pop(uuid, None)
                if q is not None:
                    while not q.empty():
                        _, ident, _ = q.get()
                        server._send_response(
                            socket, ident, ERROR, "Item deleted while waiting"
                        )
                    server._open_read_pending.discard(uuid)
                q = server.wait_for_readable_completion.pop(uuid, None)
                if q is not None:
                    while not q.empty():
                        _, ident = q.get()
                        server._send_response(
                            socket, ident, ERROR, "Item deleted while waiting"
                        )
                    server._wait_readable_pending.discard(uuid)

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
            # and any pending memory requests (now memory may be freed)
            if command == CLOSE_WRITE:
                try:
                    close_req = json.loads(payloads[0].decode())
                    uuid = close_req.get("uuid")
                    if uuid:
                        server.clean_expired_open_read(socket, uuid)
                        server.clean_expired_wait_readable(socket, uuid)
                        server.defer_open_read(socket, uuid)
                        server.defer_wait_readable(socket, uuid)
                        # Also attempt to satisfy pending memory requests
                        server._defer_memory_requests(socket)
                except Exception:
                    pass

            # After CLOSE_READ, try to satisfy deferred memory requests
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
