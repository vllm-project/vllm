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
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.utils import random_uuid

from .constant import (
    CLOSE_READ,
    CLOSE_WRITE,
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


class PagedShmServer:
    """Server‑side wrapper that exposes PagedShmManager over ZMQ."""

    def __init__(self, size: int, block_size: int):
        self._resources = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, self._resources.close)

        self.storage = PagedShmStorage(size, block_size, pin=False)
        self._resources.callback(self.storage.close)

        self.manager = PagedShmManager(size, block_size)

        self.size = self.storage.size
        self.shm_name = self.storage.name
        self.n_block = self.storage.n_block
        self.block_size = self.storage.block_size

        # Priority queue for pending open_write requests.
        # Elements: (deadline, identity, list_of_ShmItem)
        self.wait_for_open_write: PriorityQueue = PriorityQueue()

        # Per-uuid priority queues for pending open_read requests.
        # Keys are item UUIDs; values are PriorityQueues with elements
        # (deadline, identity). Grouping by uuid allows fast wake-up
        # after close_write without scanning all waiters.
        self.wait_for_open_read: dict[str, PriorityQueue] = {}

        # Per-uuid priority queues for pending WAIT_WRITE requests.
        self.wait_for_write_completion: dict[str, PriorityQueue] = {}

        # Read token storage: token -> real_uuid
        self._read_tokens: dict[str, str] = {}
        # Reverse mapping: real_uuid -> set of tokens
        self._item_to_read_tokens: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Helper for constructing write responses
    # ------------------------------------------------------------------
    def _build_write_response(self, allocated: list, requests: list[ShmWriteRequest]) -> str:
        """Build JSON response for open_write, including read tokens if requested."""
        data = []
        for a, req in zip(allocated, requests):
            token = None
            if req.generate_read_token:
                token = random_uuid()
                # Store token -> item_uuid
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
    # Request handlers
    # ------------------------------------------------------------------
    def open_write(self, data: bytes, identity: bytes) -> str | None:
        """Allocate blocks for a batch of items to be written.

        Args:
            data: JSON payload containing 'items' and optional 'timeout'.
            identity: ZMQ identity of the requesting client.

        Returns:
            JSON response string if the request was satisfied immediately,
            None if the request was queued due to insufficient space.

        Raises:
            MemoryError: if timeout is 0 and space is insufficient.
            ValueError: if any item UUID already exists or size is invalid.
        """
        write_request = json.loads(data)
        items = write_request["items"]
        timeout = float(write_request.get("timeout", 0.0))

        item_objs = [ShmWriteRequest(**item) for item in items]
        try:
            allocated = self.manager.open_write(item_objs)
        except MemoryError:
            if timeout == 0.0:
                raise  # fail immediately
            # Queue the request with a deadline (inf for infinite wait)
            deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
            self.wait_for_open_write.put((deadline, identity, item_objs))
            return None

        return self._build_write_response(allocated, item_objs)

    def open_read(self, data: bytes, identity: bytes) -> str | None:
        """Acquire a read reference to an item, returning its block list and size.

        Args:
            data: JSON payload containing 'uuid' and optional 'timeout'.
            identity: ZMQ identity of the requesting client.

        Returns:
            JSON response string if the item is immediately available,
            None if the request was queued because the item is being written.

        Raises:
            ValueError: if uuid does not exist or timeout=0 and item is being written.
        """
        read_request = json.loads(data)
        uuid = read_request["uuid"]
        timeout = float(read_request.get("timeout", 0.0))

        # Resolve read token to real UUID
        real_uuid, is_token = self._resolve_read_token(uuid)

        # Get item info
        try:
            info = self.manager.get_info(real_uuid)
        except ValueError:
            # If token points to a deleted item, invalidate token and re-raise
            if is_token:
                self._invalidate_token(uuid)
            raise

        if info["ref_count"] < 0:  # being written
            if timeout == 0.0:
                # Fail immediately
                raise RuntimeError(f"Item {uuid} is still being written")
            else:
                # Queue the request (store token if present)
                deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
                q = self.wait_for_open_read.setdefault(real_uuid, PriorityQueue())
                # Store the original identifier (token or uuid) to be used when waking
                q.put((deadline, identity, uuid if is_token else None))
                return None

        # Item is readable: consume token if present, then open read
        if is_token:
            self._consume_token(uuid)  # token consumed now

        return self._open_read(real_uuid)

    def _resolve_read_token(self, uuid: str) -> tuple[str, bool]:
        """Return (real_uuid, is_token) after validating the token."""
        if uuid in self._read_tokens:
            return self._read_tokens[uuid], True
        return uuid, False

    def _consume_token(self, token: str) -> None:
        """Consume a read token (remove it)."""
        real_uuid = self._read_tokens.pop(token, None)
        if real_uuid is not None:
            self._item_to_read_tokens.get(real_uuid, set()).discard(token)

    def _invalidate_token(self, token: str) -> None:
        """Force-invalidate a token (e.g., on item deletion)."""
        real_uuid = self._read_tokens.pop(token, None)
        if real_uuid is not None:
            self._item_to_read_tokens.get(real_uuid, set()).discard(token)

    def _open_read(self, uuid: str) -> str:
        """Internal helper to open a read reference and build response."""
        item = self.manager.open_read(uuid)
        resp = ShmAllocation(
            uuid=item.uuid, size=item.size, blocks=item.blocks, use_cache=item.use_cache
        )
        return json.dumps({"status": "ok", "data": asdict(resp)})

    def wait_write(self, data: bytes, identity: bytes) -> str | None:
        """Wait for an item to become readable. Supports read tokens."""
        req = json.loads(data)
        uuid_or_token = req["uuid"]
        timeout = float(req.get("timeout", 0.0))

        # Resolve token to real UUID
        try:
            real_uuid, is_token = self._resolve_read_token(uuid_or_token)
        except ValueError:
            # Token is invalid or not found; re-raise with clear error
            raise ValueError(f"Invalid read token or UUID: {uuid_or_token}")

        # Wait only supports real UUID (token resolution succeeded)
        try:
            info = self.manager.get_info(real_uuid)
        except ValueError:
            raise ValueError(f"UUID {uuid_or_token} not found")

        if info["ref_count"] >= 0:
            # Already readable: respond immediately (token is untouched)
            return json.dumps({"status": "ok"})
        else:
            # Still being written: queue the request
            if timeout == 0.0:
                raise RuntimeError(f"Item {uuid_or_token} is still being written")
            deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
            q = self.wait_for_write_completion.setdefault(real_uuid, PriorityQueue())
            q.put((deadline, identity))
            return None

    # ------------------------------------------------------------------
    # Deferred request processing
    # ------------------------------------------------------------------
    def defer_open_write(self, socket: zmq.Socket, max_attempts: int = 5) -> None:
        """Process pending open_write requests, up to max_attempts.

        Expired requests are discarded with a timeout error. Then non-expired
        requests are attempted one by one; if any fails due to insufficient
        space, we stop to avoid blocking the main loop. This prevents a
        single large request from starving smaller ones.
        """
        now = time.monotonic()

        # Discard expired requests
        while not self.wait_for_open_write.empty():
            deadline, identity, _ = self.wait_for_open_write.queue[0]
            if deadline <= now:
                _, ident, _ = self.wait_for_open_write.get()
                self._send_response(
                    socket, ident, ERROR, b"TimeoutError: open_write timed out"
                )
                continue
            break  # queue is sorted by deadline; no more expired entries

        # Try to satisfy up to max_attempts requests
        attempts = 0
        while not self.wait_for_open_write.empty() and attempts < max_attempts:
            deadline, identity, item_objs = self.wait_for_open_write.queue[0]
            # Re-check deadline (it might have expired while we were processing)
            if deadline <= now:
                _, ident, _ = self.wait_for_open_write.get()
                self._send_response(
                    socket, ident, ERROR, b"TimeoutError: open_write timed out"
                )
                continue
            try:
                allocated = self.manager.open_write(item_objs)
                self.wait_for_open_write.get()
                response = self._build_write_response(allocated, item_objs)
                self._send_response(socket, identity, OK, response.encode("utf-8"))
                attempts += 1
            except MemoryError:
                # Not enough space for this request; stop and try later
                break
            except Exception as e:
                # Other errors (duplicate UUID, etc.) are reported immediately
                self.wait_for_open_write.get()  # remove the faulty request
                self._send_response(
                    socket, identity, ERROR, f"{type(e).__name__}: {e}".encode()
                )
                attempts += 1  # count as attempted, but stop? Continue to next.

    def defer_open_read(self, socket: zmq.Socket, uuid: str | None = None) -> None:
        """Process pending open_read requests.

        If *uuid* is given (typically after a close_write), only the wait
        queue for that uuid is examined, allowing quick wake‑up without
        scanning all waiters.
        If *uuid* is None (periodic call), only expired requests are discarded
        to avoid full scans; we iterate over all queues for cleanup.

        Args:
            socket: ZMQ socket used to send responses.
            uuid: Optional UUID whose waiters should be processed now.
        """
        now = time.monotonic()

        if uuid is not None:
            q = self.wait_for_open_read.get(uuid)
            if q is None:
                return

            # Discard expired
            while not q.empty():
                deadline, identity, token = q.queue[0]
                if deadline <= now:
                    _, ident, _ = q.get()
                    self._send_response(
                        socket, ident, ERROR, b"TimeoutError: open_read timed out"
                    )
                    continue
                break

            # Satisfy waiters if item is readable
            if not q.empty():
                info = self.manager.get_info(uuid)
                if info["ref_count"] >= 0:
                    while not q.empty():
                        deadline, identity, token_or_none = q.get()
                        try:
                            # If this was a token, consume it now
                            if token_or_none is not None:
                                self._consume_token(token_or_none)
                            result = self._open_read(uuid)
                            self._send_response(
                                socket, identity, OK, result.encode("utf-8")
                            )
                        except Exception as e:
                            # Should not happen if ref_count >= 0, but be safe
                            self._send_response(
                                socket,
                                identity,
                                ERROR,
                                f"{type(e).__name__}: {e}".encode(),
                            )

            # Clean up empty queue
            if q.empty():
                self.wait_for_open_read.pop(uuid, None)

        else:
            # Periodic scan: only handle timeouts to avoid heavy processing
            empty_uuids = []
            for u, q in self.wait_for_open_read.items():
                # Remove expired requests from this queue
                while not q.empty():
                    deadline, identity, token = q.queue[0]
                    if deadline <= now:
                        _, ident, _ = q.get()
                        self._send_response(
                            socket, ident, ERROR, b"TimeoutError: open_read timed out"
                        )
                        continue
                    break
                if q.empty():
                    empty_uuids.append(u)

            # Clean up queues that became empty
            for u in empty_uuids:
                self.wait_for_open_read.pop(u, None)

    def defer_wait_write(self, socket: zmq.Socket, uuid: str | None = None) -> None:
        """
        Process pending WAIT_WRITE requests.

        If *uuid* is given (after close_write), we wake all waiters for that
        uuid and send a success response (no read lock is acquired).
        If *uuid* is None (periodic cleanup), we only discard expired requests.

        Args:
            socket: ZMQ socket used to send responses.
            uuid: Optional UUID whose waiters should be processed now.
        """
        now = time.monotonic()

        if uuid is not None:
            q = self.wait_for_write_completion.get(uuid)
            if q is None:
                return

            # Remove expired requests
            while not q.empty():
                deadline, identity = q.queue[0]
                if deadline <= now:
                    _, ident = q.get()
                    self._send_response(
                        socket, ident, ERROR, b"TimeoutError: wait_write timed out"
                    )
                    continue
                break

            # Satisfy all remaining waiters (item is now readable)
            if not q.empty():
                # Ensure the item actually exists and is readable
                try:
                    info = self.manager.get_info(uuid)
                    if info["ref_count"] >= 0:
                        while not q.empty():
                            deadline, identity = q.get()
                            self._send_response(
                                socket, identity, OK, b'{"status":"ok"}'
                            )
                except ValueError:
                    # Item disappeared; send error to all waiters
                    while not q.empty():
                        _, ident = q.get()
                        self._send_response(
                            socket, ident, ERROR, b"UUID not found"
                        )

            if q.empty():
                self.wait_for_write_completion.pop(uuid, None)

        else:
            # Periodic cleanup: remove expired entries from all queues
            empty_uuids = []
            for u, q in self.wait_for_write_completion.items():
                while not q.empty():
                    deadline, identity = q.queue[0]
                    if deadline <= now:
                        _, ident = q.get()
                        self._send_response(
                            socket, ident, ERROR, b"TimeoutError: wait_write timed out"
                        )
                        continue
                    break
                if q.empty():
                    empty_uuids.append(u)
            for u in empty_uuids:
                self.wait_for_write_completion.pop(u, None)

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------
    def _send_response(
        self, socket: zmq.Socket, identity: bytes, status: bytes, payload: bytes
    ) -> None:
        """Send a multipart response, ignoring errors if the client disconnected."""
        try:
            socket.send_multipart([identity, EMPTY, status, payload])
        except zmq.ZMQError as e:
            logger.debug("Failed to send response to %s: %s", identity, e)

    # ------------------------------------------------------------------
    # Other command handlers
    # ------------------------------------------------------------------
    def close_write(self, items_data: bytes) -> str:
        """Finish writing an item, making it readable and cacheable."""
        items = json.loads(items_data)
        uuid = items["uuid"]
        open_read = items["open_read"]
        self.manager.close_write(uuid, open_read)
        return json.dumps({"status": "ok"})

    def close_read(self, uuid: str) -> str:
        """Release a read reference."""
        real_uuid, _ = self._resolve_read_token(uuid)
        self.manager.close_read(real_uuid)
        return json.dumps({"status": "ok"})

    def pin(self, uuid: str) -> str:
        """Pin an item so it is not evicted from the LRU cache."""
        self.manager.pin(uuid)
        return json.dumps({"status": "ok"})

    def unpin(self, uuid: str) -> str:
        """Unpin an item, allowing it to be evicted if idle."""
        self.manager.unpin(uuid)
        return json.dumps({"status": "ok"})

    def delete(self, uuid: str) -> str:
        """Delete an item and free its blocks."""
        # Invalidate all read tokens associated with this item
        if uuid in self._item_to_read_tokens:
            for token in self._item_to_read_tokens.pop(uuid):
                self._read_tokens.pop(token, None)
        self.manager.delete(uuid)
        return json.dumps({"status": "ok"})

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
        # Resolve token if needed, but get_info returns info for the real item
        real_uuid, _ = self._resolve_read_token(uuid)
        info = self.manager.get_info(real_uuid)
        return json.dumps(info)

    def close(self):
        """Close the shared memory storage."""
        self._resources.close()


def _zmq_server(size: int, block_size: int, conn):
    context = zmq.Context()
    socket = None
    server = None

    try:
        # Create server and storage
        server = PagedShmServer(size, block_size)

        # Bind to an available IPC path
        address = get_open_zmq_ipc_path()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(address)

        # Notify parent process of the address
        conn.send(address)
        conn.close()

        # Command dispatcher (excluding OPEN_WRITE, OPEN_READ, DELETE, and WAIT_WRITE
        # which are handled separately because they may return None or need
        # special cleanup of waiting queues)
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
                # Terminate gracefully if the context was closed or signal received
                break

            if socket not in socks or socks[socket] != zmq.POLLIN:
                # Periodic processing: handle timeouts and pending requests
                server.defer_open_write(socket)
                server.defer_open_read(socket)  # only cleans up timeouts
                server.defer_wait_write(socket)  # cleanup for WAIT_WRITE
                continue

            # Receive request — expect at least [identity, delimiter, command]
            try:
                frames = socket.recv_multipart()
            except zmq.ZMQError as e:
                logger.error("Error receiving message: %s", e)
                # If the error is fatal (e.g. context terminated), exit loop
                if e.errno == zmq.ETERM:
                    break
                continue

            if len(frames) < 3:
                logger.warning(
                    "Received malformed message with %d frames, ignoring", len(frames)
                )
                continue

            identity, delimiter, command, *payloads = frames
            if delimiter != EMPTY:
                logger.warning(
                    "Invalid delimiter in message from %s, ignoring", identity
                )
                continue

            # ------------------------------------------------------------------
            # Handle SHUTDOWN command before entering the normal dispatch table
            # ------------------------------------------------------------------
            if command == SHUTDOWN:
                logger.info("Received SHUTDOWN command from %s, exiting", identity)
                server._send_response(socket, identity, OK, b"shutting down")
                break

            # Handle OPEN_WRITE and OPEN_READ separately
            if command == OPEN_WRITE:
                try:
                    result = server.open_write(payloads[0].decode("utf-8"), identity)
                    if result is not None:
                        server._send_response(
                            socket, identity, OK, result.encode("utf-8")
                        )
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}".encode()
                    server._send_response(socket, identity, ERROR, error_msg)
                    logger.warning("Command OPEN_WRITE failed: %s", e)
                continue

            if command == OPEN_READ:
                try:
                    result = server.open_read(payloads[0].decode("utf-8"), identity)
                    if result is not None:
                        server._send_response(
                            socket, identity, OK, result.encode("utf-8")
                        )
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}".encode()
                    server._send_response(socket, identity, ERROR, error_msg)
                    logger.warning("Command OPEN_READ failed: %s", e)
                continue

            # Handle WAIT_WRITE
            if command == WAIT_WRITE:
                try:
                    result = server.wait_write(payloads[0].decode("utf-8"), identity)
                    if result is not None:
                        server._send_response(
                            socket, identity, OK, result.encode("utf-8")
                        )
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}".encode()
                    server._send_response(socket, identity, ERROR, error_msg)
                    logger.warning("Command WAIT_WRITE failed: %s", e)
                continue

            # Handle DELETE separately to clear pending open_read and wait_write waiters
            if command == DELETE:
                uuid = payloads[0].decode("utf-8")
                try:
                    # First, attempt to delete; if it fails, raise an error
                    server.delete(uuid)
                    # Clear waiters for both open_read and wait_write
                    q = server.wait_for_open_read.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident, _ = q.get()
                            server._send_response(
                                socket,
                                ident,
                                ERROR,
                                b"Item deleted while waiting"
                            )
                    # (wait_write waiters)
                    q = server.wait_for_write_completion.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident = q.get()
                            server._send_response(
                                socket,
                                ident,
                                ERROR,
                                b"Item deleted while waiting"
                            )
                    server._send_response(
                        socket, identity, OK, b'{"status":"ok"}'
                    )
                    server.defer_open_write(socket)
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}".encode()
                    server._send_response(socket, identity, ERROR, error_msg)
                    logger.warning("DELETE failed: %s", e)
                continue

            # Dispatch other commands
            handler_info = handlers.get(command)
            if handler_info is None:
                server._send_response(
                    socket,
                    identity,
                    ERROR,
                    f"Unknown command: "
                    f"{command.decode('utf-8', errors='replace')}".encode(),
                )
                continue

            handler, requires_payload = handler_info
            success = False
            try:
                if requires_payload:
                    param = payloads[0].decode("utf-8")
                    result = handler(param)
                else:
                    result = handler()

                if result is not None:
                    server._send_response(
                        socket, identity, OK, result.encode("utf-8")
                    )
                success = True
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}".encode()
                server._send_response(socket, identity, ERROR, error_msg)
                logger.info(
                    "Command %s failed: %s",
                    command.decode("utf-8", errors="replace"),
                    e,
                )

            if success and command == CLOSE_WRITE:
                # Wake up open_read waiters for this specific uuid
                try:
                    close_req = json.loads(payloads[0].decode("utf-8"))
                    uuid = close_req.get("uuid")
                    if uuid:
                        server.defer_open_read(socket, uuid)
                        server.defer_wait_write(socket, uuid)
                except Exception:
                    # If parsing fails, fall back to periodic processing
                    server.defer_open_read(socket)
                    server.defer_wait_write(socket)

            if success and command == CLOSE_READ:
                # Space may have been freed; try one open_write
                server.defer_open_write(socket)

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


class PagedShmServerProc:
    def __init__(self, size: int, block_size: int):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        proc = ctx.Process(
            target=_zmq_server,
            args=(size, block_size, child_conn),
        )

        self.proc = proc
        self.address = ""
        self.parent_conn = parent_conn
        self._finalizer = weakref.finalize(self, self.shutdown)

    def start(self, timeout: float = 5.0):
        """Start the server process and wait for its address.

        Raises:
            TimeoutError: if the child process does not send its address within timeout.
        """
        self.proc.start()
        if not self.parent_conn.poll(timeout):
            self.proc.terminate()
            self.proc.join()
            raise TimeoutError("Server process did not send address within timeout")
        self.address = self.parent_conn.recv()
        self.parent_conn.close()

    def shutdown(self, timeout: float = 5.0):
        """Gracefully stop the server process.

        Sends a SHUTDOWN command over ZMQ to wake up the poller immediately.
        If the process does not exit within the timeout, it is terminated.
        """
        if self.proc.is_alive() and self.address:
            # Attempt graceful shutdown via ZMQ SHUTDOWN command
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
                logger.debug("Failed to send SHUTDOWN command: %s", e)

        # Wait for the process to exit gracefully
        self.proc.join(timeout)

        # If still alive after timeout, terminate forcefully
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()


def maybe_start_paged_shm_server(
    model_config: ModelConfig,
) -> PagedShmServerProc | None:
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return None

    if not multimodal_config.is_paged_shm_enabled():
        return None

    paged_shm_server = PagedShmServerProc(
        size=multimodal_config.paged_shm_size,
        block_size=multimodal_config.paged_shm_block_size,
    )
    paged_shm_server.start()

    multimodal_config.paged_shm_server_address = paged_shm_server.address
    return paged_shm_server
