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
        # These clients wait until the item becomes readable (ref_count >= 0)
        # but do NOT acquire a read lock.
        self.wait_for_write_completion: dict[str, PriorityQueue] = {}

    # ------------------------------------------------------------------
    # Helper for constructing write responses
    # ------------------------------------------------------------------
    def _build_write_response(self, allocated: list) -> str:
        """Convert a list of allocated ShmSlot to a JSON response."""
        return json.dumps(
            {
                "status": "ok",
                "data": [
                    asdict(
                        ShmAllocation(
                            uuid=a.uuid,
                            size=a.size,
                            blocks=a.blocks,
                            use_cache=a.use_cache,
                        )
                    )
                    for a in allocated
                ],
            }
        )

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

        return self._build_write_response(allocated)

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

        info = self.manager.get_info(uuid)  # raises ValueError if uuid unknown
        if info["ref_count"] < 0:  # item is being written
            if timeout == 0.0:
                # Simulate the same error as an immediate open_read
                return self._open_read(uuid)  # will raise ValueError
            deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
            # Get or create the per-uuid wait queue
            q = self.wait_for_open_read.setdefault(uuid, PriorityQueue())
            q.put((deadline, identity))
            return None
        else:
            return self._open_read(uuid)

    def _open_read(self, uuid: str) -> str:
        """Internal helper to open a read reference and build the response."""
        item = self.manager.open_read(uuid)
        resp = ShmAllocation(
            uuid=item.uuid, size=item.size, blocks=item.blocks, use_cache=item.use_cache
        )
        return json.dumps({"status": "ok", "data": asdict(resp)})

    def wait_write(self, data: bytes, identity: bytes) -> str | None:
        """
        Wait for the item with the given UUID to become readable (i.e., the
        write operation has been closed). Unlike open_read, this does NOT
        acquire a read lock; it only signals that the item is available.
        If the item is already readable, returns immediately with a success
        status. Otherwise, the request is queued and the client will be
        notified when close_write is called for this UUID.

        Args:
            data: JSON payload containing 'uuid' and optional 'timeout'.
            identity: ZMQ identity of the requesting client.

        Returns:
            JSON response string if the item is already readable,
            None if the request was queued (item is being written).

        Raises:
            ValueError: if uuid does not exist or timeout=0 and item is being written.
        """
        req = json.loads(data)
        uuid = req["uuid"]
        timeout = float(req.get("timeout", 0.0))

        # Check if the item exists and is not being written.
        try:
            info = self.manager.get_info(uuid)
        except ValueError:
            # Re-raise with a clear message; the server will catch and send ERROR.
            raise ValueError(f"UUID {uuid} not found") from None

        if info["ref_count"] >= 0:
            # Already readable: respond immediately with success.
            return json.dumps({"status": "ok"})
        else:  # ref_count == REF_WRITING
            if timeout == 0.0:
                # Immediate failure: item is still being written.
                raise RuntimeError(f"UUID {uuid} is still being written")
            deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
            q = self.wait_for_write_completion.setdefault(uuid, PriorityQueue())
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

        # Discard all requests whose deadline has passed
        while not self.wait_for_open_write.empty():
            deadline, identity, _ = self.wait_for_open_write.queue[0]
            if deadline <= now:
                _, ident, _ = self.wait_for_open_write.get()
                self._send_response(
                    socket, ident, ERROR, b"TimeoutError: open_write timed out"
                )
                continue
            break  # queue is sorted by deadline; no more expired entries

        # Try to satisfy up to max_attempts non‑expired requests
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
                self.wait_for_open_write.get()  # success, remove from queue
                response = self._build_write_response(allocated)
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
            # Process a specific uuid queue
            q = self.wait_for_open_read.get(uuid)
            if q is None:
                return

            # Discard expired requests
            while not q.empty():
                deadline, identity = q.queue[0]
                if deadline <= now:
                    _, ident = q.get()
                    self._send_response(
                        socket, ident, ERROR, b"TimeoutError: open_read timed out"
                    )
                    continue
                break

            # Satisfy as many requests as possible (the item is now readable)
            if not q.empty():
                info = self.manager.get_info(uuid)
                if info["ref_count"] >= 0:
                    while not q.empty():
                        deadline, identity = q.get()
                        try:
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
                    deadline, identity = q.queue[0]
                    if deadline <= now:
                        _, ident = q.get()
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
                    else:
                        # Still being written? This should not happen if we are
                        # called after close_write; but if it does, leave queue
                        # for later processing.
                        pass
                except ValueError:
                    # Item disappeared; send error to all waiters
                    while not q.empty():
                        _, ident = q.get()
                        self._send_response(socket, ident, ERROR, b"UUID not found")

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
        self.manager.close_read(uuid)
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
        info = self.manager.get_info(uuid)
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
                    # Deletion succeeded; now clear all waiters for this uuid
                    # (open_read waiters)
                    q = server.wait_for_open_read.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident = q.get()
                            server._send_response(
                                socket, ident, ERROR, b"Item deleted while waiting"
                            )
                    # (wait_write waiters)
                    q = server.wait_for_write_completion.pop(uuid, None)
                    if q is not None:
                        while not q.empty():
                            _, ident = q.get()
                            server._send_response(
                                socket, ident, ERROR, b"Item deleted while waiting"
                            )
                    server._send_response(socket, identity, OK, b'{"status":"ok"}')
                    # Space may have been freed, try open_write waiters
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
                    server._send_response(socket, identity, OK, result.encode("utf-8"))
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
