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
)
from .manager import PagedShmManager
from .storage import PagedShmStorage
from .types import AllocatedShmItem, ShmItem

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
        # (deadline, identity).  Grouping by uuid allows fast wake-up
        # after close_write without scanning all waiters.
        self.wait_for_open_read: dict[str, PriorityQueue] = {}

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

        item_objs = [ShmItem(**item) for item in items]
        try:
            allocated = self.manager.open_write(item_objs)
        except MemoryError:
            if timeout == 0.0:
                raise  # fail immediately
            # Queue the request with a deadline (inf for infinite wait)
            deadline = float("inf") if timeout < 0 else time.monotonic() + timeout
            self.wait_for_open_write.put((deadline, identity, item_objs))
            return None

        result = [
            asdict(
                AllocatedShmItem(
                    uuid=a.uuid, size=a.size, blocks=a.blocks, use_cache=a.use_cache
                )
            )
            for a in allocated
        ]
        return json.dumps({"status": "ok", "data": result})

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
        resp = AllocatedShmItem(
            uuid=item.uuid, size=item.size, blocks=item.blocks, use_cache=item.use_cache
        )
        return json.dumps({"status": "ok", "data": asdict(resp)})

    # ------------------------------------------------------------------
    # Deferred request processing
    # ------------------------------------------------------------------
    def defer_open_write(self, socket: zmq.Socket) -> None:
        """Process at most one pending open_write request.

        Expired requests are discarded with a timeout error. Then one
        non-expired request is attempted; if it still cannot be satisfied
        it is put back and the method returns immediately to avoid long
        scans of a large queue. The request will be retried in subsequent
        poll cycles or after a CLOSE_READ operation.
        """
        now = time.monotonic()

        # Discard all requests whose deadline has passed
        while not self.wait_for_open_write.empty():
            deadline, identity, item_objs = self.wait_for_open_write.queue[0]
            if deadline <= now:
                _, ident, _ = self.wait_for_open_write.get()
                self._send_response(
                    socket, ident, ERROR, b"TimeoutError: open_write timed out"
                )
                continue
            break  # queue is sorted by deadline; no more expired entries

        # Try to satisfy one non‑expired request (if any)
        if not self.wait_for_open_write.empty():
            deadline, identity, item_objs = self.wait_for_open_write.get()
            try:
                allocated = self.manager.open_write(item_objs)
                result = json.dumps(
                    {
                        "status": "ok",
                        "data": [
                            asdict(
                                AllocatedShmItem(
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
                self._send_response(socket, identity, OK, result.encode("utf-8"))
            except MemoryError:
                # Still not enough space; put back and stop processing
                self.wait_for_open_write.put((deadline, identity, item_objs))
            except Exception as e:
                # Other errors (e.g. duplicate UUID) are reported immediately
                self._send_response(
                    socket, identity, ERROR, f"{type(e).__name__}: {e}".encode()
                )

    def defer_open_read(self, socket: zmq.Socket, uuid: str | None = None) -> None:
        """Process pending open_read requests.

        If *uuid* is given (typically after a close_write), only the wait
        queue for that uuid is examined, allowing quick wake‑up without
        scanning all waiters.  If *uuid* is None (periodic call), only
        expired requests are discarded; satisfiable requests are left for
        the next close_write trigger.

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
                # else: still being written, leave requests in queue

            # Clean up empty queue
            if q.empty():
                self.wait_for_open_read.pop(uuid, None)

        else:
            # Periodic scan: only handle timeouts to avoid full scans
            # Collect uuids with empty queues after timeout removal
            empty_uuids = []
            for uuid, q in self.wait_for_open_read.items():
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
                    empty_uuids.append(uuid)

            # Clean up queues that became empty
            for uuid in empty_uuids:
                self.wait_for_open_read.pop(uuid, None)

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

    def run(self, conn: mp.connection.Connection) -> None:
        """Run the ZMQ server main loop.

        This method binds to an IPC address, notifies the parent via the
        given connection, and processes incoming ZMQ requests until a
        SHUTDOWN command is received or an error occurs.

        Args:
            conn: Pipe connection to send the bound address to the parent.
        """
        context = zmq.Context()
        socket = None

        try:
            # Bind to an available IPC path
            address = get_open_zmq_ipc_path()
            socket = context.socket(zmq.ROUTER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(address)

            # Notify parent process of the address
            conn.send(address)
            conn.close()

            # Command dispatcher (excluding OPEN_WRITE, OPEN_READ, and DELETE,
            # which are handled separately because they may return None or need
            # special cleanup of waiting queues)
            handlers: dict[bytes, tuple[Callable, bool]] = {
                CLOSE_WRITE: (self.close_write, True),
                CLOSE_READ: (self.close_read, True),
                PIN: (self.pin, True),
                UNPIN: (self.unpin, True),
                GET_INFO: (self.get_info, True),
                GET_MANAGER_STATE: (self.get_manager_state, False),
                GET_STORAGE_INFO: (self.get_storage_info, False),
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
                    # Periodic processing: handle timeouts and maybe satisfy one request
                    self.defer_open_write(socket)
                    self.defer_open_read(socket)  # only cleans up timeouts
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

                # Helper to send a response with standard multipart format
                def _send_response(frames: list):
                    try:
                        socket.send_multipart(frames)
                    except zmq.ZMQError as e:
                        logger.debug("Failed to send response to %s: %s", frames[0], e)

                # ------------------------------------------------------------------
                # Handle SHUTDOWN command before entering the normal dispatch table
                # ------------------------------------------------------------------
                if command == SHUTDOWN:
                    logger.info("Received SHUTDOWN command from %s, exiting", identity)
                    _send_response([identity, EMPTY, OK, b"shutting down"])
                    break

                # Handle OPEN_WRITE and OPEN_READ separately
                if command == OPEN_WRITE:
                    try:
                        result = self.open_write(payloads[0].decode("utf-8"), identity)
                        if result is not None:
                            _send_response([identity, EMPTY, OK, result.encode("utf-8")])
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}".encode()
                        _send_response([identity, EMPTY, ERROR, error_msg])
                        logger.warning("Command OPEN_WRITE failed: %s", e)
                    continue

                if command == OPEN_READ:
                    try:
                        result = self.open_read(payloads[0].decode("utf-8"), identity)
                        if result is not None:
                            _send_response([identity, EMPTY, OK, result.encode("utf-8")])
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}".encode()
                        _send_response([identity, EMPTY, ERROR, error_msg])
                        logger.warning("Command OPEN_READ failed: %s", e)
                    continue

                # Handle DELETE separately to clear pending open_read waiters
                if command == DELETE:
                    uuid = payloads[0].decode("utf-8")
                    try:
                        # Clear waiting open_read requests for this uuid
                        q = self.wait_for_open_read.pop(uuid, None)
                        if q is not None:
                            while not q.empty():
                                _, ident = q.get()
                                _send_response(
                                    [ident, EMPTY, ERROR, b"Item deleted while waiting"]
                                )
                        self.delete(uuid)
                        _send_response([identity, EMPTY, OK, b'{"status":"ok"}'])
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}".encode()
                        _send_response([identity, EMPTY, ERROR, error_msg])
                        logger.warning("DELETE failed: %s", e)
                    continue

                # Dispatch other commands
                handler_info = handlers.get(command)
                if handler_info is None:
                    _send_response(
                        [
                            identity,
                            EMPTY,
                            ERROR,
                            f"Unknown command: "
                            f"{command.decode('utf-8', errors='replace')}".encode(),
                        ]
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
                        _send_response([identity, EMPTY, OK, result.encode("utf-8")])
                    success = True
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}".encode()
                    _send_response([identity, EMPTY, ERROR, error_msg])
                    logger.warning(
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
                            self.defer_open_read(socket, uuid)
                    except Exception:
                        # If parsing fails, fall back to periodic processing
                        self.defer_open_read(socket)

                if success and command == CLOSE_READ:
                    # Space may have been freed; try one open_write
                    self.defer_open_write(socket)

        except Exception as e:
            logger.exception("Fatal error in zmq_server: %s", e)
        finally:
            if socket is not None:
                with contextlib.suppress(Exception):
                    socket.close()
            if context is not None:
                with contextlib.suppress(Exception):
                    context.term()
            with contextlib.suppress(Exception):
                self.close()
            logger.debug("[shutdown] PagedShmServer stopped")


class PagedShmServerProc:
    """Process wrapper for PagedShmServer.

    Manages the lifecycle of a PagedShmServer running in a separate process.
    """

    def __init__(self, size: int, block_size: int):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        # Create the server instance; its run() method will be the process target
        self.server = PagedShmServer(size, block_size)
        proc = ctx.Process(
            target=self.server.run,
            args=(child_conn,),
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
    """Conditionally start a PagedShmServer process based on model configuration."""
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
