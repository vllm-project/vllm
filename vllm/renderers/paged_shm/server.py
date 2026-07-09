# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import multiprocessing as mp

import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path

from .manager import Item, PagedShmManager
from .storage import PagedShmStorage

POLL_INTERVAL = 1000

logger = init_logger(__name__)


class PagedShmServer:
    """Server‑side wrapper that exposes PagedShmManager over ZMQ."""

    def __init__(self, size: int, block_size: int):
        self.storage = PagedShmStorage(size, block_size, pin=False)
        self.manager = PagedShmManager(size, block_size)

        self.size = self.storage.size
        self.shm_name = self.storage.name
        self.n_block = self.storage.n_block
        self.block_size = self.storage.block_size

    def open_write(self, items_data: bytes) -> str:
        """Allocate blocks for a batch of items to be written.

        Exceptions raised by the manager (ValueError for duplicate UUIDs,
        MemoryError if space is insufficient) are propagated directly to the
        caller so the client receives detailed error messages.
        """
        items = json.loads(items_data)
        item_objs = [Item(**item) for item in items]
        allocated = self.manager.open_write(item_objs)
        result = [
            {
                "uuid": a.uuid,
                "size": a.size,
                "use_cache": a.use_cache,
                "blocks": a.blocks,
                "ref_count": a.ref_count,
            }
            for a in allocated
        ]
        return json.dumps(result)

    def close_write(self, uuid: str) -> str:
        """Finish writing an item, making it readable and cacheable."""
        self.manager.close_write(uuid)
        return json.dumps({"status": "ok"})

    def open_read(self, uuid: str) -> str:
        """Acquire a read reference to an item, returning its block list and size."""
        item = self.manager.open_read(uuid)
        return json.dumps({"status": "ok", "blocks": item.blocks, "size": item.size})

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
        return json.dumps(info)

    def close(self):
        """Close the shared memory storage."""
        self.storage.close()


def zmq_server(size: int, block_size: int, conn, stop_event: mp.Event):
    context = zmq.Context()
    socket = None
    server = None

    try:
        # Create server and storage
        server = PagedShmServer(size, block_size)

        # Bind to an available IPC path
        address = get_open_zmq_ipc_path()
        socket = context.socket(zmq.ROUTER)
        socket.bind(address)

        # Notify parent process of the address
        conn.send(address)
        conn.close()

        # Command dispatcher: {command_bytes: (handler, requires_payload)}
        handlers = {
            b"open_write": (server.open_write, True),
            b"close_write": (server.close_write, True),
            b"open_read": (server.open_read, True),
            b"close_read": (server.close_read, True),
            b"pin": (server.pin, True),
            b"unpin": (server.unpin, True),
            b"delete": (server.delete, True),
            b"get_manager_state": (server.get_manager_state, False),
            b"get_storage_info": (server.get_storage_info, False),
        }

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        logger.info("PagedShmServer started at %s", address)

        while not stop_event.is_set():
            try:
                socks = dict(poller.poll(POLL_INTERVAL))
            except (zmq.ZMQError, KeyboardInterrupt, EOFError):
                # Terminate gracefully if the context was closed or signal received
                break

            if socket not in socks or socks[socket] != zmq.POLLIN:
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
            if delimiter != b"":
                logger.warning(
                    "Invalid delimiter in message from %s, ignoring", identity
                )
                continue

            def _send_response(socket: zmq.Socket, frames: list):
                """Send a multipart response, ignoring errors if the client
                disconnected."""
                try:
                    socket.send_multipart(frames)
                except zmq.ZMQError as e:
                    # Client may have gone away – this is not a server error
                    logger.debug("Failed to send response to %s: %s", frames[0], e)

            # Dispatch command
            handler_info = handlers.get(command)
            if handler_info is None:
                response_frames = [
                    identity,
                    b"",
                    b"ERROR",
                    f"Unknown command: "
                    f"{command.decode('utf-8', errors='replace')}".encode(),
                ]
                _send_response(socket, response_frames)
                continue

            handler, requires_payload = handler_info
            try:
                if requires_payload:
                    param = payloads[0].decode("utf-8")
                    result = handler(param)
                else:
                    result = handler()
                response_frames = [identity, b"", b"OK", result.encode("utf-8")]
            except Exception as e:
                # Include exception type and message for client debugging
                error_msg = f"{type(e).__name__}: {e}".encode()
                response_frames = [identity, b"", b"ERROR", error_msg]
                logger.warning(
                    "Command %s failed: %s",
                    command.decode("utf-8", errors="replace"),
                    e,
                )

            _send_response(socket, response_frames)

    except Exception as e:
        logger.exception("Fatal error in zmq_server: %s", e)
    finally:
        # Clean up resources in reverse order
        if socket is not None:
            with contextlib.suppress(Exception):
                socket.close()
        if context is not None:
            with contextlib.suppress(Exception):
                context.term()
        if server is not None:
            with contextlib.suppress(Exception):
                server.close()
        logger.info("PagedShmServer stopped.")
