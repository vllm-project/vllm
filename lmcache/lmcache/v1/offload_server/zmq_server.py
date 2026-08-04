# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import os
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.offload_server.abstract_server import OffloadServerInterface
from lmcache.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)


class ZMQOffloadServer(OffloadServerInterface):
    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        tp_rank: int,
    ):
        metadata = lmcache_engine.metadata
        self.ctx = get_zmq_context(use_asyncio=False)
        offload_rpc_port = int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100))
        engine_id = metadata.engine_id or "default"
        socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "offload", offload_rpc_port, tp_rank
        )
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            "ipc",
            zmq.REP,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            # First Party
            from lmcache.logging import init_logger

            logger = init_logger(__name__)

            while self.running:
                try:
                    frame = self.socket.recv(copy=False)
                    offload_msg = msgspec.msgpack.decode(frame, type=OffloadMsg)
                    result = self.offload(
                        offload_msg.hashes,
                        offload_msg.slot_mapping,
                        offload_msg.offsets,
                    )
                    response = OffloadRetMsg(success=result)
                    response = msgspec.msgpack.encode(response)
                    self.socket.send(response)
                except zmq.ZMQError as e:
                    # Socket was closed, exit gracefully
                    if not self.running:
                        logger.info("ZMQ socket closed, exiting offload server thread")
                        break
                    logger.error(f"ZMQ error in offload server: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in offload server: {e}")
                    if not self.running:
                        break

        self.thread = threading.Thread(
            target=process_request, daemon=True, name="offload-server-thread"
        )
        self.thread.start()

    def offload(
        self,
        hashes: List[int],
        slot_mapping: List[int],
        offsets: List[int],
    ) -> bool:
        self.lmcache_engine.store(
            hashes=hashes, slot_mapping=slot_mapping, offsets=offsets
        )
        return True

    def close(self) -> None:
        # First Party
        from lmcache.logging import init_logger

        logger = init_logger(__name__)

        logger.info("Closing ZMQOffloadServer...")
        self.running = False

        # Close socket to interrupt blocking recv()
        try:
            self.socket.close(linger=0)
            logger.info("ZMQ socket closed")
        except Exception as e:
            logger.warning(f"Error closing ZMQ socket: {e}")

        # Wait for thread with timeout to prevent deadlock
        if self.thread.is_alive():
            logger.info("Waiting for offload server thread to finish...")
            self.thread.join(timeout=5.0)

            if self.thread.is_alive():
                logger.warning(
                    "Offload server thread did not terminate within timeout. "
                    "Thread may be stuck in blocking recv(). "
                    "Proceeding with shutdown anyway."
                )
            else:
                logger.info("Offload server thread terminated successfully")
        else:
            logger.info("Offload server thread already stopped")
