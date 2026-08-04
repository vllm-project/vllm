# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import asyncio
import threading

# Third Party
import msgspec
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.full_sync_sender import FullSyncSender
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    ClearWorkerMsg,
    ClearWorkerRetMsg,
    CompressWorkerMsg,
    CompressWorkerRetMsg,
    DecompressWorkerMsg,
    DecompressWorkerRetMsg,
    DeRegisterMsg,
    ErrorMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    MoveWorkerMsg,
    MoveWorkerRetMsg,
    Msg,
    PinWorkerMsg,
    PinWorkerRetMsg,
    RegisterMsg,
    RegisterRetMsg,
    WorkerMsg,
    WorkerReqMsg,
    WorkerReqRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    close_zmq_socket,
    get_ip,
    get_zmq_context,
    get_zmq_socket,
    get_zmq_socket_with_timeout,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)


class LMCacheWorker:
    """
    LMCache Worker class to handle the execution of cache operations.
    This class is responsible for receiving requests from the executor and
    executing the corresponding operations on the LMCache engine.
    Each worker is associated with a specific LMCache instance and a worker id.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        lmcache_engine: "LMCacheEngine",
    ):
        # TODO (Jiayi): "instance_id" might not be needed anymore.
        # Please consider removing it.
        self.config = config
        self.lmcache_instance_id = config.lmcache_instance_id
        if self.lmcache_instance_id is None:
            raise ValueError(
                "lmcache_instance_id is required when enable_controller=True"
            )
        self.lmcache_engine = lmcache_engine
        self.worker_id = metadata.worker_id

        self.context = get_zmq_context()

        # Load timeout configurations from extra_config (in milliseconds)
        self.socket_recv_timeout_ms = config.get_extra_config_value(
            "worker_socket_recv_timeout_ms", DEFAULT_SOCKET_RECV_TIMEOUT_MS
        )
        self.socket_send_timeout_ms = config.get_extra_config_value(
            "worker_socket_send_timeout_ms", DEFAULT_SOCKET_SEND_TIMEOUT_MS
        )

        if config.controller_pull_url is None:
            raise ValueError(
                "controller_pull_url is required when enable_controller=True"
            )

        controller_pull_url = config.controller_pull_url
        self.push_socket = get_zmq_socket(
            self.context,
            controller_pull_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        if config.controller_reply_url is not None:
            self.controller_rep_url = config.controller_reply_url
            self._create_req_socket()

        # Heartbeat socket will be created dynamically after register
        # based on heartbeat_url returned from controller
        self.heartbeat_socket: Optional[zmq.asyncio.Socket] = None
        self.controller_heartbeat_url: Optional[str] = None

        # metadata.world_size comes from vLLM's parallel_config.world_size.
        # For MLA models, vLLM divides this by tp_size (e.g. TP=8 PP=1 on
        # 8 GPUs → world_size=1), so it may be much smaller than the total
        # GPU count.  For non-MLA models it equals TP × PP.
        #
        # get_lmcache_worker_ids() decides which workers run an LMCache
        # instance: [0] for MLA (only one worker needed since KV caches
        # are not TP-sharded), or range(world_size) for non-MLA.
        #
        # We use >= because extra ports are harmless — port selection
        # indexes by worker_id or lmcache_worker_ids position, so
        # trailing entries are never bound to sockets.
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        if not lmcache_worker_ids:
            # start lmcache worker on all ranks;
            # need at least one port per rank (world_size)
            if len(config.lmcache_worker_ports) < metadata.world_size:
                raise ValueError(
                    f"lmcache_worker_ports must have at least {metadata.world_size} "
                    f"port(s) (world_size), got {len(config.lmcache_worker_ports)}"
                )
            lmcache_worker_port = config.lmcache_worker_ports[self.worker_id]
        else:
            # start lmcache worker on given worker ids;
            # need at least one port per explicitly listed worker
            if len(config.lmcache_worker_ports) < len(lmcache_worker_ids):
                raise ValueError(
                    f"lmcache_worker_ports must have at least "
                    f"{len(lmcache_worker_ids)} "
                    f"port(s) (one per lmcache worker), "
                    f"got {len(config.lmcache_worker_ports)}"
                )
            index = lmcache_worker_ids.index(self.worker_id)
            lmcache_worker_port = config.lmcache_worker_ports[index]

        self.lmcache_worker_internal_url = f"*:{lmcache_worker_port}"
        self.lmcache_worker_ip = get_ip()
        self.lmcache_worker_port = lmcache_worker_port

        self.p2p_init_url = None
        if config.enable_p2p:
            self.p2p_host = config.p2p_host
            self.p2p_init_port = config.p2p_init_ports[self.worker_id]
            self.p2p_init_url = f"{self.p2p_host}:{self.p2p_init_port}"

        self.reply_socket = get_zmq_socket(
            self.context,
            self.lmcache_worker_internal_url,
            protocol="tcp",
            role=zmq.REP,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        logger.info("Reply socket established at %s", self.lmcache_worker_internal_url)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="lmcache-worker-thread"
        )
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)

        self.msg_queue: asyncio.Queue[WorkerMsg] = asyncio.Queue()

        # Full sync sender (initialized lazily when needed)
        self._full_sync_sender: Optional["FullSyncSender"] = None

    async def register(self):
        """
        Register the lmcache worker with the controller via DEALER-ROUTER.

        This method sends a RegisterMsg and waits for RegisterRetMsg
        which contains extra_config (e.g., heartbeat_url).
        """
        assert self.lmcache_instance_id is not None
        logger.info(
            "Registering lmcache instance-worker: %s",
            (self.lmcache_instance_id, self.worker_id),
        )

        register_msg = RegisterMsg(
            instance_id=self.lmcache_instance_id,
            worker_id=self.worker_id,
            ip=self.lmcache_worker_ip,
            port=self.lmcache_worker_port,
            peer_init_url=self.p2p_init_url,
        )

        # Send via DEALER socket (empty frame + payload) and wait for response
        try:
            await self.req_socket.send_multipart(
                [b"", msgspec.msgpack.encode(register_msg)]
            )
            # DEALER receives: [empty_frame, payload]
            frames = await self.req_socket.recv_multipart()
            serialized_ret_msg = frames[-1]
            ret_msg = msgspec.msgpack.decode(serialized_ret_msg, type=Msg)

            if isinstance(ret_msg, RegisterRetMsg):
                self._process_register_response(ret_msg)
            else:
                logger.warning("Unexpected register response type: %s", type(ret_msg))
        except zmq.ZMQError as e:
            logger.error("Failed to register with controller: %s", e)
            raise

    def _process_register_response(self, ret_msg: RegisterRetMsg):
        """Process RegisterRetMsg and initialize components based on extra_config."""
        extra_config = ret_msg.extra_config

        # Initialize heartbeat socket if heartbeat_url is provided
        heartbeat_url = extra_config.get("heartbeat_url")
        if heartbeat_url:
            logger.info("Received heartbeat_url from controller: %s", heartbeat_url)
            self.controller_heartbeat_url = heartbeat_url
            self._create_heartbeat_socket()
        else:
            logger.info("No dedicated heartbeat_url provided by controller")

    def deregister(self):
        """
        De-register the lmcache worker from the controller.
        """
        assert self.lmcache_instance_id is not None
        self.put_msg(
            DeRegisterMsg(
                instance_id=self.lmcache_instance_id,
                worker_id=self.worker_id,
                ip=self.lmcache_worker_ip,
                port=self.lmcache_worker_port,
            )
        )

    async def async_put_and_wait_msg(
        self,
        msg: WorkerReqMsg,
    ) -> WorkerReqRetMsg:
        """
        Send a message to the controller and wait for the response.

        This method handles different types of WorkerReqMsg using appropriate sockets:
        - HeartbeatMsg: Uses dedicated heartbeat socket
        - Other messages (RegisterMsg, BatchedP2PLookupMsg, FullSyncStartMsg,
          FullSyncStatusMsg): Uses DEALER socket (req_socket)

        Note: With DEALER-ROUTER mode, we no longer need to separate FullSync
        messages to heartbeat socket since DEALER supports async concurrent requests.
        """
        # Send heartbeat via dedicated heartbeat socket
        if isinstance(msg, HeartbeatMsg):
            return await self._send_heartbeat_msg(msg)

        # Send other messages via DEALER socket
        try:
            # DEALER socket: send [empty_frame, payload]
            await self.req_socket.send_multipart([b"", msgspec.msgpack.encode(msg)])
            frames = await self.req_socket.recv_multipart()
            # DEALER receives: [empty_frame, payload]
            serialized_ret_msg = frames[-1]
            ret_msg = msgspec.msgpack.decode(serialized_ret_msg, type=Msg)
            return ret_msg
        except zmq.Again as e:
            logger.error("Timeout occurred, recreating socket. Error: %s", e)
            self._recreate_req_socket()
            return self._on_request_failure(msg)
        except zmq.ZMQError as e:
            logger.error("ZMQ error occurred, recreating socket. Error: %s", e)
            self._recreate_req_socket()
            return self._on_request_failure(msg)
        except Exception as e:
            logger.error("Error happens in lmcache worker req_socket. Error: %s", e)
            return self._on_request_failure(msg)

    def _create_req_socket(self):
        self.req_socket = get_zmq_socket_with_timeout(
            self.context,
            self.controller_rep_url,
            "tcp",
            zmq.DEALER,  # type: ignore[attr-defined]
            "connect",
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )

    def _create_heartbeat_socket(self):
        logger.info(
            "Creating heartbeat socket to connect to: %s, "
            "recv_timeout: %dms, send_timeout: %dms",
            self.controller_heartbeat_url,
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )
        self.heartbeat_socket = get_zmq_socket_with_timeout(
            self.context,
            self.controller_heartbeat_url,
            "tcp",
            zmq.DEALER,  # type: ignore[attr-defined]
            "connect",
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )

    def _recreate_heartbeat_socket(self):
        try:
            self.heartbeat_socket.close(linger=0)
        except Exception as e:
            logger.error("Error closing heartbeat socket: %s", e)
        self._create_heartbeat_socket()

    def _recreate_req_socket(self):
        try:
            self.req_socket.close(linger=0)
        except Exception as e:
            logger.error("Error closing req socket: %s", e)
        self._create_req_socket()

    def _get_full_sync_sender(self):
        """Lazy initialization of FullSyncSender"""
        if self._full_sync_sender is None:
            # Get the local_cpu_backend from lmcache_engine
            local_cpu_backend = self.lmcache_engine.storage_manager.local_cpu_backend
            self._full_sync_sender = FullSyncSender(
                config=self.config,
                worker=self,
                lmcache_engine=self.lmcache_engine,
                local_cpu_backend=local_cpu_backend,
            )
        return self._full_sync_sender

    def _on_request_failure(self, msg: WorkerReqMsg) -> WorkerReqRetMsg:
        """
        Create a default return message when worker -> controller
        request encounters an error (e.g., timeout, ZMQ error).
        """
        if isinstance(msg, BatchedP2PLookupMsg):
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])
        elif isinstance(msg, HeartbeatMsg):
            return HeartbeatRetMsg()  # No command by default
        elif isinstance(msg, FullSyncStartMsg):
            return FullSyncStartRetMsg(
                sync_id=msg.sync_id,
                accepted=False,
                error_msg="Communication error",
            )
        elif isinstance(msg, FullSyncStatusMsg):
            return FullSyncStatusRetMsg(
                sync_id=msg.sync_id,
                is_complete=False,
                global_progress=0.0,
                can_exit_freeze=False,
            )
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")

    def put_msg(self, msg: WorkerMsg):
        """
        Put a message into the message queue.
        """
        # TODO(Jiayi): This might introduce ~0.05ms latency than
        # a normal function call.
        # Not sure how much overhead is blocking though.
        self.loop.call_soon_threadsafe(self.msg_queue.put_nowait, msg)

    async def batched_get_msg(self, max_bsz: int = 50) -> list[WorkerMsg]:
        """
        Get a batch of messages from the message queue.
        """
        batch = []

        # use blocking get for the first msg
        try:
            item = await self.msg_queue.get()
            batch.append(item)
        except asyncio.CancelledError:
            return batch  # shutdown path

        for _ in range(max_bsz - 1):
            try:
                item = self.msg_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        return batch

    async def _send_heartbeat_msg(self, msg: HeartbeatMsg) -> HeartbeatRetMsg:
        """
        Send heartbeat message via dedicated heartbeat DEALER socket.
        This is separate from async_put_and_wait_msg to keep heartbeat independent.
        """
        if self.heartbeat_socket is None:
            logger.warning("Heartbeat socket is not initialized")
            return HeartbeatRetMsg()
        try:
            # DEALER socket: send [empty_frame, payload]
            logger.info("Sending heartbeat message to controller...")
            await self.heartbeat_socket.send_multipart(
                [b"", msgspec.msgpack.encode(msg)]
            )
            logger.info("Heartbeat message sent, waiting for response...")
            frames = await self.heartbeat_socket.recv_multipart()
            logger.info("Received heartbeat response with %d frames", len(frames))
            # DEALER receives: [empty_frame, payload]
            serialized_ret_msg = frames[-1]
            ret_msg = msgspec.msgpack.decode(serialized_ret_msg, type=Msg)
            return ret_msg
        except zmq.Again as e:
            logger.error("Heartbeat timeout occurred, recreating socket. Error: %s", e)
            self._recreate_heartbeat_socket()
            return HeartbeatRetMsg()
        except zmq.ZMQError as e:
            logger.error(
                "Heartbeat ZMQ error occurred, recreating socket. Error: %s", e
            )
            self._recreate_heartbeat_socket()
            return HeartbeatRetMsg()
        except Exception as e:
            logger.error("Error happens in heartbeat socket. Error: %s", e)
            return HeartbeatRetMsg()

    async def heartbeat(self):
        """
        Send periodic heartbeats to the controller (DEALER-ROUTER mode).

        Process any commands received in the heartbeat response.
        Uses dedicated heartbeat socket to avoid blocking from other requests.
        """
        enable_heartbeat = (
            self.config.lmcache_worker_heartbeat_time is not None
            and self.config.lmcache_worker_heartbeat_time > 0
            and self.heartbeat_socket is not None
        )
        if enable_heartbeat:
            await asyncio.sleep(self.config.lmcache_worker_heartbeat_delay_time)
            logger.info(
                "Start heartbeat in %s : %s, delay time: %ss, heartbeat time: %ss",
                self.lmcache_instance_id,
                self.worker_id,
                self.config.lmcache_worker_heartbeat_delay_time,
                self.config.lmcache_worker_heartbeat_time,
            )
            while True:
                # Send heartbeat via dedicated heartbeat socket
                heartbeat_msg = HeartbeatMsg(
                    instance_id=self.lmcache_instance_id,
                    worker_id=self.worker_id,
                    ip=self.lmcache_worker_ip,
                    port=self.lmcache_worker_port,
                    peer_init_url=self.p2p_init_url,
                )

                try:
                    ret_msg = await self._send_heartbeat_msg(heartbeat_msg)

                    if isinstance(ret_msg, HeartbeatRetMsg):
                        self._handle_heartbeat_commands(ret_msg)
                    else:
                        logger.warning(
                            "Unexpected heartbeat response type: %s", type(ret_msg)
                        )
                except Exception as e:
                    logger.error("Error during heartbeat: %s", e)

                await asyncio.sleep(self.config.lmcache_worker_heartbeat_time)

    def _handle_heartbeat_commands(self, ret_msg: HeartbeatRetMsg):
        """
        Handle commands received in heartbeat response.

        Uses polymorphic dispatch - each command class implements its own
        execute() method. Commands are executed sequentially.
        """
        if not ret_msg.has_commands():
            return

        for command in ret_msg.commands:
            logger.info(
                "Executing heartbeat command: %s",
                command.describe(),
            )
            try:
                command.execute(self)
            except NotImplementedError:
                logger.warning(
                    "Command %s.execute() not implemented yet",
                    type(command).__name__,
                )
            except Exception as e:
                logger.error(
                    "Error executing command %s: %s",
                    type(command).__name__,
                    e,
                )

    async def push(self):
        while True:
            try:
                msgs = await self.batched_get_msg()
                logger.debug("Sending %d messages", len(msgs))
                self.push_socket.send_multipart(
                    [msgspec.msgpack.encode(msg) for msg in msgs]
                )

            except Exception as e:
                logger.error("Push error: %s", e)

    async def handle_request(self):
        """
        Handle incoming requests (control msgs) from the controller.
        """
        while True:
            try:
                serialized_request = await self.reply_socket.recv()
                request = msgspec.msgpack.decode(serialized_request, type=Msg)
                logger.debug("Received message: %s", request)
                if isinstance(request, MoveWorkerMsg):
                    tokens = request.tokens
                    old_position = request.old_position
                    new_position = request.new_position
                    do_copy = request.copy
                    worker_event_id = request.worker_event_id

                    # Intra node move
                    if new_position[0] == self.lmcache_worker_internal_url:
                        # TODO(Jiayi): currently we only support moving from
                        # local disk to local cpu.
                        assert old_position[1] == "LocalDiskBackend"
                        assert new_position[1] == "LocalCPUBackend"
                        assert do_copy

                        # TODO(Jiayi): We need to align prefetch and move.
                        logger.debug("Executing prefetch operation.")
                        raise NotImplementedError(
                            "Prefetch from controller is not implemented yet."
                        )
                    else:
                        assert new_position[1] == "LocalCPUBackend", (
                            "Only support moving to cpu for now."
                        )
                        logger.debug("Executing cross-node move operation.")
                        num_tokens = self.lmcache_engine.move(
                            tokens=tokens,
                            old_position=old_position,
                            new_position=new_position,
                            event_id=worker_event_id,
                            do_copy=do_copy,
                        )

                    # TODO(Jiayi): LMCache needs to have an event tracking
                    # pool to enable more advanced control-plane optims.
                    # For now, we use a dummy `event_id`.
                    serialized_ret_msg = msgspec.msgpack.encode(
                        MoveWorkerRetMsg(num_tokens=num_tokens)
                    )
                elif isinstance(request, CompressWorkerMsg):
                    num_compressed_tokens = self.lmcache_engine.compress(
                        tokens=request.tokens,
                        method=request.method,
                        location=request.location,
                        event_id=request.worker_event_id,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        CompressWorkerRetMsg(num_tokens=num_compressed_tokens)
                    )
                elif isinstance(request, DecompressWorkerMsg):
                    num_decompressed_tokens = self.lmcache_engine.decompress(
                        tokens=request.tokens,
                        method=request.method,
                        location=request.location,
                        event_id=request.worker_event_id,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        DecompressWorkerRetMsg(num_tokens=num_decompressed_tokens)
                    )
                elif isinstance(request, PinWorkerMsg):
                    num_pinned_tokens = self.lmcache_engine.lookup(
                        tokens=request.tokens,
                        search_range=[request.location],
                        lookup_id=request.worker_event_id,
                        pin=True,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        PinWorkerRetMsg(num_tokens=num_pinned_tokens)
                    )
                elif isinstance(request, ClearWorkerMsg):
                    num_cleared_tokens = self.lmcache_engine.clear(
                        locations=[request.location],
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        ClearWorkerRetMsg(num_tokens=num_cleared_tokens)
                    )
                elif isinstance(request, HealthWorkerMsg):
                    error_code = self.lmcache_engine.health()
                    serialized_ret_msg = msgspec.msgpack.encode(
                        HealthWorkerRetMsg(error_code=error_code)
                    )
                else:
                    logger.error("Unknown message: %s", request)
                    serialized_ret_msg = msgspec.msgpack.encode(
                        ErrorMsg(error=f"Unknown message: {request}")
                    )

                await self.reply_socket.send(serialized_ret_msg)
            except Exception as e:
                logger.error("Worker error: %s", e)
                serialized_ret_msg = msgspec.msgpack.encode(
                    ErrorMsg(error=f"Worker error: {e}")
                )
                await self.reply_socket.send(serialized_ret_msg)

    async def start_all(self):
        try:
            # Register first to get heartbeat_url before starting heartbeat task
            await self.register()

            logger.info(
                f"Starting lmcache worker {self.worker_id}"
                f"for instance {self.lmcache_instance_id}"
            )
            await asyncio.gather(
                self.push(),
                self.handle_request(),
                self.heartbeat(),
            )
        except Exception as e:
            logger.error(
                f"Instance {self.lmcache_instance_id}, "
                f"worker {self.worker_id} error: {e}"
            )

    def close(self):
        self.deregister()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
        self.loop.close()
        close_zmq_socket(self.push_socket)
        close_zmq_socket(self.reply_socket)
        if self.heartbeat_socket is not None:
            close_zmq_socket(self.heartbeat_socket)
        if hasattr(self, "req_socket"):
            close_zmq_socket(self.req_socket)
