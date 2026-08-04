# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import asyncio
import json
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controllers import KVController, RegistrationController
from lmcache.v1.cache_controller.executor import LMCacheClusterExecutor
from lmcache.v1.cache_controller.observability import (
    PrometheusLogger,
    SocketMetricsContext,
    SocketType,
)
from lmcache.v1.rpc_utils import (
    get_ip,
    get_zmq_context,
    get_zmq_socket,
)

from lmcache.v1.cache_controller.message import (  # isort: skip
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    CheckFinishMsg,
    ClearMsg,
    CompressMsg,
    DecompressMsg,
    DeRegisterMsg,
    ErrorMsg,
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStatusMsg,
    HealthMsg,
    HeartbeatMsg,
    LookupMsg,
    MoveMsg,
    Msg,
    MsgBase,
    OrchMsg,
    OrchRetMsg,
    PinMsg,
    QueryInstMsg,
    QueryWorkerInfoMsg,
    RegisterMsg,
    WorkerMsg,
    WorkerReqMsg,
    WorkerReqRetMsg,
)

logger = init_logger(__name__)

# TODO(Jiayi): Need to align the message types. For example,
# a controller should take in an control message and return
# a control message.


class LMCacheControllerManager:
    def __init__(
        self,
        controller_urls: dict[str, str],
        health_check_interval: int,
        lmcache_worker_timeout: int,
        full_sync_completion_threshold: float = 0.8,
        full_sync_timeout_s: float = 300.0,
    ):
        # Initialize stats logger
        prometheus_labels = {
            "role": "controller",
        }
        PrometheusLogger.GetOrCreate(prometheus_labels)
        self.zmq_context = get_zmq_context()
        self.controller_urls = controller_urls
        # TODO(Jiayi): We might need multiple sockets if there are more
        # controllers. For now, we use a single socket to receive messages
        # for all controllers.
        # Similarly we might need more sockets to handle different control
        # messages. For now, we use one socket to handle all control messages.

        # TODO(Jiayi): Another thing is that we might need to decoupe the
        # interactions among `handle_worker_message`, `handle_control_message`
        # and `handle_orchestration_message`. For example, in
        # `handle_orchestration_message`, we might need to call
        # `issue_control_message`. This will make the system less concurrent.

        # Micro controllers
        self.controller_pull_socket = get_zmq_socket(
            self.zmq_context,
            self.controller_urls["pull"],
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        if self.controller_urls["reply"] is not None:
            self.controller_reply_socket = get_zmq_socket(
                self.zmq_context,
                self.controller_urls["reply"],
                protocol="tcp",
                role=zmq.ROUTER,  # type: ignore[attr-defined]
                bind_or_connect="bind",
            )

        # Dedicated heartbeat socket to avoid blocking from other requests
        if self.controller_urls.get("heartbeat") is not None:
            self.controller_heartbeat_socket = get_zmq_socket(
                self.zmq_context,
                self.controller_urls["heartbeat"],
                protocol="tcp",
                role=zmq.ROUTER,  # type: ignore[attr-defined]
                bind_or_connect="bind",
            )
        else:
            self.controller_heartbeat_socket = None
        self.reg_controller = RegistrationController()
        self.kv_controller = KVController(
            registry=self.reg_controller.registry,
            full_sync_completion_threshold=full_sync_completion_threshold,
            full_sync_timeout_s=full_sync_timeout_s,
        )

        # Cluster executor
        self.cluster_executor = LMCacheClusterExecutor(
            reg_controller=self.reg_controller,
        )

        # post initialization of controllers
        self.kv_controller.post_init(
            reg_controller=self.reg_controller,
            cluster_executor=self.cluster_executor,
        )
        self.reg_controller.post_init(
            kv_controller=self.kv_controller,
            cluster_executor=self.cluster_executor,
        )
        self.health_check_interval = health_check_interval
        self.lmcache_worker_timeout = lmcache_worker_timeout

        if self.health_check_interval > 0:
            logger.info(
                "Start health check thread, interval: %s", self.health_check_interval
            )
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(
                target=self.loop.run_forever,
                daemon=True,
                name="controller-health-thread",
            )
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.health_check(), self.loop)

        # Setup socket message count metrics
        self._setup_socket_metrics()

    async def handle_worker_message(self, msg: WorkerMsg) -> None:
        if isinstance(msg, RegisterMsg):
            await self.reg_controller.register(msg)
        elif isinstance(msg, DeRegisterMsg):
            await self.reg_controller.deregister(msg)
        elif isinstance(msg, BatchedKVOperationMsg):
            await self.kv_controller.handle_batched_kv_operations(msg)
        elif isinstance(msg, FullSyncBatchMsg):
            await self.kv_controller.handle_full_sync_batch(msg)
        elif isinstance(msg, FullSyncEndMsg):
            await self.kv_controller.handle_full_sync_end(msg)
        else:
            logger.error(f"Unknown worker message type: {msg}")

    async def handle_worker_req_message(
        self, msg: WorkerReqMsg
    ) -> Union[WorkerReqRetMsg, ErrorMsg]:
        ret_msg: Union[WorkerReqRetMsg, ErrorMsg]
        if isinstance(msg, RegisterMsg):
            # Build extra_config with heartbeat_url if available
            extra_config: dict[str, str] = {}
            if self.controller_urls.get("heartbeat") is not None:
                # Convert bind address (e.g., "0.0.0.0:8082" or "*:8082")
                # to a connectable address using actual controller IP
                heartbeat_bind_url = self.controller_urls["heartbeat"]
                heartbeat_url = self._convert_bind_to_connect_url(
                    heartbeat_bind_url, worker_ip=msg.ip
                )
                extra_config["heartbeat_url"] = heartbeat_url
                logger.debug(
                    "Returning heartbeat_url to worker: %s (bind: %s)",
                    heartbeat_url,
                    heartbeat_bind_url,
                )
            ret_msg = await self.reg_controller.register(msg, extra_config)
        elif isinstance(msg, BatchedP2PLookupMsg):
            ret_msg = await self.kv_controller.batched_p2p_lookup(msg)
        elif isinstance(msg, HeartbeatMsg):
            ret_msg = await self.reg_controller.heartbeat(msg)
        elif isinstance(msg, FullSyncStartMsg):
            ret_msg = await self.kv_controller.handle_full_sync_start(msg)
        elif isinstance(msg, FullSyncStatusMsg):
            ret_msg = await self.kv_controller.handle_full_sync_status(msg)
        else:
            logger.error(f"Unknown worker request message type: {msg}")
            ret_msg = ErrorMsg(error=f"Unknown message type: {type(msg)}")
        return ret_msg

    async def handle_orchestration_message(self, msg: OrchMsg) -> OrchRetMsg:
        if isinstance(msg, LookupMsg):
            return await self.kv_controller.lookup(msg)
        elif isinstance(msg, HealthMsg):
            return await self.reg_controller.health(msg)
        elif isinstance(msg, QueryInstMsg):
            return await self.reg_controller.get_instance_id(msg)
        elif isinstance(msg, ClearMsg):
            return await self.kv_controller.clear(msg)
        elif isinstance(msg, PinMsg):
            return await self.kv_controller.pin(msg)
        elif isinstance(msg, CompressMsg):
            return await self.kv_controller.compress(msg)
        elif isinstance(msg, DecompressMsg):
            return await self.kv_controller.decompress(msg)
        elif isinstance(msg, MoveMsg):
            return await self.kv_controller.move(msg)
        elif isinstance(msg, CheckFinishMsg):
            # FIXME(Jiayi): This `check_finish` thing
            # shouldn't be implemented in kv_controller.
            return await self.kv_controller.check_finish(msg)
        elif isinstance(msg, QueryWorkerInfoMsg):
            return await self.reg_controller.query_worker_info(msg)
        else:
            logger.error(f"Unknown orchestration message type: {msg}")
            raise RuntimeError(f"Unknown orchestration message type: {msg}")

    def _setup_socket_metrics(self):
        """Setup metrics for socket message counts."""
        # Initialize message counters for observability
        self.pull_socket_message_count = 0
        self.reply_socket_message_count = 0

        # Initialize active request counters
        self.pull_socket_active_requests = 0
        self.reply_socket_active_requests = 0

        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.pull_socket_message_count.set_function(
                lambda: self.pull_socket_message_count
            )
            prometheus_logger.reply_socket_message_count.set_function(
                lambda: self.reply_socket_message_count
            )

            # Socket queue/backlog metrics
            prometheus_logger.pull_socket_has_pending.set_function(
                lambda: self._check_socket_has_pending(self.controller_pull_socket)
            )
            if self.controller_urls["reply"] is not None:
                prometheus_logger.reply_socket_has_pending.set_function(
                    lambda: self._check_socket_has_pending(self.controller_reply_socket)
                )

            # Active request metrics
            prometheus_logger.pull_socket_active_requests.set_function(
                lambda: self.pull_socket_active_requests
            )
            prometheus_logger.reply_socket_active_requests.set_function(
                lambda: self.reply_socket_active_requests
            )

    def _convert_bind_to_connect_url(
        self, bind_url: str, worker_ip: Optional[str] = None
    ) -> str:
        """Convert a bind address to a connectable address.

        Bind addresses like "0.0.0.0:port" or "*:port" cannot be used
        by workers to connect. We need to replace them with the actual
        controller IP address.

        If worker_ip is provided and matches the controller's IP, use
        127.0.0.1 for loopback connection (more reliable than external IP).

        Args:
            bind_url: The bind URL (e.g., "0.0.0.0:8082" or "*:8082")
            worker_ip: The IP address of the requesting worker (optional)

        Returns:
            A connectable URL (e.g., "192.168.1.100:8082" or "127.0.0.1:8082")
        """
        if ":" not in bind_url:
            return bind_url

        host, port = bind_url.rsplit(":", 1)
        # Replace bind-all addresses with actual IP
        if host in ("0.0.0.0", "*", ""):
            actual_ip = get_ip()
            # If worker is on the same machine, use loopback for reliability
            # This handles cases where external IP (e.g., VPN) doesn't support
            # local loopback connections
            if worker_ip is not None and worker_ip == actual_ip:
                logger.debug(
                    "Worker IP %s matches controller IP, using 127.0.0.1",
                    worker_ip,
                )
                return f"127.0.0.1:{port}"
            return f"{actual_ip}:{port}"
        return bind_url

    def _check_socket_has_pending(self, socket) -> int:
        """Check if socket has pending messages.

        Returns:
            1 if socket has pending messages, 0 otherwise
        """
        try:
            events = socket.get(zmq.EVENTS)  # type: ignore[attr-defined]
            # Check if POLLIN flag is set (indicates readable/pending messages)
            has_pending = 1 if (events & zmq.POLLIN) else 0  # type: ignore[attr-defined]
            return has_pending
        except Exception as e:
            logger.error(f"Error checking socket pending status: {e}")
            return 0

    async def handle_batched_push_request(self, socket) -> Optional[MsgBase]:
        while True:
            parts = await socket.recv_multipart()
            part_count = len(parts)
            with SocketMetricsContext(self, SocketType.PULL, part_count):
                for part in parts:
                    # Parse message based on format
                    if part.startswith(b"{"):
                        # JSON format - typically from external systems
                        # like Mooncake
                        msg_dict = json.loads(part)
                        msg = msgspec.convert(msg_dict, type=Msg)
                    else:
                        # MessagePack format - internal LMCache communication
                        msg = msgspec.msgpack.decode(part, type=Msg)
                    if isinstance(msg, WorkerMsg):
                        await self.handle_worker_message(msg)

                    # FIXME(Jiayi): The abstraction of control messages
                    # might not be necessary.
                    # elif isinstance(msg, ControlMsg):
                    #    await self.issue_control_message(msg)
                    elif isinstance(msg, OrchMsg):
                        await self.handle_orchestration_message(msg)
                    else:
                        logger.error(f"Unknown message type: {type(msg)}")

    async def handle_batched_req_request(self, socket) -> Optional[MsgBase]:
        """Handle requests on ROUTER socket.

        ROUTER socket receives multi-part messages:
        [identity, empty_frame, payload]
        and must reply with the same identity frame.
        """
        while True:
            frames = await socket.recv_multipart()
            with SocketMetricsContext(self, SocketType.REPLY):
                identity = None
                try:
                    # ROUTER socket: [identity, empty_frame, payload]
                    if len(frames) < 3:
                        logger.error(
                            "Invalid ROUTER message format, expected >= 3 frames, "
                            "got %d",
                            len(frames),
                        )
                        continue
                    identity = frames[0]
                    # frames[1] is empty delimiter
                    part = frames[2]

                    # Parse message based on format
                    if part.startswith(b"{"):
                        # JSON format - typically from external systems like Mooncake
                        msg_dict = json.loads(part)
                        msg = msgspec.convert(msg_dict, type=Msg)
                    else:
                        # MessagePack format - internal LMCache communication
                        msg = msgspec.msgpack.decode(part, type=Msg)

                    if isinstance(msg, WorkerReqMsg):
                        ret_msg = await self.handle_worker_req_message(msg)
                        # Reply with identity frame for ROUTER socket
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(ret_msg)]
                        )
                    else:
                        logger.error("Unknown message type: %s", type(msg))
                        err_msg = ErrorMsg(error=f"Unknown message type: {type(msg)}")
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(err_msg)]
                        )
                except (
                    json.JSONDecodeError,
                    msgspec.DecodeError,
                    msgspec.ValidationError,
                    zmq.ZMQError,
                ) as e:
                    logger.error("Error handling request message: %s", e, exc_info=True)
                    err_msg = ErrorMsg(error=str(e))
                    # Try to reply with error if we have identity
                    if identity is not None:
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(err_msg)]
                        )

    async def handle_heartbeat_request(self, socket) -> None:
        """Handle heartbeat requests on dedicated ROUTER socket.

        This runs on a separate socket to ensure heartbeats are processed
        without being blocked by other requests.

        ROUTER socket receives multi-part messages:
        [identity, empty_frame, payload]
        """
        logger.info("Heartbeat handler task started, waiting for heartbeat requests...")
        while True:
            frames = await socket.recv_multipart()
            logger.debug("Received heartbeat request with %d frames", len(frames))
            with SocketMetricsContext(self, SocketType.REPLY):
                identity = None
                try:
                    # ROUTER socket: [identity, empty_frame, payload]
                    if len(frames) < 3:
                        logger.error(
                            "Invalid heartbeat ROUTER message format, "
                            "expected >= 3 frames, got %d",
                            len(frames),
                        )
                        continue
                    identity = frames[0]
                    part = frames[2]

                    if part.startswith(b"{"):
                        msg_dict = json.loads(part)
                        msg = msgspec.convert(msg_dict, type=Msg)
                    else:
                        msg = msgspec.msgpack.decode(part, type=Msg)

                    if isinstance(msg, HeartbeatMsg):
                        ret_msg = await self.reg_controller.heartbeat(msg)
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(ret_msg)]
                        )
                    else:
                        logger.error(
                            "Unexpected message type on heartbeat socket: %s",
                            type(msg),
                        )
                        err_msg = ErrorMsg(
                            error=f"Expected HeartbeatMsg, got {type(msg)}"
                        )
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(err_msg)]
                        )
                except (
                    json.JSONDecodeError,
                    msgspec.DecodeError,
                    msgspec.ValidationError,
                    zmq.ZMQError,
                ) as e:
                    logger.error(
                        "Error handling heartbeat request: %s", e, exc_info=True
                    )
                    err_msg = ErrorMsg(error=str(e))
                    # Try to reply with error if we have identity
                    if identity is not None:
                        await socket.send_multipart(
                            [identity, b"", msgspec.msgpack.encode(err_msg)]
                        )

    async def health_check(self):
        while True:
            await asyncio.sleep(self.health_check_interval)
            worker_infos = self.reg_controller.registry.get_all_worker_infos_cached(1)
            for worker_info in worker_infos:
                if (
                    time.time() - worker_info.last_heartbeat_time
                    > self.lmcache_worker_timeout
                ):
                    logger.warning(
                        "Worker %s_%s last heartbeat time: %s, "
                        "current time: %s, more than %s seconds",
                        worker_info.instance_id,
                        worker_info.worker_id,
                        worker_info.last_heartbeat_time,
                        time.time(),
                        self.lmcache_worker_timeout,
                    )
                    # Perform a full deregister to clean up all associated resources.
                    deregister_msg = DeRegisterMsg(
                        instance_id=worker_info.instance_id,
                        worker_id=worker_info.worker_id,
                        ip=worker_info.ip,
                        port=worker_info.port,
                    )
                    await self.reg_controller.deregister(deregister_msg)

    async def start_all(self):
        tasks = []
        if self.controller_urls["reply"] is not None:
            logger.info(
                "Starting reply request handler on %s", self.controller_urls["reply"]
            )
            tasks.append(self.handle_batched_req_request(self.controller_reply_socket))
        if self.controller_heartbeat_socket is not None:
            logger.info(
                "Starting heartbeat request handler on %s",
                self.controller_urls.get("heartbeat"),
            )
            tasks.append(
                self.handle_heartbeat_request(self.controller_heartbeat_socket)
            )
        else:
            logger.warning("Heartbeat socket is None, heartbeat handler not started!")
        logger.info("Starting pull request handler on %s", self.controller_urls["pull"])
        tasks.append(self.handle_batched_push_request(self.controller_pull_socket))
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    def close(self):
        """Clean up all resources owned by the controller manager."""
        if hasattr(self, "controller_pull_socket"):
            self.controller_pull_socket.close()
        if hasattr(self, "controller_reply_socket"):
            self.controller_reply_socket.close()
        if hasattr(self, "zmq_context"):
            self.zmq_context.destroy()
