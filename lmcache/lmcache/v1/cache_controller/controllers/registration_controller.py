# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import time

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.commands import FullSyncCommand, HeartbeatCommand
from lmcache.v1.cache_controller.message import (
    DeRegisterMsg,
    HealthMsg,
    HealthRetMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    RegisterMsg,
    RegisterRetMsg,
)
from lmcache.v1.cache_controller.observability import PrometheusLogger
from lmcache.v1.cache_controller.utils import RegistryTree
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)


class RegistrationController:
    def __init__(self):
        # Central registry tree managing all instances and workers
        self.registry = RegistryTree()
        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.registered_workers_count.set_function(
                lambda: len(self.registry.get_all_worker_infos_cached())
            )

    def post_init(self, kv_controller, cluster_executor):
        """
        Post initialization of the Registration Controller.
        """
        self.kv_controller = kv_controller
        self.cluster_executor = cluster_executor

    def get_socket(
        self, instance_id: str, worker_id: int
    ) -> Optional[zmq.asyncio.Socket]:
        """
        Get the socket for a given instance and worker ID.
        """
        worker_node = self.registry.get_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered", (instance_id, worker_id)
            )
            return None
        return worker_node.socket

    def get_peer_init_url(self, instance_id: str, worker_id: int) -> Optional[str]:
        """
        Get the URL for a given instance and worker ID.
        """
        worker_node = self.registry.get_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered or P2P is not used",
                (instance_id, worker_id),
            )
            return None
        return worker_node.peer_init_url

    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.registry.get_worker_ids(instance_id)

    async def get_instance_id(self, msg: QueryInstMsg) -> QueryInstRetMsg:
        """
        Get the instance id given an ip address.
        """
        ip = msg.ip
        event_id = msg.event_id
        instance_node = self.registry.get_instance_by_ip(ip)
        if instance_node is None:
            logger.warning("Instance not registered for IP %s", ip)
            return QueryInstRetMsg(instance_id=None, event_id=event_id)
        return QueryInstRetMsg(instance_id=instance_node.instance_id, event_id=event_id)

    async def register(
        self, msg: RegisterMsg, extra_config: Optional[dict[str, str]] = None
    ) -> RegisterRetMsg:
        """
        Register a new instance-worker connection mapping.

        Args:
            msg: RegisterMsg from worker
            extra_config: Optional extra configuration to return to worker,
                          e.g., {"heartbeat_url": "tcp://...:8082"}

        Returns:
            RegisterRetMsg with extra_config for worker initialization
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        ip = msg.ip
        port = msg.port
        url = f"{ip}:{port}"

        # prevent duplicate registration
        existing_worker = self.registry.get_worker(instance_id, worker_id)
        if existing_worker is not None:
            logger.warning(
                "Instance-worker %s already registered, skip registration",
                (instance_id, worker_id),
            )
            self.registry.clear_worker_kv(instance_id, worker_id)
            return (
                RegisterRetMsg()
                if extra_config is None
                else RegisterRetMsg(extra_config=extra_config)
            )

        peer_init_url = msg.peer_init_url
        if peer_init_url is None:
            logger.info(
                "peer init url of %s is None, only register when p2p is used.",
                (instance_id, worker_id),
            )

        context = get_zmq_context()
        socket = get_zmq_socket(
            context,
            url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        # Register worker in the tree
        self.registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
            socket=socket,
            registration_time=time.time(),
        )

        logger.info(
            "Registered instance-worker %s with URL %s", (instance_id, worker_id), url
        )
        return (
            RegisterRetMsg()
            if extra_config is None
            else RegisterRetMsg(extra_config=extra_config)
        )

    async def deregister(self, msg: DeRegisterMsg) -> None:
        """
        Deregister an instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id

        worker_node = self.registry.deregister_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered", (instance_id, worker_id)
            )
            return

        # Close socket
        if worker_node.socket is not None:
            close_zmq_socket(worker_node.socket)

        logger.info("Deregistered instance-worker %s", (instance_id, worker_id))

    async def health(self, msg: HealthMsg) -> HealthRetMsg:
        """
        Check the health of the lmcache worker.
        """
        return await self.cluster_executor.execute(
            "health",
            msg,
        )

    async def heartbeat(self, msg: HeartbeatMsg) -> HeartbeatRetMsg:
        """
        Heartbeat from lmcache worker (REQ-REP mode).

        Returns HeartbeatRetMsg with optional commands for the worker to execute.
        Commands are executed sequentially by the worker.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        success = self.registry.update_heartbeat(instance_id, worker_id, time.time())

        commands: list[HeartbeatCommand] = []

        if not success:
            logger.warning(
                "%s has not been registered, re-register the worker.",
                (instance_id, worker_id),
            )
            # re-register the worker
            register_msg = RegisterMsg(
                instance_id=msg.instance_id,
                worker_id=msg.worker_id,
                ip=msg.ip,
                port=msg.port,
                peer_init_url=msg.peer_init_url,
            )
            await self.register(register_msg)
            # New worker needs full sync
            commands.append(FullSyncCommand(reason="worker_re_registered"))
        else:
            # Check if full sync is needed (e.g., controller restart)
            if self.kv_controller is not None and hasattr(
                self.kv_controller, "full_sync_tracker"
            ):
                should_sync, reason = (
                    self.kv_controller.full_sync_tracker.should_request_full_sync(
                        instance_id, worker_id
                    )
                )
                if should_sync:
                    commands.append(FullSyncCommand(reason=reason))

        return HeartbeatRetMsg(commands=commands)

    async def query_worker_info(self, msg: QueryWorkerInfoMsg) -> QueryWorkerInfoRetMsg:
        """
        Query worker info.
        """
        event_id = msg.event_id
        worker_infos = []

        # Handle special case: instance_id = "all"
        if msg.instance_id == "all":
            # Get all worker infos from the registry
            worker_infos = self.registry.get_all_worker_infos_cached()
            # If specific worker_ids are requested, filter the results
            if msg.worker_ids is not None and len(msg.worker_ids) > 0:
                worker_infos = [
                    worker_info
                    for worker_info in worker_infos
                    if worker_info.worker_id in msg.worker_ids
                ]
            return QueryWorkerInfoRetMsg(event_id=event_id, worker_infos=worker_infos)

        # Normal case: query specific instance
        instance_node = self.registry.get_instance(msg.instance_id)
        if instance_node is None:
            logger.warning("instance %s not registered.", msg.instance_id)
        else:
            worker_ids = msg.worker_ids
            if worker_ids is None or len(worker_ids) == 0:
                worker_ids = instance_node.get_worker_ids()
            for worker_id in worker_ids:
                worker_node = instance_node.get_worker(worker_id)
                if worker_node is not None:
                    worker_infos.append(worker_node.to_worker_info(msg.instance_id))
                else:
                    logger.warning(
                        "worker %s not registered.", (msg.instance_id, worker_id)
                    )

        return QueryWorkerInfoRetMsg(event_id=event_id, worker_infos=worker_infos)
