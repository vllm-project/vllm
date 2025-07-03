# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Callable, Optional, Union

import msgspec
import zmq

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_mp_context, get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor.abstract import Executor
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: Optional[str] = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: Optional[str] = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: Optional[str] = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """
    addresses: EngineZmqAddresses
    parallel_config: dict[str, Union[int, str]]


class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
        }

        if client_handshake_address:
            common_kwargs[
                "client_handshake_address"] = client_handshake_address

        self.processes: list[BaseProcess] = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index
            # Start EngineCore in background process.
            self.processes.append(
                context.Process(target=target_fn,
                                name=f"EngineCore_{global_index}",
                                kwargs=common_kwargs | {
                                    "dp_rank": global_index,
                                    "local_dp_rank": local_index,
                                }))

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        try:
            for proc in self.processes:
                proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.close()

    def close(self):
        """Shutdown all procs."""
        self._finalizer()

    def join_first(self):
        """Wait for any process to exit."""
        connection.wait(proc.sentinel for proc in self.processes)

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes if proc.exitcode is not None
        }


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: Optional[list["PlacementGroup"]] = None,
        local_dp_ranks: Optional[list[int]] = None,
    ):
        import copy

        import ray
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy)

        from vllm.v1.engine.core import DPEngineCoreActor

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_engine_count = \
            vllm_config.parallel_config.data_parallel_size_local
        world_size = vllm_config.parallel_config.world_size

        if ray.is_initialized():
            logger.info(
                "Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if "
                "placement_groups is provided")
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must "
                "have the same length")
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            placement_groups, local_dp_ranks = \
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            self.created_placement_groups = placement_groups
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size")

        refs = []
        for index in range(dp_size):
            local_index = local_dp_ranks[index]
            dp_vllm_config = copy.deepcopy(vllm_config)
            pg = placement_groups[index]
            dp_vllm_config.parallel_config.placement_group = pg
            local_client = index < local_engine_count
            actor = ray.remote(DPEngineCoreActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=world_size,
                )).remote(vllm_config=dp_vllm_config,
                          executor_class=executor_class,
                          log_stats=log_stats,
                          local_client=local_client,
                          addresses=addresses,
                          dp_rank=index,
                          local_dp_rank=local_index)
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        for actor in self.local_engine_actors + self.remote_engine_actors:
            self.run_refs.append(actor.run.remote())

    @staticmethod
    def create_dp_placement_groups(
            vllm_config: VllmConfig
    ) -> tuple[list["PlacementGroup"], list[int]]:

        import ray
        from ray._private.state import available_resources_per_node
        from ray.util.state import list_nodes

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = \
            vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_engine_count = \
            vllm_config.parallel_config.data_parallel_size_local

        nodes = sorted(list_nodes(),
                       key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, (
            "The first node must be the head node")
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node")

        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        for node in nodes:
            node_ip = node.node_ip
            node_resources = available_resources[node.node_id]
            # For now, each DP rank can only be assigned to one node
            # TODO(rui): support allocating a single DP rank
            # to multiple nodes
            available_engine_count = int(node_resources["GPU"]) // world_size
            if node_ip == dp_master_ip:
                assert available_engine_count >= local_engine_count, (
                    "Not enough resources to allocate DP ranks "
                    f"on DP master node {node_ip}")
                for i in range(local_engine_count):
                    bundles = [{
                        "GPU": 1.0,
                        "node:" + dp_master_ip: 0.001
                    }] * world_size + [{
                        "CPU": 1.0
                    }]
                    pg = ray.util.placement_group(
                        name=f"dp_rank_{len(placement_groups)}",
                        strategy="STRICT_PACK",
                        bundles=bundles,
                    )
                    placement_groups.append(pg)
                    local_dp_ranks.append(i)
            else:
                for i in range(available_engine_count):
                    if len(placement_groups) == dp_size:
                        break
                    bundles = [{"GPU": 1.0}] * world_size + [{"CPU": 1.0}]
                    pg = ray.util.placement_group(
                        name=f"dp_rank_{len(placement_groups)}",
                        strategy="STRICT_PACK",
                        bundles=bundles,
                    )
                    placement_groups.append(pg)
                    local_dp_ranks.append(i)
        return placement_groups, local_dp_ranks

    def get_run_refs(self):
        return self.run_refs

    def close(self):
        import ray
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    num_api_servers: int = 1,
) -> Iterator[tuple[
        Optional[Union[CoreEngineProcManager, CoreEngineActorManager]],
        Optional[DPCoordinator],
        EngineZmqAddresses,
]]:
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    external_dp_lb = parallel_config.data_parallel_external_lb

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/offline_inference/data_parallel.py.
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = offline_mode or external_dp_lb or (local_engine_count
                                                           == dp_size)

    # Set up input and output addresses.
    addresses = EngineZmqAddresses(
        inputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
        outputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
    )

    # Run the DP Coordinator process with rank 0 when in
    # online DP mode.
    run_coordinator = dp_size > 1 and not offline_mode and dp_rank == 0

    if run_coordinator:
        coordinator = DPCoordinator(parallel_config)

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses())
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address())

        logger.info("Started DP Coordinator process (PID: %d)",
                    coordinator.proc.pid)
    else:
        coordinator = None

    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses
        return

    if offline_mode or (external_dp_lb and dp_rank > 0):
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    else:
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port)

    if external_dp_lb and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with zmq_socket_ctx(local_handshake_address, zmq.ROUTER,
                        bind=True) as handshake_socket:

        from vllm.v1.engine.core import EngineCoreProc

        # Start local engines.
        if local_engine_count:
            # In server mode, start_index and local_start_index will
            # both be 0.
            local_engine_manager = CoreEngineProcManager(
                EngineCoreProc.run_engine_core,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0)
        else:
            local_engine_manager = None

        yield local_engine_manager, coordinator, addresses

        # Now wait for engines to start.
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    proc_manager: Optional[CoreEngineProcManager],
    coord_process: Optional[Process],
):
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] counts
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)
    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) "
                    "to connect.", *conn_pending)
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) "
                    "to start.", *start_pending)
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core processes exited.
            finished = proc_manager.finished_procs() if proc_manager else {}
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode
            raise RuntimeError("Engine core initialization failed. "
                               "See root cause above. "
                               f"Failed core proc(s): {finished}")

        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in core_engines if e.identity == eng_identity),
                      None)
        if engine is None:
            raise RuntimeError(f"Message from engine with unexpected data "
                               f"parallel rank: {eng_index}")
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        status, local = msg["status"], msg["local"]
        if local != engine.local:
            raise RuntimeError(f"{status} message from "
                               f"{'local' if local else 'remote'} "
                               f"engine {eng_index}, expected it to be "
                               f"{'local' if engine.local else 'remote'}")

        if status == "HELLO" and engine.state == CoreEngineState.NEW:

            # Send init message with DP config info.
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        "data_parallel_master_ip":
                        parallel_config.data_parallel_master_ip,
                        "data_parallel_master_port":
                        parallel_config.data_parallel_master_port,
                        "data_parallel_size":
                        parallel_config.data_parallel_size,
                    }))
            handshake_socket.send_multipart((eng_identity, init_message),
                                            copy=False)
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # Setup KV cache config with initialization state from
            # engine core process. Sum values from all engines in DP case.
            num_gpu_blocks = cache_config.num_gpu_blocks or 0
            num_gpu_blocks += msg["num_gpu_blocks"]
            cache_config.num_gpu_blocks = num_gpu_blocks

            # In external DP LB mode, the coordinator address that the
            # front-end procs connect to is obtained from rank 0 via
            # one of the engine handshakes, and passed to the local
            # front-end process in the response from the other.
            if addresses.frontend_stats_publish_address is None:
                addresses.frontend_stats_publish_address = msg.get(
                    "dp_stats_address")

            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            raise RuntimeError(f"Unexpected {status} message for "
                               f"{'local' if local else 'remote'} engine "
                               f"{eng_index} in {engine.state} state.")

        logger.debug("%s from %s core engine process %s.", status,
                     "local" if local else "remote", eng_index)
