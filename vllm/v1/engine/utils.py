# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Callable, Optional, Union
from unittest.mock import patch

import msgspec
import zmq

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
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
    parallel_config: dict[str, Union[int, str, list[int]]]


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
        local_dp_ranks = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(target=target_fn,
                                name=f"EngineCore_DP{global_index}",
                                kwargs=common_kwargs | {
                                    "dp_rank": global_index,
                                    "local_dp_rank": local_index,
                                }))

        self._finalizer = weakref.finalize(self, shutdown, self.processes)

        data_parallel = vllm_config.parallel_config.data_parallel_size > 1
        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                with set_device_control_env_var(
                        vllm_config, local_dp_rank) if (
                            data_parallel) else contextlib.nullcontext():
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


@contextlib.contextmanager
def set_device_control_env_var(vllm_config: VllmConfig,
                               local_dp_rank: int) -> Iterator[None]:
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size)
    with patch.dict(os.environ, values=((evar, value), )):
        yield


def get_device_indices(device_control_env_var: str, local_dp_rank: int,
                       world_size: int):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    try:
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(local_dp_rank * world_size, (local_dp_rank + 1) *
                           world_size))
    except IndexError as e:
        raise Exception(f"Error setting {device_control_env_var}: "
                        f"local range: [{local_dp_rank * world_size}, "
                        f"{(local_dp_rank + 1) * world_size}) "
                        "base value: "
                        f"\"{os.getenv(device_control_env_var)}\"") from e
    return value


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
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy)

        from vllm.v1.engine.core import DPEngineCoreActor

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        env_vars_list = get_env_vars_to_copy(destination="DPEngineCoreActor")
        self.env_vars_dict = {
            name: os.environ[name]
            for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
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

        self.placement_group_is_local = []
        refs = []
        for index, local_index, pg in zip(range(dp_size), local_dp_ranks,
                                          placement_groups):
            dp_vllm_config = copy.deepcopy(vllm_config)
            dp_vllm_config.parallel_config.placement_group = pg
            local_client = index < local_engine_count

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                device_indices = get_device_indices(device_evar, local_index,
                                                    world_size)
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = device_indices
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            actor = ray.remote(DPEngineCoreActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=world_size,
                ),
                runtime_env=runtime_env).remote(vllm_config=dp_vllm_config,
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
            self.placement_group_is_local.append(local_client)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        for actor in self.local_engine_actors + self.remote_engine_actors:
            self.run_refs.append(actor.run.remote())

    @staticmethod
    def create_dp_placement_groups(
            vllm_config: VllmConfig
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = \
            vllm_config.parallel_config.data_parallel_master_ip
        num_pg_to_create = vllm_config.parallel_config.data_parallel_size
        local_engine_count = \
            vllm_config.parallel_config.data_parallel_size_local

        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []
        dp_master_ip_key = f'node:{dp_master_ip}'
        nodes = sorted(available_resources.values(),
                       key=lambda x: dp_master_ip_key not in x)
        assert len(nodes) > 0, (
            "No nodes with resources found in Ray cluster.")
        assert dp_master_ip_key in nodes[0], (
            "The DP master node (ip: %s) is missing or dead", dp_master_ip)
        device_str = current_platform.ray_device_key
        for node_resources in nodes:
            if device_str not in node_resources:
                continue
            # For now, each DP rank can only be assigned to one node
            # TODO(rui): support allocating a single DP rank
            # to multiple nodes
            available_engine_count = int(
                node_resources[device_str]) // world_size
            if dp_master_ip_key in node_resources:
                assert available_engine_count >= local_engine_count, (
                    "Not enough resources to allocate DP ranks "
                    f"on DP master node {dp_master_ip}")
                for i in range(local_engine_count):
                    bundles = [{
                        device_str: 1.0,
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
                    if len(placement_groups) == num_pg_to_create:
                        break
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]
                    pg = ray.util.placement_group(
                        name=f"dp_rank_{len(placement_groups)}",
                        strategy="STRICT_PACK",
                        bundles=bundles,
                    )
                    placement_groups.append(pg)
                    local_dp_ranks.append(i)
        if len(placement_groups) < num_pg_to_create:
            raise ValueError(
                f"Not enough resources to allocate {num_pg_to_create} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}")
        return placement_groups, local_dp_ranks

    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (available_resources_per_node,
                                        total_resources_per_node)
        from ray.util.state import list_nodes

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        world_size = old_vllm_config.parallel_config.world_size

        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, (
            "The first node must be the head node")
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node")

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            available_gpus = int(available_resources[node_id][device_str])

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # Create bundles with node constraint for master node
                if node_ip == dp_master_ip:
                    bundles = [{
                        device_str: 1.0,
                        "node:" + dp_master_ip: 0.001
                    }] * world_size + [{
                        "CPU": 1.0
                    }]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # Local rank starts from the number of engines already used
                # on this node
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    def scale_up_elastic_ep(self, cur_vllm_config: VllmConfig,
                            new_data_parallel_size: int) -> None:
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy)

        from vllm.v1.engine.core import DPEngineCoreActor

        cur_data_parallel_size = len(self.local_engine_actors) + \
            len(self.remote_engine_actors)

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up")

        placement_groups, local_dp_ranks = \
            self.add_dp_placement_groups(
                cur_vllm_config, new_data_parallel_size)

        world_size = cur_vllm_config.parallel_config.world_size
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict
                                 | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"})
        for i, (pg,
                local_rank) in enumerate(zip(placement_groups,
                                             local_dp_ranks)):
            rank = cur_data_parallel_size + i
            dp_vllm_config = copy.deepcopy(cur_vllm_config)
            dp_vllm_config.parallel_config.data_parallel_size = \
                new_data_parallel_size
            dp_vllm_config.parallel_config.placement_group = pg

            # Check if this placement group is on the head node
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0
                for bundle in pg.bundle_specs)

            if local_client:
                new_local_engines += 1
                # Update data_parallel_size_local
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local +
                    new_local_engines)

            actor = ray.remote(DPEngineCoreActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=world_size,
                ),
                runtime_env=runtime_env).remote(
                    vllm_config=dp_vllm_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank)

            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        ray.get([
            actor.wait_for_init.remote()
            for actor in (self.local_engine_actors[-new_local_engines:]
                          if new_local_engines > 0 else []) +
            self.remote_engine_actors[-(len(placement_groups) -
                                        new_local_engines):]
        ])

        actors = (self.local_engine_actors[-new_local_engines:]
                  if new_local_engines > 0 else []) + \
            self.remote_engine_actors[-(len(placement_groups) -
                                        new_local_engines):]

        for actor in actors:
            self.run_refs.append(actor.run.remote())

        cur_vllm_config.parallel_config.data_parallel_size = \
            new_data_parallel_size
        # Update old_vllm_config with new data_parallel_size_local if any new
        # local engines were added
        if new_local_engines > 0:
            cur_vllm_config.parallel_config.data_parallel_size_local += \
                new_local_engines

    def scale_down_elastic_ep(self, cur_data_parallel_size: int,
                              new_data_parallel_size: int) -> None:
        import ray
        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down")
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

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
    local_engines_only = (parallel_config.data_parallel_hybrid_lb
                          or parallel_config.data_parallel_external_lb)

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/offline_inference/data_parallel.py.
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (offline_mode or local_engines_only
                         or (local_engine_count == dp_size))

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

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible.")
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port)

    if local_engines_only and dp_rank > 0:
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

    remote_should_be_headless = not parallel_config.data_parallel_hybrid_lb \
        and not parallel_config.data_parallel_external_lb

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
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        if local != engine.local:
            raise RuntimeError(f"{status} message from "
                               f"{'local' if local else 'remote'} "
                               f"engine {eng_index}, expected it to be "
                               f"{'local' if engine.local else 'remote'}")

        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(f"Remote engine {eng_index} must not use "
                                   f"--headless in external or hybrid dp lb "
                                   f"mode")
            else:
                raise RuntimeError(f"Remote engine {eng_index} must use "
                                   f"--headless unless in external or hybrid "
                                   f"dp lb mode")

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
                        "_data_parallel_master_port_list":
                        parallel_config._data_parallel_master_port_list,
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
