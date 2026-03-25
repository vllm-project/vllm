# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
import threading
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_open_port,
)
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    build_actor_name,
    get_bundles_for_indices,
    get_bundles_sorted_by_node,
    initialize_ray_cluster,
    ray,
)

if ray is not None:
    from ray.actor import ActorHandle
    from ray.types import ObjectRef
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

logger = init_logger(__name__)


@dataclass
class RayWorkerHandle:
    """Handle for a Ray worker actor, compatible with MultiprocExecutor."""

    actor: ActorHandle
    """Ray worker actor"""

    rank: int
    """Rank of the worker"""

    local_rank: int
    """Local rank of the worker"""

    node_id: str
    """Node ID of the worker"""

    bundle_id_idx: int = -1
    """Placement group bundle index for the worker"""

    run_ref: ObjectRef = None
    """run() ObjectRef used as a sentinel for health monitoring"""


class RayWorkerProc(WorkerProc):
    """Worker process that runs inside a Ray actor.

    Initialization is split into two phases:
    1. __init__: lightweight setup, stores init args (no device/model init)
    2. initialize_worker: called after GPU IDs are discovered, completes
       the full WorkerProc initialization with the correct local_rank and
       CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        is_driver_worker: bool,
        is_driver_node: bool = False,
    ):
        # Defer WorkerProc.__init__ until GPU IDs are known.
        self._is_driver_node = is_driver_node
        self._init_kwargs = dict(
            vllm_config=vllm_config,
            rank=rank,
            distributed_init_method=distributed_init_method,
            input_shm_handle=input_shm_handle,
            shared_worker_lock=None,
            is_driver_worker=is_driver_worker,
        )

    def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
        """Return (node_id, gpu_ids) assigned to this actor by Ray."""
        node_id = ray.get_runtime_context().get_node_id()
        device_key = current_platform.ray_device_key
        if not device_key:
            raise RuntimeError(
                "current platform %s does not support ray.",
                current_platform.device_name,
            )
        gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
        return node_id, [int(x) for x in gpu_ids]

    def initialize_worker(self, local_rank: int, env_vars: dict[str, str]) -> None:
        """Complete initialization after GPU assignment is known.

        Sets CUDA_VISIBLE_DEVICES and initializes the underlying WorkerProc
        with the correct local_rank.
        """
        for key, value in env_vars.items():
            os.environ[key] = value

        self.local_rank = local_rank
        super().__init__(
            local_rank=local_rank,
            **self._init_kwargs,
        )

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        """
        Workers on the same node as the executor use shared memory for
        both the broadcast (input) MQ and the response MQ. Workers on
        different nodes use TCP (n_local_reader=0).
        """
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )

        n_local = 1 if self._is_driver_node else 0
        # Use ray.util.get_node_ip_address() to get Ray's internal IP.
        # get_ip() returns host's external IP which is typically not
        # routable between nodes within the cluster.
        self.worker_response_mq = MessageQueue(
            n_reader=1,
            n_local_reader=n_local,
            connect_ip=ray.util.get_node_ip_address(),
        )
        self.peer_response_handles: list[dict] = []

    def wait_for_init(self) -> dict:
        """Respond to the driver's wait_until_ready() barrier."""
        assert self.worker_response_mq is not None
        return {
            "status": self.READY_STR,
            "handle": self.worker_response_mq.export_handle(),
        }

    def run(self) -> None:
        """Main entry point called via actor.run.remote()."""
        try:
            assert self.rpc_broadcast_mq is not None
            self.rpc_broadcast_mq.wait_until_ready()
            assert self.worker_response_mq is not None
            self.worker_response_mq.wait_until_ready()

            self.worker_busy_loop()
        finally:
            self.shutdown()


class RayExecutorV2(MultiprocExecutor):
    """Ray-based distributed executor using MessageQueue communication.

    Inherits from MultiprocExecutor to reuse the MQ-based control plane
    and NCCL data plane. Workers are Ray actors.
    """

    uses_ray: bool = True
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig):
        super(MultiprocExecutor, self).__init__(vllm_config)

    def _init_executor(self) -> None:
        """Initialize the RayExecutorV2 executor."""
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None
        self.shutting_down = False

        # Step 1: Initialize Ray cluster and retrieve placement group
        if ray is None:
            raise ImportError("Ray is required for RayExecutorV2")
        initialize_ray_cluster(self.parallel_config, require_gpu_on_driver=False)
        placement_group = self.parallel_config.placement_group

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # Step 2: Build bundle assignments for worker rank placement
        # while respecting VLLM_RAY_BUNDLE_INDICES.
        if envs.VLLM_RAY_BUNDLE_INDICES:
            bundle_to_node_id = get_bundles_for_indices(
                placement_group,
                list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(","))),
                self.world_size,
            )
        else:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)
        driver_node = ray.get_runtime_context().get_node_id()

        bundle_assignments: list[dict[str, Any]] = []
        for rank, (bundle_id_idx, node_id, node_ip) in enumerate(bundle_to_node_id):
            bundle_assignments.append(
                {
                    "rank": rank,
                    "bundle_id_idx": bundle_id_idx,
                    "node_id": node_id,
                    "node_ip": node_ip,
                }
            )

        # Step 3: Resolve the IP for torch.distributed TCPStore.
        # The TCPStore server runs on rank 0's node, so all workers
        # must be able to reach this address.
        dist_ip = bundle_assignments[0]["node_ip"]
        distributed_init_method = get_distributed_init_method(dist_ip, get_open_port())

        # Step 4: Create broadcast MessageQueue.
        # Workers on the driver node use shared memory; the rest use TCP.
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        n_local = sum(1 for a in bundle_assignments if a["node_id"] == driver_node)
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            n_local,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=ray.util.get_node_ip_address(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Step 5: Spawn RayWorkerProc actors into PG bundles (deferred init).
        # Workers are created lightweight here; full initialization happens
        # in Step 7 after GPU IDs are discovered.
        self.ray_worker_handles: list[RayWorkerHandle] = []
        instance_id = self.vllm_config.instance_id

        # Collect env vars to propagate from driver to workers (NCCL,
        # HF, vLLM flags, etc.) — same mechanism as RayDistributedExecutor.
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="RayWorkerProc actors",
        )
        self._worker_env_vars = {
            name: os.environ[name] for name in env_vars_to_copy if name in os.environ
        }

        for bundle_idx in range(self.world_size):
            bundle = bundle_assignments[bundle_idx]
            is_driver_worker = self._is_driver_worker(bundle["rank"])
            is_driver_node = bundle["node_id"] == driver_node

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle["bundle_id_idx"],
            )

            # Build runtime_env for the worker actor:
            # 1. Start from parallel_config.ray_runtime_env (pip, working_dir,
            #    etc.) so that packages installed at the job level are
            #    available inside RayWorkerProc actors.
            # 2. Merge in driver env vars (NCCL, HF, vLLM flags) collected
            #    by get_env_vars_to_copy.
            # 3. Prevent Ray from setting CUDA_VISIBLE_DEVICES; we set it
            #    ourselves in initialize_worker after discovering GPU IDs.
            base_runtime_env = self.parallel_config.ray_runtime_env
            runtime_env: dict = (
                copy.deepcopy(dict(base_runtime_env)) if base_runtime_env else {}
            )
            env_vars = runtime_env.setdefault("env_vars", {})
            env_vars.update(self._worker_env_vars)
            env_vars.update(
                {v: "1" for v in current_platform.ray_noset_device_env_vars}
            )

            actor_name = build_actor_name(
                instance_id, bundle["rank"], tp_size, pp_size, pcp_size
            )

            actor = (
                ray.remote(RayWorkerProc)
                .options(
                    name=actor_name,
                    num_cpus=0,
                    num_gpus=envs.VLLM_RAY_PER_WORKER_GPUS,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=self.vllm_config,
                    rank=bundle["rank"],
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                    is_driver_node=is_driver_node,
                )
            )

            handle = RayWorkerHandle(
                actor=actor,
                rank=bundle["rank"],
                local_rank=-1,  # Set in Step 7 after GPU ID discovery
                node_id=bundle["node_id"],
                bundle_id_idx=bundle["bundle_id_idx"],
            )
            self.ray_worker_handles.append(handle)

        # Step 6: Discover GPU IDs assigned to each worker via Ray runtime context.
        worker_node_and_gpu_ids = ray.get(
            [h.actor.get_node_and_gpu_ids.remote() for h in self.ray_worker_handles]
        )

        node_workers: dict[str, list[int]] = defaultdict(list)
        node_gpus: dict[str, list[int]] = defaultdict(list)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # Step 7: Initialize workers with correct local_rank and
        # CUDA_VISIBLE_DEVICES. Each worker sees all GPUs assigned to
        # this executor on its node; local_rank indexes into that set.
        init_worker_refs = []
        for i, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(i)
            worker_env_vars = {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            self.ray_worker_handles[i].local_rank = local_rank
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    local_rank, worker_env_vars
                )
            )
        ray.get(init_worker_refs)

        # Step 8: Collect response MQ handles
        init_refs = [h.actor.wait_for_init.remote() for h in self.ray_worker_handles]
        init_results = ray.get(init_refs)

        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != RayWorkerProc.READY_STR:
                raise RuntimeError(f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0)
            )

        # Step 9: Start run() before wait_until_ready() to avoid
        # deadlock — workers send subscriptions inside run().
        for handle in self.ray_worker_handles:
            handle.run_ref = handle.actor.run.remote()

        # Step 10: wait_until_ready() barrier
        self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        self.futures_queue = deque[tuple[FutureWrapper, Any]]()
        self._post_init_executor()

        self.start_worker_monitor()
        self.output_rank = self._get_output_rank()

    def start_worker_monitor(self, inline=False) -> None:
        """Monitor worker liveness via ray.wait() on run() ObjectRefs."""
        run_refs = [h.run_ref for h in self.ray_worker_handles if h.run_ref is not None]
        if not run_refs:
            raise RuntimeError("Ray workers have not started successfully.")

        self_ref = weakref.ref(self)
        ref_to_rank = {
            h.run_ref: h.rank for h in self.ray_worker_handles if h.run_ref is not None
        }

        def _should_stop() -> bool:
            executor = self_ref()
            return not executor or executor.shutting_down

        def monitor_workers():
            # Poll with a timeout rather than blocking on ray.wait()
            # because a blocking call would segfault if Ray is torn down
            # while this thread is inside it.
            while not _should_stop() and ray.is_initialized():
                try:
                    done, _ = ray.wait(run_refs, num_returns=1, timeout=5.0)
                except Exception:
                    return
                if not done or _should_stop():
                    continue

                dead_ranks = [ref_to_rank[r] for r in done if r in ref_to_rank]
                executor = self_ref()
                if not executor:
                    return
                executor.is_failed = True
                logger.error(
                    "RayWorkerProc rank=%s died unexpectedly, shutting down executor.",
                    dead_ranks,
                )
                executor.shutdown()
                if executor.failure_callback is not None:
                    callback = executor.failure_callback
                    executor.failure_callback = None
                    callback()
                return

        t = threading.Thread(
            target=monitor_workers, daemon=True, name="RayWorkerMonitor"
        )
        t.start()
        self._monitor_thread = t

    def _join_monitor_thread(self) -> None:
        """Wait for the monitor thread to exit.

        Must be called before tearing down Ray resources — the monitor
        may be inside ray.wait() which would segfault if Ray is shut
        down underneath it. When the monitor itself calls shutdown()
        on worker death, we skip the join because the thread is about
        to return anyway.
        """
        monitor = getattr(self, "_monitor_thread", None)
        if (
            monitor is not None
            and monitor.is_alive()
            and threading.current_thread() is not monitor
        ):
            monitor.join(timeout=10)

    def shutdown(self) -> None:
        """Properly shut down the executor and its workers"""
        if getattr(self, "shutting_down", False):
            self._join_monitor_thread()
            return
        self.shutting_down = True

        self._join_monitor_thread()

        for handle in getattr(self, "ray_worker_handles", []):
            try:
                ray.kill(handle.actor)
            except Exception:
                logger.exception("Failed to kill actor rank=%d", handle.rank)

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None

        for mq in getattr(self, "response_mqs", []):
            mq.shutdown()
        self.response_mqs = []
