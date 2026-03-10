# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)
from vllm.v1.executor.ray_utils import (
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

    bundle_index: int
    """Placement group bundle index to schedule the actor on"""

    run_ref: ObjectRef = None
    """run() ObjectRef used as a sentinel for health monitoring"""


class RayWorkerProc(WorkerProc):
    """Worker process that runs inside a Ray actor."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        is_driver_worker: bool,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            input_shm_handle=input_shm_handle,
            shared_worker_lock=None,
            is_driver_worker=is_driver_worker,
        )
        self.local_rank = local_rank

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
        # Skip MultiprocExecutor.__init__; we monitor via ray.wait()
        self.monitor_workers = False
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
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # Step 2: Query PG table, sort bundles, assign ranks
        pg_table = ray.util.placement_group_table(placement_group)
        bundle_to_node = pg_table["bundles_to_node_id"]

        # Prefer driver node; group by node for TP locality
        bundle_to_node_id = []
        for i, bundle in enumerate(placement_group.bundle_specs):
            ray_device_key = current_platform.ray_device_key
            if not ray_device_key:
                raise ValueError(
                    f"current platform {current_platform.device_name}"
                    " does not support ray."
                )

            if bundle.get(ray_device_key):
                node_id = bundle_to_node.get(i) or bundle_to_node.get(str(i))
                bundle_to_node_id.append((i, node_id))

        bundle_to_node_id = bundle_to_node_id[: self.world_size]
        driver_node = ray.get_runtime_context().get_node_id()

        def _sort_key(item):
            _, node_id = item
            return (0 if node_id == driver_node else 1, node_id)

        bundle_to_node_id.sort(key=_sort_key)

        # Assign each worker a local rank
        node_rank_counter: dict[str, int] = defaultdict(int)
        bundle_assignments: list[dict[str, Any]] = []
        for rank, (bundle_id, node_id) in enumerate(bundle_to_node_id):
            local_rank = node_rank_counter[node_id]
            node_rank_counter[node_id] += 1
            bundle_assignments.append(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "bundle_id": bundle_id,
                    "node_id": node_id,
                }
            )

        # Determine node topology
        node_ids = list(dict.fromkeys(a["node_id"] for a in bundle_assignments))
        is_single_node = len(node_ids) == 1

        # Step 3: Create broadcast MessageQueue
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip() if is_single_node else get_ip(), get_open_port()
        )

        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        mq_connect_ip = get_ip()
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            self.local_world_size,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=mq_connect_ip,
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Step 4: Spawn RayWorkerProc actors into PG bundles
        self.ray_worker_handles: list[RayWorkerHandle] = []
        self._ray_actors: list[Any] = []

        # Create the remote actor
        for assignment in bundle_assignments:
            is_driver_worker = self._is_driver_worker(assignment["rank"])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=assignment["bundle_id"],
            )

            # Prevent Ray from setting CUDA_VISIBLE_DEVICES
            runtime_env = {
                "env_vars": {
                    env_var: "1"
                    for env_var in current_platform.ray_noset_device_env_vars
                },
            }

            actor = (
                ray.remote(RayWorkerProc)
                .options(
                    num_cpus=0,
                    num_gpus=envs.VLLM_RAY_PER_WORKER_GPUS,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=self.vllm_config,
                    local_rank=assignment["local_rank"],
                    rank=assignment["rank"],
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                )
            )

            handle = RayWorkerHandle(
                actor=actor,
                rank=assignment["rank"],
                local_rank=assignment["local_rank"],
                node_id=assignment["node_id"],
                bundle_index=assignment["bundle_id"],
            )
            self.ray_worker_handles.append(handle)
            self._ray_actors.append(actor)

        # Step 5: Collect response MQ handles
        init_refs = [h.actor.wait_for_init.remote() for h in self.ray_worker_handles]
        init_results = ray.get(init_refs)

        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != RayWorkerProc.READY_STR:
                raise RuntimeError(f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0)
            )

        # Step 6: Start run() before wait_until_ready() to avoid
        # deadlock — workers send subscriptions inside run().
        for handle in self.ray_worker_handles:
            handle.run_ref = handle.actor.run.remote()

        # Step 7: wait_until_ready() barrier
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
            return

        self_ref = weakref.ref(self)

        ref_to_rank = {
            h.run_ref: h.rank for h in self.ray_worker_handles if h.run_ref is not None
        }

        def monitor_workers():
            done, _ = ray.wait(run_refs, num_returns=1)
            executor = self_ref()
            if not executor or executor.shutting_down:
                return

            dead_ranks = [ref_to_rank[ref] for ref in done if ref in ref_to_rank]
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

        threading.Thread(
            target=monitor_workers, daemon=True, name="RayWorkerMonitor"
        ).start()

    def shutdown(self) -> None:
        """Properly shut down the executor and its workers"""
        if getattr(self, "shutting_down", False):
            return
        self.shutting_down = True

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
