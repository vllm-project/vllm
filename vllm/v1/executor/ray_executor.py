# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cloudpickle

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_utils import (
    FutureWrapper,
    RayWorkerWrapper,
    initialize_ray_cluster,
    ray,
)
from vllm.v1.outputs import ModelRunnerOutput

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None] = Future()
COMPLETED_NONE_FUTURE.set_result(None)


@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """

    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


class RayDistributedExecutor(Executor):
    """Ray-based distributed executor"""

    # These env vars are worker-specific, therefore are NOT copied
    # from the driver to the workers
    WORKER_SPECIFIC_ENV_VARS = {
        "VLLM_HOST_IP",
        "VLLM_HOST_PORT",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    }

    # These non-vLLM env vars are copied from the driver to workers
    ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

    uses_ray: bool = True
    supports_pp: bool = True

    def _init_executor(self) -> None:
        self.forward_dag: ray.dag.CompiledDAG | None = None

        # For TPU or XPU, avoid compiling NVIDIA's NCCL
        if current_platform.is_tpu() or current_platform.is_xpu():
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or not self.vllm_config.ec_transfer_config.is_ec_producer
        )

        self.scheduler_output: SchedulerOutput | None = None

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def shutdown(self) -> None:
        if logger:
            # Somehow logger can be None here.
            logger.info(
                "Shutting down Ray distributed executor. If you see error log "
                "from logging.cc regarding SIGTERM received, please ignore "
                "because this is the expected termination process in Ray."
            )
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray

            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    def _configure_ray_workers_use_nsight(self, ray_remote_kwargs) -> dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update(
            {
                "nsight": {
                    "t": "cuda,cudnn,cublas",
                    "o": "'worker_process_%p'",
                    "cuda-graph-trace": "node",
                }
            }
        )

        return ray_remote_kwargs

    def _update_noset_device_env_vars(self, ray_remote_kwargs):
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars.update(
            {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
        )
        return ray_remote_kwargs

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # The remaining workers are the actual ray actors.
        self.workers: list[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )

        # The way ray actors are setup in vllm is that the visible devices are
        # not set by actors, they are left unset by ray. Internally we index
        # the right gpu with local_rank. This is similar to how mp mode works.
        self._update_noset_device_env_vars(ray_remote_kwargs)

        # Create the workers.
        bundle_indices: list[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, (
                "VLLM_RAY_BUNDLE_INDICES must have the same size"
                f" as the world size, but got {bundle_indices=} "
                f"and {self.parallel_config.world_size=}"
            )
            assert len(set(bundle_indices)) == len(bundle_indices), (
                "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                f" but got {bundle_indices=}"
            )
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[: self.parallel_config.world_size]

        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)

            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)

        ip_counts: dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(
            worker_metadata, key=sort_by_driver_then_worker_ip
        )
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank for item in sorted_worker_metadata
        }
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote())
            )  # type: ignore[attr-defined]

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node."
            )

        # Set environment variables for the driver and workers.
        # We set CUDA_VISIBLE_DEVICES to ALL GPUs on the node for each worker.
        # This is needed because:
        # 1. Ray's compiled DAG needs to find the allocated GPU in
        #    CUDA_VISIBLE_DEVICES.
        # 2. vLLM's communication layer (NCCL, CustomAllreduce) needs to see
        #    all GPUs for P2P checks and communication setup. Though if it was
        #    just this reason, we could have also just kept the visible devices
        #    unset.
        # Each worker will use local_rank to index into the visible devices.
        all_args_to_update_environment_variables = [
            {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            for (node_id, _) in worker_node_and_gpu_ids
        ]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars).union(
                self.ADDITIONAL_ENV_VARS
            ),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self.collective_rpc("init_worker", args=(all_kwargs,))

        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.collective_rpc("reinitialize_distributed", args=(reconfig_request,))
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # Model will not execute, call model runner immediately.
            return self._execute_dag(scheduler_output, None, non_block)

        # Model will execute, defer to sample_tokens() call.
        self.scheduler_output = scheduler_output
        return COMPLETED_NONE_FUTURE if non_block else None

    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on the Ray workers.

        The scheduler output to use should have been provided in
        a prior call to execute_model().

        Args:
            grammar_output: The structured outputs grammar bitmask, if applicable.
            non_block: If True, the method will return a Future.

        Returns:
            The model runner output.
        """
        scheduler_output = self.scheduler_output
        if scheduler_output is None:
            return COMPLETED_NONE_FUTURE if non_block else None

        self.scheduler_output = None

        return self._execute_dag(scheduler_output, grammar_output, non_block)

    def _execute_dag(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if not non_block:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs[0])

        # Get output from all workers when connector is present
        assert self.kv_output_aggregator is not None
        if not non_block:
            # Block and get results from all workers
            return self.kv_output_aggregator.aggregate(ray.get(refs))

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
        """Runs the given method on all workers."""
        sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
        del method

        if kwargs is None:
            kwargs = {}
        ray_worker_outputs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                sent_method, *args, **kwargs
            )
            for worker in self.workers
        ]

        # Get the results of the ray workers.
        if non_block:
            return FutureWrapper(ray_worker_outputs)

        return ray.get(ray_worker_outputs, timeout=timeout)

    def _check_ray_cgraph_installation(self):
        import importlib.metadata

        from packaging import version

        required_version = version.parse("2.43.0")
        current_version = version.parse(importlib.metadata.version("ray"))
        if current_version < required_version:
            raise ValueError(
                f"Ray version {required_version} is "
                f"required, but found {current_version}"
            )

        import importlib.util

        cgraph_spec = importlib.util.find_spec("ray.experimental.compiled_dag_ref")
        if cgraph_spec is None:
            raise ValueError(
                "Ray Compiled Graph is not installed. "
                "Run `pip install ray[cgraph]` to install it."
            )

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "nccl":
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'. "
                "Run `pip install ray[cgraph]` and check cupy installation."
            )

    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        # Note: we should set this env var before importing
        # ray.dag, otherwise it will not take effect.
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        from ray.dag import InputNode, MultiOutputNode

        logger.info(
            "RAY_CGRAPH_get_timeout is set to %s",
            os.environ["RAY_CGRAPH_get_timeout"],  # noqa: SIM112
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE,
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        if channel_type not in ("auto", "nccl", "shm"):
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'."
            )

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(tp_group)
                ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if (
                    pp_rank < last_pp_rank
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                ):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        for output in outputs
                    ]

            forward_dag = MultiOutputNode(outputs)

        if envs.VLLM_USE_RAY_WRAPPED_PP_COMM:
            from ray.experimental.channel.accelerator_context import (
                register_accelerator_context,
            )

            from vllm.distributed.device_communicators.ray_communicator import (
                RayPPCommunicator,
            )

            register_accelerator_context(
                torch_module_name="cuda", communicator_cls=RayPPCommunicator
            )
            logger.info(
                "Using RayPPCommunicator "
                "(which wraps vLLM _PP GroupCoordinator) "
                "for Ray Compiled Graph communication."
            )
        else:
            logger.info(
                "Using Ray's NCCL communicator for Ray Compiled Graph communication."
            )

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

    def __del__(self):
        self.shutdown()

    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return
