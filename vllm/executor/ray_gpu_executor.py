import asyncio
import os
from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (_run_task_with_lock,
                        error_on_invalid_device_count_status,
                        get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


class RayGPUExecutor(DistributedGPUExecutor):

    uses_ray: bool = True

    def _init_executor(self) -> None:
        # If the env var is set, it uses the Ray's compiled DAG API
        # which optimizes the control plane overhead.
        # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
        # Currently, this requires USE_RAY_SPMD_WORKER=True.
        self.use_ray_compiled_dag = envs.VLLM_USE_RAY_COMPILED_DAG
        # If the env var is set, then we do not distinguish between the
        # "driver worker" vs other workers. Also, the rank 0 worker will
        # be executed in a remote Ray worker. Currently this requires
        # USE_RAY_COMPILED_DAG=True.
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if self.use_ray_compiled_dag:
            assert self.use_ray_spmd_worker, (
                "VLLM_USE_RAY_COMPILED_DAG=1 requires "
                "VLLM_USE_RAY_SPMD_WORKER=1")
        if self.use_ray_spmd_worker:
            # TODO: Support SPMD worker for non-DAG Ray executor.
            assert self.use_ray_compiled_dag, (
                "VLLM_USE_RAY_SPMD_WORKER=1 requires "
                "VLLM_USE_RAY_COMPILED_DAG=1")

        assert self.uses_ray
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.forward_dag: Optional["ray.dag.CompiledDAG"] = None

    def _configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas",
                "o": "'worker_process_%p'",
                "cuda-graph-trace": "node",
            }
        })

        return ray_remote_kwargs

    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        if self.speculative_config is not None:
            worker_module_name = "vllm.spec_decode.spec_decode_worker"
            worker_class_name = "create_spec_worker"
        else:
            worker_module_name = "vllm.worker.worker"
            worker_class_name = "Worker"

        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if (self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1):
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        worker_wrapper_kwargs = self._get_worker_wrapper_args()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(**worker_wrapper_kwargs)

            if self.use_ray_spmd_worker:
                self.workers.append(worker)
            else:
                worker_ip = ray.get(worker.get_node_ip.remote())
                if worker_ip == driver_ip and self.driver_dummy_worker is None:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(
                        **worker_wrapper_kwargs)
                else:
                    # Else, added to the list of workers.
                    self.workers.append(worker)

        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        # the order in `worker_node_and_gpu_ids` does not necessarily match
        # the machine boundaries. We need to make sure that workers in the
        # same node are assigned consecutive ranks.
        # examples:
        # [('852a09a13c7503ef126d7c828454c741494b1be33a8627a5206604d9', [0]), ('dfaad7adfdae57a694cc74490db45bd112c9f31243523e43ddc2e7f0', [0]), ('dfaad7adfdae57a694cc74490db45bd112c9f31243523e43ddc2e7f0', [1]), ('dfaad7adfdae57a694cc74490db45bd112c9f31243523e43ddc2e7f0', [2]), ('dfaad7adfdae57a694cc74490db45bd112c9f31243523e43ddc2e7f0', [3]), ('852a09a13c7503ef126d7c828454c741494b1be33a8627a5206604d9', [1]), ('852a09a13c7503ef126d7c828454c741494b1be33a8627a5206604d9', [2]), ('852a09a13c7503ef126d7c828454c741494b1be33a8627a5206604d9', [3])] # noqa

        # initialize worker ranks with -1 (unassigned)
        worker_ranks = [-1 for x in worker_node_and_gpu_ids]
        current_rank = 0
        while -1 in worker_ranks:
            # whenever we find an unassigned worker, find the node
            index = worker_ranks.index(-1)
            current_node_id = worker_node_and_gpu_ids[index][0]
            # assign ranks to all workers in the same node
            for i, (node_id, _) in enumerate(worker_node_and_gpu_ids):
                if node_id == current_node_id:
                    worker_ranks[i] = current_rank
                    current_rank += 1
        # with the above example, worker_ranks will be [0, 4, 5, 6, 7, 1, 2, 3]

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for worker_rank, (node_id, gpu_ids) in zip(worker_ranks,
                                                   worker_node_and_gpu_ids):
            node_workers[node_id].append(worker_rank)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
        }, ) for (node_id, _) in worker_node_and_gpu_ids]
        self._run_workers("update_environment_variables",
                          all_args=all_args_to_update_environment_variables)

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
            driver_ip, get_open_port())

        error_on_invalid_device_count_status()

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            ) for rank, (node_id,
                         _) in zip(worker_ranks, worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for rank, worker in sorted(zip(worker_ranks[1:], self.workers)):
            # We need to skip the driver worker, which we
            # do by skipping worker_ranks[0] which is always 0.
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        assert not self.use_ray_spmd_worker, (
            "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1")
        return self.driver_worker.execute_method("execute_model",
                                                 execute_model_req)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not self.use_ray_spmd_worker:
            return super().execute_model(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        outputs = ray.get(self.forward_dag.execute(execute_model_req))
        return outputs[0]

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        if self.use_ray_spmd_worker:
            assert not async_run_tensor_parallel_workers_only, (
                "async_run_tensor_parallel_workers_only is not supported for "
                "spmd mode.")

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        count = len(self.workers) if not \
            async_run_tensor_parallel_workers_only \
            else len(self.non_driver_workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0 if self.use_ray_spmd_worker else 1
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_workers = self.workers
        if async_run_tensor_parallel_workers_only:
            ray_workers = self.non_driver_workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_worker_output = []
        # In SPMD mode, the driver worker is the same as any other worker,
        # so we only explicitly execute on the driver worker if using a
        # non-SPMD worker class.
        if not self.use_ray_spmd_worker:
            driver_args = args if all_args is None else all_args[0]
            driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

            # Start the driver worker after all the ray workers.
            if not use_dummy_driver:
                driver_worker_output = [
                    self.driver_worker.execute_method(method, *driver_args,
                                                      **driver_kwargs)
                ]
            else:
                assert self.driver_dummy_worker is not None
                driver_worker_output = [
                    ray.get(
                        self.driver_dummy_worker.execute_method.remote(
                            method, *driver_args, **driver_kwargs))
                ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return driver_worker_output + ray_worker_outputs

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        ray.get(parallel_worker_tasks)

    def _compiled_ray_dag(self, enable_asyncio: bool):
        import pkg_resources
        from packaging import version

        required_version = version.parse("2.32")
        current_version = version.parse(
            pkg_resources.get_distribution("ray").version)
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} or greater is "
                             f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        assert self.parallel_config.use_ray

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode([
                worker.execute_model_spmd.bind(  # type: ignore[attr-defined]
                    input_data) for worker in self.workers
            ])
        return forward_dag.experimental_compile(enable_asyncio=enable_asyncio)

    def __del__(self):
        if self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)


class RayGPUExecutorAsync(RayGPUExecutor, DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp_locks: Optional[List[asyncio.Lock]] = None
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = make_async(
                self.driver_worker.execute_method)

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not self.use_ray_spmd_worker:
            return await super().execute_model_async(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        dag_future = await self.forward_dag.execute_async(execute_model_req)
        outputs = await dag_future
        return outputs[0]

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        assert not self.use_ray_spmd_worker, (
            "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1")
        if not self.tp_driver_workers:
            return await self.driver_exec_method("execute_model",
                                                 execute_model_req)
        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(self.driver_exec_method, self.pp_locks[0],
                                    "execute_model", execute_model_req))
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers,
                                                start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(driver_worker.execute_method.remote,
                                        self.pp_locks[pp_rank],
                                        "execute_model", execute_model_req)))

        results = await asyncio.gather(*tasks)

        # Only the last PP stage has the final results.
        return results[-1]

    async def _start_worker_execution_loop(self):
        assert not self.use_ray_spmd_worker, (
            "worker loop is disabled for VLLM_USE_RAY_SPMD_WORKER=1")
        coros = [
            worker.execute_method.remote("start_worker_execution_loop")
            for worker in self.non_driver_workers
        ]
        return await asyncio.gather(*coros)

    def __del__(self):
        if self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
