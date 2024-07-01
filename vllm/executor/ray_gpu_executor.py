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
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, make_async)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# If the env var is set, it uses the Ray's compiled DAG API
# which optimizes the control plane overhead.
# Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
# Currently, this requires USE_SPMD_WORKER=True.
USE_RAY_COMPILED_DAG = envs.VLLM_USE_RAY_COMPILED_DAG
# If the env var is set, then we do not distinguish between the "driver worker"
# vs other workers. Also, the rank 0 worker will be executed in a remote Ray
# worker. Currently this requires USE_RAY_COMPILED_DAG=True.
USE_SPMD_WORKER = envs.VLLM_USE_SPMD_WORKER


class RayGPUExecutor(DistributedGPUExecutor):

    def _init_executor(self) -> None:
        if USE_RAY_COMPILED_DAG:
            assert USE_SPMD_WORKER, (
                "VLLM_USE_RAY_COMPILED_DAG=1 requires VLLM_USE_SPMD_WORKER=1")
        if USE_SPMD_WORKER:
            # TODO: Support SPMD worker for non-DAG Ray executor.
            assert USE_RAY_COMPILED_DAG, ("VLLM_USE_SPMD_WORKER=1 requires "
                                          "VLLM_USE_RAY_COMPILED_DAG=1")

        assert self.parallel_config.distributed_executor_backend == "ray"
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

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if self.parallel_config.tensor_parallel_size == 1:
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
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if self.speculative_config is not None:
                worker_module_name = "vllm.spec_decode.spec_decode_worker"
                worker_class_name = "create_spec_worker"
            else:
                worker_module_name = "vllm.worker.worker"
                worker_class_name = "Worker"

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(
                worker_module_name=worker_module_name,
                worker_class_name=worker_class_name,
                trust_remote_code=self.model_config.trust_remote_code,
                use_spmd_worker=USE_SPMD_WORKER,
            )

            if USE_SPMD_WORKER:
                self.workers.append(worker)
            else:
                worker_ip = ray.get(worker.get_node_ip.remote())
                if worker_ip == driver_ip and self.driver_dummy_worker is None:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(
                        worker_module_name=worker_module_name,
                        worker_class_name=worker_class_name,
                        trust_remote_code=self.model_config.trust_remote_code,
                    )
                else:
                    # Else, added to the list of workers.
                    self.workers.append(worker)

        if not USE_SPMD_WORKER and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

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

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        assert not USE_SPMD_WORKER, (
            "driver_worker does not exist for VLLM_USE_SPMD_WORKER=1")
        return self.driver_worker.execute_method("execute_model",
                                                 execute_model_req)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not USE_SPMD_WORKER:
            return super().execute_model(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        outputs = ray.get(self.forward_dag.execute(execute_model_req))
        return outputs[0]

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_remote_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        - async_run_remote_workers_only: If True the method will be run only
          in the remote workers, not the driver worker. It will also be
          run asynchronously and return a list of futures rather than blocking
          on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        count = len(self.workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0 if USE_SPMD_WORKER else 1
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(self.workers, all_worker_args, all_worker_kwargs)
        ]

        if async_run_remote_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_worker_output = []
        # In SPMD mode, the driver worker is the same as any other worker,
        # so we only explicitly execute on the driver worker if using a
        # non-SPMD worker class.
        if not USE_SPMD_WORKER:
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

        # TODO(swang): Upgrade version.
        required_version = "2.9"
        current_version = pkg_resources.get_distribution("ray").version
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} or greater is "
                             f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        assert self.parallel_config.distributed_executor_backend == "ray"

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode([
                worker.execute_model_spmd.bind(  # type: ignore[attr-defined]
                    input_data) for worker in self.workers
            ])
        return forward_dag.experimental_compile(enable_asyncio=enable_asyncio)


class RayGPUExecutorAsync(RayGPUExecutor, DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_method = make_async(self.driver_worker.execute_method)

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not USE_SPMD_WORKER:
            return await super().execute_model_async(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        outputs = await self.forward_dag.execute_async(execute_model_req)
        outputs = await outputs
        return outputs[0]

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        assert not USE_SPMD_WORKER, (
            "driver_worker does not exist for VLLM_USE_SPMD_WORKER=1")
        return await self.driver_exec_method("execute_model",
                                             execute_model_req)

    async def _start_worker_execution_loop(self):
        assert not USE_SPMD_WORKER, (
            "worker loop is disabled for VLLM_USE_SPMD_WORKER=1")
        coros = [
            worker.execute_method.remote("start_worker_execution_loop")
            for worker in self.workers
        ]
        return await asyncio.gather(*coros)
