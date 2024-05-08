import asyncio
import os
import pickle
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

USE_RAY_COMPILED_DAG = envs.VLLM_USE_RAY_COMPILED_DAG


class RayGPUExecutor(DistributedGPUExecutor):

    def _init_executor(self) -> None:
        assert (not self.speculative_config
                ), "Speculative decoding not yet supported for RayGPU backend."

        assert self.parallel_config.worker_use_ray
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.forward_dag = None
        if USE_RAY_COMPILED_DAG:
            self.forward_dag = self._compiled_ray_dag()

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
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(
                worker_module_name="vllm.worker.worker",
                worker_class_name="Worker",
                trust_remote_code=self.model_config.trust_remote_code,
            )

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
                self.driver_worker = RayWorkerWrapper(
                    worker_module_name="vllm.worker.worker",
                    worker_class_name="Worker",
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                # Else, added to the list of workers.
                self.workers.append(worker)

        if self.driver_dummy_worker is None:
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

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={"execute_model_req": execute_model_req},
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        - args/kwargs: All workers share the same args/kwargs
        - args/kwargs and driver_args/driver_kwargs: Driver worker has
          different args
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        if driver_args is None:
            driver_args = args if all_args is None else all_args[0]
        if driver_kwargs is None:
            driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 1, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 1, None)

        if use_ray_compiled_dag:
            # Right now, compiled DAG can only accept a single
            # input. TODO(sang): Fix it.
            assert self.forward_dag is not None
            output_channels = self.forward_dag.execute(1)
        else:
            # Start the ray workers first.
            ray_worker_outputs = [
                worker.execute_method.remote(method, *worker_args,
                                             **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                     ) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]

        # Start the driver worker after all the ray workers.
        if not use_dummy_driver:
            driver_worker_output = self.driver_worker.execute_method(
                method, *driver_args, **driver_kwargs)
        else:
            assert self.driver_dummy_worker is not None
            driver_worker_output = ray.get(
                self.driver_dummy_worker.execute_method.remote(
                    method, *driver_args, **driver_kwargs))
        # Get the results of the ray workers.
        if self.workers:
            if use_ray_compiled_dag:
                try:
                    ray_worker_outputs = [
                        pickle.loads(chan.begin_read())
                        for chan in output_channels
                    ]
                finally:
                    # Has to call end_read in order to reuse the DAG.
                    for chan in output_channels:
                        chan.end_read()
            else:
                ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs

    def _compiled_ray_dag(self):
        import pkg_resources
        required_version = "2.9"
        current_version = pkg_resources.get_distribution("ray").version
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} or greater is "
                             f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        assert self.parallel_config.worker_use_ray

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode([
                worker.execute_model_compiled_dag_remote.
                bind(  # type: ignore[attr-defined]
                    input_data) for worker in self.workers
            ])
        return forward_dag.experimental_compile()

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError("At least one Worker is dead. "
                               f"Dead Workers: {dead_actors}. ")


class RayGPUExecutorAsync(RayGPUExecutor, DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_executor = make_async(self.driver_worker.execute_method)

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        coros.append(
            self.driver_executor(method, *driver_args, **driver_kwargs))

        # Run the ray workers asynchronously.
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))

        all_outputs = await asyncio.gather(*coros)
        return all_outputs
