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
USE_RAY_COMPILED_DAG_NCCL = True


class RayGPUExecutor(DistributedGPUExecutor):

    def _init_executor(self) -> None:
        assert self.parallel_config.distributed_executor_backend == "ray"
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.forward_dag = None

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
        # All workers, including the driver dummy worker.
        self.all_workers: List[RayWorkerWrapper] = []

        # Indexed first by PP rank, and then TP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

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
            )

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
            self.all_workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        driver_ip = ray.get(self.driver_dummy_worker.get_node_ip.remote())
        worker_ips = [ray.get(worker.get_node_ip.remote()) for worker in self.workers]
        ip_counts = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(worker):
            ip = ray.get(worker.get_node_ip.remote())
            return (ip != driver_ip, ip_counts[ip], ip)
        self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                use_dummy_driver=True,
                use_local_driver=False)

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

        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank *
                        self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.all_workers[rank])

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={"execute_model_req": execute_model_req},
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]

    def _driver_execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        raise NotImplementedError

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        raise NotImplementedError

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        use_local_driver: bool = True,
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
        if use_ray_compiled_dag:
            # Ray DAG already includes the task for the dummy driver
            # worker.
            use_dummy_driver = False
            # The physical driver does not participate either.
            use_local_driver = False
        elif USE_RAY_COMPILED_DAG:
            # Worker setup: Whatever is normally executed on the local driver
            # should instead be executed on the "dummy" driver worker.
            if use_local_driver:
                use_dummy_driver = use_local_driver
            use_local_driver = False

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
            if self.forward_dag is None:
                self.forward_dag = self._compiled_ray_dag()

            ray_worker_outputs = self.forward_dag.execute(driver_kwargs.pop("execute_model_req"))
        else:
            # Start the ray workers first.
            ray_worker_outputs = [
                worker.execute_method.remote(method, *worker_args,
                                             **worker_kwargs)
                for (worker, worker_args, worker_kwargs
                     ) in zip(self.workers, all_worker_args, all_worker_kwargs)
            ]

        driver_worker_output = []
        # Start the driver worker after all the ray workers.
        if use_dummy_driver:
            assert self.driver_dummy_worker is not None
            driver_worker_output.append(ray.get(
                self.driver_dummy_worker.execute_method.remote(
                    method, *driver_args, **driver_kwargs)))
        if use_local_driver:
            driver_worker_output.append(self.driver_worker.execute_method(
                method, *driver_args, **driver_kwargs))

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return driver_worker_output + ray_worker_outputs

    def _compiled_ray_dag(self):
        import pkg_resources
        required_version = "2.9"
        # current_version = pkg_resources.get_distribution("ray").version
        # if current_version < required_version:
        #     raise ValueError(f"Ray version {required_version} or greater is "
        #                      f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        assert self.parallel_config.distributed_executor_backend == "ray"

        # Right now, compiled DAG requires at least 1 arg. We send
        # a dummy value for now. It will be fixed soon.
        with InputNode() as input_data:
            forward_dag = MultiOutputNode([
                worker.execute_model_compiled_dag_remote.
                bind(  # type: ignore[attr-defined]
                    input_data) for worker in self.all_workers
            ])
        return forward_dag.experimental_compile()


class RayGPUExecutorAsync(RayGPUExecutor, DistributedGPUExecutorAsync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_method = make_async(self.driver_worker.execute_method)

    def _compiled_ray_dag(self):
        import pkg_resources
        required_version = "2.9"
        # current_version = pkg_resources.get_distribution("ray").version
        # if current_version < required_version:
        #     raise ValueError(f"Ray version {required_version} or greater is "
        #                      f"required, but found {current_version}")

        from ray.dag import InputNode, MultiOutputNode
        from ray.experimental.channel.torch_tensor_type import TorchTensorType
        assert self.parallel_config.distributed_executor_backend == "ray"

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            # (ExecuteModelReq, None) -> 0 -> (ExecuteModelReq, IntermediateOutput) -> 4 -> SamplerOutput
            #                         -> 1 -> (ExecuteModelReq, IntermediateOutput) -> 5 -> SamplerOutput
            #                         -> 2 -> (ExecuteModelReq, IntermediateOutput) -> 6 -> SamplerOutput
            #                         -> 3 -> (ExecuteModelReq, IntermediateOutput) -> 7 -> SamplerOutput
            # All workers in the first TP group will take in the

            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker.
                outputs = [worker.execute_model_compiled_dag_remote.bind(outputs[i])
                           for i, worker in enumerate(tp_group)]

                # Specify how intermediate tensors should be passed between
                # workers.
                if pp_rank < len(self.pp_tp_workers) - 1:
                    if USE_RAY_COMPILED_DAG_NCCL:
                        # Transfer the tensors through NCCL.
                        outputs = [output.with_type_hint(TorchTensorType(transport="nccl"))
                                   for output in outputs]
                    else:
                        # Just transfer the tensors through Ray's shared memory
                        # store.
                        outputs = [output.with_type_hint(TorchTensorType())
                                   for output in outputs]

            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile(enable_asyncio=True)

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        assert USE_RAY_COMPILED_DAG
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag()
        fut = await self.forward_dag.execute_async((execute_model_req, None))
        outputs = await fut
        # from ray.experimental.compiled_dag_ref import RayDAGTaskError
        # if isinstance(outputs[0], RayDAGTaskError):
        #     print(outputs[0])
        # Only the first driver returns sampler output.
        return outputs[0]

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        async with self.pp_locks[0]:
            output = await self.driver_exec_method("execute_model",
                                                   execute_model_req)
        for pp_rank, tp_group in enumerate(self.pp_tp_workers[1:]):
            async with self.pp_locks[pp_rank + 1]:
                output = await tp_group[0].execute_method.remote(
                    "execute_model", execute_model_req)
        return output

    async def _start_worker_execution_loop(self):
        coros = []
        for pp_rank, tp_group in enumerate(self.pp_tp_workers):
            coros += [
                worker.execute_method.remote("start_worker_execution_loop")
                for worker in tp_group[1:]
                ]
        return await asyncio.gather(*coros)
