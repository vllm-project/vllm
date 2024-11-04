import os
from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.v1.executor.distributed_gpu_executor import DistributedGPUExecutor
from vllm.v1.executor.ray_utils import RayWorkerWrapper, ray
from vllm.logger import init_logger
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, get_vllm_instance_id)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.scheduler import SchedulerOutput

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


class RayGPUExecutor(DistributedGPUExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_executor()

    def _init_executor(self) -> None:
        self.forward_dag: Optional["ray.dag.CompiledDAG"] = None
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

    def shutdown(self) -> None:
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
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

    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        worker_module_name, worker_class_name = self._get_worker_module_and_class()

        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if (self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1):
            # For single GPU case, we use a ray worker with constrained memory.
            # TODO-SANG Q: Is it necessary?
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # A list of workers to run a model.
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
            self.workers.append(worker)

        logger.debug("workers: %s", self.workers)
        worker_ips = [
            ray.get(worker.get_node_ip.remote())  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(worker):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = ray.get(worker.get_node_ip.remote())
            return (ip != driver_ip, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids")

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

        all_ips = set(worker_ips)
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP` or "
                "`HOST_IP` environment variable, make sure it is unique for"
                " each node.")

        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
            **({
                "VLLM_ATTENTION_BACKEND": envs.VLLM_ATTENTION_BACKEND
            } if envs.VLLM_ATTENTION_BACKEND is not None else {})
        }, ) for (node_id, _) in worker_node_and_gpu_ids]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          all_args=self._get_env_vars_to_be_updated())

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
        self._run_workers("initialize")
        self._run_workers("load_model")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag()
        # All workers are supposed to produce the same output. Only
        # get the first output.
        output = ray.get(self.forward_dag.execute(scheduler_output))[0]
        return output

    def _run_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
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
        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 0, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 0, None)

        # Start the ray workers first.
        ray_workers = self.workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs

    def _check_ray_compiled_graph_installation(self):
        import pkg_resources
        from packaging import version

        import importlib.util
        adag_spec = importlib.util.find_spec(
            "ray.experimental.compiled_dag_ref")
        if adag_spec is None:
            raise ValueError("Ray accelerated DAG is not installed. "
                             "Run `pip install ray[adag]` to install it.")

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL:
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL is set."
                "Run `pip install ray[adag]` and check cupy installation.")

    def _compiled_ray_dag(self):
        assert self.parallel_config.use_ray
        self._check_ray_compiled_graph_installation()
        from ray.dag import InputNode, MultiOutputNode
        from ray.experimental.channel.torch_tensor_type import TorchTensorType

        with InputNode() as input_batches:
            outputs = [
                worker.execute_model.bind(input_batches)
                    for worker in self.workers
            ]
            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile()

    def __del__(self):
        self.shutdown()
