import os
from collections import defaultdict
from itertools import islice, repeat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_utils import (RayWorkerWrapper,
                                        initialize_ray_cluster, ray)
from vllm.v1.outputs import ModelRunnerOutput

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


class RayExecutor(Executor):

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # A list of workers to run a model.
        self.workers: List[RayWorkerWrapper] = []
        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                # Skip bundles that don't have GPUs,
                # as each worker needs one GPU.
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(vllm_config=self.vllm_config)
            self.workers.append(worker)

        logger.debug("workers: %s", self.workers)
        worker_ips = [
            ray.get(worker.get_node_ip.remote())  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        worker_to_ip = dict(zip(self.workers, worker_ips))

        def sort_by_driver_then_worker_ip(worker):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first. This is simply a tiebreaker to make
                sure the workers are sorted in a deterministic way.
            """
            ip = worker_to_ip[worker]
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

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
            "VLLM_USE_V1":
            str(int(envs.VLLM_USE_V1)),
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

    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Return worker init args for a given rank.
        """
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks.
        
        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.
        
        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks")

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize(self, num_gpu_blocks: int) -> None:
        """
        Initialize the KV cache in all workers.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d", num_gpu_blocks)
        self._run_workers("initialize_cache", num_gpu_blocks)
        self._run_workers("compile_or_warm_up_model")

    def _run_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 0, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 0, None)

        ray_worker_refs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(self.workers, all_worker_args, all_worker_kwargs)
        ]
        return ray.get(ray_worker_refs)

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag()
        # Only the first worker (with rank 0) returns the execution result.
        # Others return None.
        output = ray.get(self.forward_dag.execute(scheduler_output))[0]
        return output

    def profile(self, is_start=True):
        raise NotImplementedError

    def shutdown(self):
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    def check_health(self) -> None:
        logger.debug("Called check_health.")

    def _check_ray_compiled_graph_installation(self):
        import pkg_resources
        from packaging import version

        required_version = version.parse("2.39")
        current_version = version.parse(
            pkg_resources.get_distribution("ray").version)
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} is "
                             f"required, but found {current_version}")

        import importlib.util
        raycg = importlib.util.find_spec("ray.experimental.compiled_dag_ref")
        if raycg is None:
            raise ValueError("Ray Compiled Graph is not installed. "
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

        with InputNode() as input_batches:
            outputs = [
                worker.execute_model.bind(  # type: ignore[attr-defined]
                    input_batches) for worker in self.workers
            ]
            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile()

    def __del__(self):
        self.shutdown()
