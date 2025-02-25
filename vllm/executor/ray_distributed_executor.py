# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import cloudpickle
import msgspec

import vllm.envs as envs
from vllm.executor.executor_base import (
    DistributedExecutorBase)  # yapf: disable
from vllm.executor.msgspec_utils import encode_hook
from vllm.executor.ray_utils import (RayWorkerWrapper, initialize_ray_cluster,
                                     ray)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


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


class RayDistributedExecutor(DistributedExecutorBase):

    uses_ray: bool = True

    def _init_executor(self) -> None:
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None
        if envs.VLLM_USE_V1:
            # v1 always uses the compiled DAG and SPMD worker.
            os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
            os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
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
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.input_encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)
        self.output_decoder = msgspec.msgpack.Decoder(
            Optional[List[SamplerOutput]])
        self.use_v1 = envs.VLLM_USE_V1

        self.pp_locks: Optional[List[asyncio.Lock]] = None
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = make_async(
                self.driver_worker.execute_method)

    def shutdown(self) -> None:
        logger.info(
            "Shutting down Ray distributed executor. If you see error log "
            "from logging.cc regarding SIGTERM received, please ignore because "
            "this is the expected termination process in Ray.")
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

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        logger.info("use_ray_spmd_worker: %s", self.use_ray_spmd_worker)

        # Create the workers.
        bundle_indices: List[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(
                map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, \
            ("VLLM_RAY_BUNDLE_INDICES must have the same size"
            f" as the world size, but got {bundle_indices=} "
            f"and {self.parallel_config.world_size=}")
            assert len(set(bundle_indices)) == len(bundle_indices), \
            ("VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
            f" but got {bundle_indices=}")
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[:self.parallel_config.world_size]

        worker_metadata: List[RayWorkerMetaData] = []
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
                )(RayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                           rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                           rpc_rank=rank)
            worker_metadata.append(
                RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get([
            each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
            for each in worker_metadata
        ])

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        if not self.use_ray_spmd_worker:
            for i, each in enumerate(worker_metadata):
                # find and remove the dummy worker from the list
                worker = each.worker
                worker_ip = each.ip
                if self.driver_dummy_worker is None and worker_ip == driver_ip:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(
                        vllm_config=self.vllm_config, rpc_rank=0)
                    worker_metadata.pop(i)
                    break

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        ip_counts: Dict[str, int] = {}
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
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(worker_metadata,
                                        key=sort_by_driver_then_worker_ip)
        start_rank = 0 if self.use_ray_spmd_worker else 1
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i + start_rank
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self._run_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote()) \
            ) # type: ignore

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
                " each node.")

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [{
            current_platform.device_control_env_var:
            ",".join(map(str, node_gpus[node_id])),
        } for (node_id, _) in worker_node_and_gpu_ids]

        for args in all_args_to_update_environment_variables:
            # some carry-over env vars from the driver
            # TODO: refactor platform-specific env vars
            for name in [
                    "VLLM_ATTENTION_BACKEND",
                    "TPU_CHIPS_PER_HOST_BOUNDS",
                    "TPU_HOST_BOUNDS",
                    "VLLM_USE_V1",
                    "VLLM_TRACE_FUNCTION",
            ]:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          self._get_env_vars_to_be_updated())

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
        self._run_workers("init_worker", all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(
                        self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size
                            ) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
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

        if self.use_v1:
            serialized_data = execute_model_req
        else:
            serialized_data = self.input_encoder.encode(execute_model_req)
        outputs = ray.get(self.forward_dag.execute(serialized_data))
        if self.use_v1:
            output = outputs[0]
        else:
            output = self.output_decoder.decode(outputs[0])
        return output

    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
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
        """
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method
        if self.use_ray_spmd_worker:
            assert not async_run_tensor_parallel_workers_only, (
                "async_run_tensor_parallel_workers_only is not supported for "
                "spmd mode.")

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the ray workers first.
        ray_workers = self.workers
        if async_run_tensor_parallel_workers_only:
            ray_workers = self.non_driver_workers
        ray_worker_outputs = [
            worker.execute_method.remote(sent_method, *args, **kwargs)
            for worker in ray_workers
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_worker_output = []
        # In SPMD mode, the driver worker is the same as any other worker,
        # so we only explicitly execute on the driver worker if using a
        # non-SPMD worker class.
        if not self.use_ray_spmd_worker:
            # Start the driver worker after all the ray workers.
            driver_worker_output = [
                self.driver_worker.execute_method(sent_method, *args, **kwargs)
            ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return driver_worker_output + ray_worker_outputs

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        ray.get(parallel_worker_tasks)

    def _check_ray_adag_installation(self):
        import pkg_resources
        from packaging import version

        required_version = version.parse("2.40")
        current_version = version.parse(
            pkg_resources.get_distribution("ray").version)
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} is "
                             f"required, but found {current_version}")

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
                "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL is set. "
                "Run `pip install ray[adag]` and check cupy installation.")

    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_adag_installation()
        from ray.dag import InputNode, MultiOutputNode
        from ray.experimental.channel.torch_tensor_type import TorchTensorType

        logger.info("VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL = %s",
                    envs.VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL)
        logger.info("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
                    envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM)
        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            # (ExecuteModelReq, None) -> 0 -> (ExecuteModelReq, IntermediateOutput) -> 4 -> SamplerOutput   # noqa: E501
            #                         -> 1 -> (ExecuteModelReq, IntermediateOutput) -> 5 -> SamplerOutput   # noqa: E501
            #                         -> 2 -> (ExecuteModelReq, IntermediateOutput) -> 6 -> SamplerOutput   # noqa: E501
            #                         -> 3 -> (ExecuteModelReq, IntermediateOutput) -> 7 -> SamplerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                if self.use_v1:
                    outputs = [
                        worker.execute_model_ray.
                        bind(  # type: ignore[attr-defined]
                            outputs[i]) for i, worker in enumerate(tp_group)
                    ]
                else:
                    outputs = [
                        worker.execute_model_spmd.
                        bind(  # type: ignore[attr-defined]
                            outputs[i]) for i, worker in enumerate(tp_group)
                    ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if pp_rank < last_pp_rank:
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage.
                    transport = "nccl" \
                        if envs.VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL \
                        else "auto"
                    outputs = [
                        output.with_type_hint(
                            TorchTensorType(transport=transport))
                        for output in outputs
                    ]

            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.
            VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM)

    def __del__(self):
        self.shutdown()

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not self.use_ray_spmd_worker:
            return await super().execute_model_async(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        serialized_data = self.input_encoder.encode(execute_model_req)
        dag_future = await self.forward_dag.execute_async(serialized_data)
        output = await dag_future[0]
        return self.output_decoder.decode(output)

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

    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return
