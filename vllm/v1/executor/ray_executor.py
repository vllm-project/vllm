# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray 执行器模块。

本模块实现了基于 Ray 的分布式执行器，负责：
- 使用 Ray 编排分布式模型执行
- 管理 Ray worker 的生命周期
- 支持流水线并行和张量并行
- 支持 Ray Compiled DAG 优化

主要类：
- RayDistributedExecutor: Ray 分布式执行器
- RayWorkerMetaData: Ray worker 元数据

Ray 执行器适用于多机多卡分布式场景，支持弹性伸缩。
"""

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
    """Ray worker 元数据。

    Ray worker 创建顺序可能是随机的，
    我们需要在创建所有 workers 后重置 rank。

    Attributes:
        worker: Ray worker ActorHandle
        created_rank: 创建时的 rank
        adjusted_rank: 调整后的 rank
        ip: worker IP 地址
    """

    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


class RayDistributedExecutor(Executor):
    """基于 Ray 的分布式执行器。

    使用 Ray 进行分布式编排，支持：
    - 多机多卡分布式执行
    - 流水线并行（Pipeline Parallelism）
    - 张量并行（Tensor Parallelism）
    - Ray Compiled DAG 优化
    - KV 连接器聚合

    Attributes:
        uses_ray: True（使用 Ray）
        supports_pp: True（支持流水线并行）
        forward_dag: Ray Compiled DAG
        workers: Ray worker 列表
        pp_tp_workers: 按 PP 和 TP 组织的 worker 列表
        driver_dummy_worker: 驱动虚拟 worker
        has_connector: 是否有 KV 连接器
        uses_sampler: 是否使用采样器
        scheduler_output: 调度器输出（用于 deferred execution）
    """

    # 这些环境变量是 worker 特定的，因此不从 driver 复制到 workers
    WORKER_SPECIFIC_ENV_VARS = {
        "VLLM_HOST_IP",
        "VLLM_HOST_PORT",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    }

    uses_ray: bool = True
    supports_pp: bool = True

    def _init_executor(self) -> None:
        """初始化执行器。"""
        self.forward_dag: ray.dag.CompiledDAG | None = None

        # 对于 TPU 或 XPU，避免编译 NVIDIA 的 NCCL
        if current_platform.is_tpu() or current_platform.is_xpu():
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # 禁用 Ray 使用统计收集
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # 创建并行 GPU workers
        self._init_workers_ray(placement_group)

        # KV 连接器设置
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or self.vllm_config.ec_transfer_config.is_ec_consumer
        )

        self.scheduler_output: SchedulerOutput | None = None

    @property
    def max_concurrent_batches(self) -> int:
        """最大并发批次数量。

        Ray 分布式执行器支持流水线并行，
        因此允许执行 PP size 个并发批次。

        Returns:
            最大并发批次数量
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def shutdown(self) -> None:
        """关闭执行器。"""
        if logger:
            # 这里 logger 可能为 None
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
        """配置 Ray workers 使用 nsight 分析。

        如果启用了 nsight 分析，我们需要将分析配置设置为
        Ray workers 的运行时环境。

        Args:
            ray_remote_kwargs: Ray remote 参数

        Returns:
            更新后的参数
        """
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
        """更新设备相关的环境变量。

        Args:
            ray_remote_kwargs: Ray remote 参数

        Returns:
            更新后的参数
        """
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars.update(
            {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
        )
        return ray_remote_kwargs

    # 子类可以重写此方法以返回实际的环境变量
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        """使用 Ray 初始化 workers。

        Args:
            placement_group: Ray 放置组
            **ray_remote_kwargs: Ray remote 参数
        """
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # 驱动虚拟 worker 实际上不使用任何资源
        # 它为驱动 worker 保留资源
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # 剩余的 workers 是实际的 Ray actors
        self.workers: list[RayWorkerWrapper] = []

        # 在 Ray Compiled DAG 中使用：首先按 PP rank 索引，然后按 TP rank 索引
        # 换句话说，内部列表是 PP rank 的 TP 组 workers
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )

        # Ray actors 的设置在 vllm 中是 visible devices 不由 actors 设置
        # 它们由 ray 保留未设置，我们在内部使用 local_rank 索引正确的 gpu
        # 这类似于 mp 模式的工作方式
        self._update_noset_device_env_vars(ray_remote_kwargs)

        # 创建 workers
        bundle_indices: list[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # 使用用户指定的 bundle indices
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
            # 使用有 GPU 资源的前 N 个 bundles
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
                # NV+AMD GPUs，和 Intel XPUs
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
            根据 3 个属性对 workers 排序：
            1. 如果 worker 在与 driver（vllm engine）相同的节点上，应首先放置
            2. 然后，如果 worker 在 worker 数量较少的节点上，应首先放置
            3. 最后，如果 worker 在 IP 地址较小的节点上，应首先放置
            """
            ip = item.ip
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip

        # 排序后，同一节点上的 workers 将彼此靠近，
        # driver 节点上的 workers 将被首先放置
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

        # 获取每个节点上使用的 GPU ID 集合
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # 使用 ray spmd worker时，driver_dummy_worker 可能为 None
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote())  # type: ignore[attr-defined]
            )

        node_workers = defaultdict(list)  # node id -> worker rank 列表
        node_gpus = defaultdict(list)  # node id -> gpu id 列表

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` 可能是字符串或整数列表
            # 将它们转换为整数以保持一致性
            # 注意：gpu_ids 可能大于 9（例如 16 个 GPU）
            # 字符串排序不够
            # 参考 https://github.com/vllm-project/vllm/issues/5590
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

        # 为 driver 和 workers 设置环境变量
        # 我们将 CUDA_VISIBLE_DEVICES 设置为节点上的所有 GPU
        # 这是必需的，因为：
        # 1. Ray 的 compiled DAG 需要在 CUDA_VISIBLE_DEVICES 中找到分配的 GPU
        # 2. vLLM 的通信层（NCCL, CustomAllreduce）需要看到所有 GPU
        #    用于 P2P 检查和通信设置。虽然如果只是因为这一点，
        #    我们也可以保持 visible devices 未设置
        # 每个 worker 将使用 local_rank 索引到 visible devices
        all_args_to_update_environment_variables = [
            {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            for (node_id, _) in worker_node_and_gpu_ids
        ]

        # 从 driver 复制到 workers 的环境变量
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers",
        )

        # 将现有环境变量复制到每个 worker 的参数
        for args in all_args_to_update_environment_variables:
            # TODO: 重构平台特定的环境变量
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )

        if len(node_gpus) == 1:
            # 在单节点情况下，我们不需要获取 IP 地址
            # loopback 地址就足够了
            # 注意：一个节点可能有多个 IP 地址，每个网络接口一个
            # `get_ip()` 可能返回其中任何一个，
            # 如果网络设置复杂，它们可能无法在节点内通信
            # 使用 loopback 地址解决了这个问题，因为它始终适用于节点内通信
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # 初始化 worker wrapper 内部的 actual workers
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
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.collective_rpc("elastic_ep_execute", args=("load_model",))
        else:
            self.collective_rpc("load_model")

        def _update_block_size(worker):
            current_platform.update_block_size_for_backend(worker.vllm_config)

        self.collective_rpc(_update_block_size)

        # 按 PP 和 TP 组织 workers
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
        """重新初始化分布式配置。

        Args:
            reconfig_request: 重新配置请求
        """
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
        """执行模型推理。

        Args:
            scheduler_output: 调度器输出
            non_block: 如果为 True，返回 Future

        Returns:
            模型 runner 输出或 None 或 Future
        """
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # 模型不会执行，立即调用 model runner
            return self._execute_dag(scheduler_output, None, non_block)

        # 模型将执行，推迟到 sample_tokens() 调用
        self.scheduler_output = scheduler_output
        return COMPLETED_NONE_FUTURE if non_block else None

    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """在 Ray workers 上执行模型。

        使用的调度器输出应该在之前的 execute_model() 调用中提供。

        Args:
            grammar_output: 结构化输出语法位掩码（如果适用）
            non_block: 如果为 True，返回 Future

        Returns:
            模型 runner 输出
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
        """执行 Ray Compiled DAG。

        Args:
            scheduler_output: 调度器输出
            grammar_output: 语法输出
            non_block: 如果为 True，返回 Future

        Returns:
            模型 runner 输出或 None 或 Future
        """
        # 首次构建 compiled DAG
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore

        if not self.has_connector:
            # 仅从单个 worker（output_rank）获取输出
            # 当不使用 PP 时，我们在这里阻塞直到结果可用
            if not non_block:
                return refs[0].get()

            # 当使用 PP 时，我们立即返回 FutureWrapper，以便
            # 调度器可以 yield 到下一批次
            return FutureWrapper(refs[0])

        # 当存在连接器时，从所有 workers 获取输出
        assert self.kv_output_aggregator is not None
        if not non_block:
            # 阻塞并从所有 workers 获取结果
            return self.kv_output_aggregator.aggregate(ray.get(refs))

        # 返回一个 future，将从所有 workers 聚合输出
        return FutureWrapper(refs, self.kv_output_aggregator)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
        """在所有 workers 上运行给定方法。

        Args:
            method: 方法名或 callable
            timeout: 超时时间
            args: 位置参数
            kwargs: 关键字参数
            non_block: 如果为 True，返回 Future

        Returns:
            结果列表或 Future
        """
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

        # 获取 ray workers 的结果
        if non_block:
            return FutureWrapper(ray_worker_outputs)

        return ray.get(ray_worker_outputs, timeout=timeout)

    def _check_ray_cgraph_installation(self):
        """检查 Ray Compiled Graph 安装。

        Raises:
            ValueError: 如果 Ray 版本过低或缺少必要的包
        """
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
        """编译 Ray DAG。

        Args:
            enable_asyncio: 是否启用 asyncio

        Returns:
            编译后的 Ray DAG
        """
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        # 将 "RAY_CGRAPH_get_timeout" 的默认值增大到 300 秒
        # （默认是 10 秒）。这是一个 Ray 环境变量，用于
        # 控制从 compiled graph execution 获取结果的超时时间，
        # 即包括模型前向运行和中间张量通信的分布式执行
        # 注意：我们应该在导入 ray.dag 之前设置此环境变量，
        # 否则它将不会生效
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
            # 示例 DAG：PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # 第一个 TP 组的所有 workers 将接收
            # ExecuteModelRequest 作为输入
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # 每个 PP worker 接收前一个 PP worker 的输出，
                # TP 组以 SPMD 方式执行
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(tp_group)
                ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if (
                    pp_rank < last_pp_rank
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                ):
                    # 指定中间张量如何在 pp stages 之间传递，
                    # 最后一个 pp stage 或使用共享内存（默认）时不需要指定
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
        """析构函数，关闭执行器。"""
        self.shutdown()

    def check_health(self) -> None:
        """检查健康状态。

        假设 Ray workers 是健康的。
        TODO: 检查 Ray workers 的健康状态
        """
        # 假设 Ray workers 是健康的
        # TODO: 检查 Ray workers 的健康状态
        return
