# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray 工具模块。

本模块提供 Ray 相关的工具函数和类，负责：
- Ray 集群初始化
- Placement Group 管理
- Ray worker 包装器
- 资源验证和等待

主要类：
- RayWorkerWrapper: Ray worker 包装器
- FutureWrapper: Ray 输出引用的包装器

主要函数：
- initialize_ray_cluster: 初始化 Ray 集群
- _verify_bundles: 验证 placement group bundles
- _wait_until_pg_ready: 等待 placement group 就绪
- get_num_tpu_nodes: 获取 TPU 节点数量
"""

import os
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Union

import vllm.platforms
from vllm.config import ParallelConfig
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.network_utils import get_ip
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup

    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x 不暴露 `available_resources_per_node`
        from ray._private.state import state as _state

        available_resources_per_node = _state._available_resources_per_node

    class RayWorkerWrapper(WorkerWrapperBase):
        """vllm.worker.Worker 的 Ray 包装器，允许 Worker 在 Ray 设置 CUDA_VISIBLE_DEVICES 后延迟初始化。

        主要功能：
        - 在 Ray 环境中包装 worker
        - 支持 Ray Compiled DAG 执行
        - 处理设备设置和中间张量传输

        Attributes:
            compiled_dag_cuda_device_set: 是否在编译 DAG 线程上调用了 cuda.set_device
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # 由于编译 DAG 在不同线程中运行主执行
            # 调用 cuda.set_device，此标志表示是否在该线程上调用了 set_device
            self.compiled_dag_cuda_device_set = False

        def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
            """根据给定映射调整 rpc_rank。

            仅在执行器初始化期间使用，在创建所有 workers 后
            调整 workers 的 rpc_rank。

            Args:
                rank_mapping: rank 映射字典
            """
            if self.rpc_rank in rank_mapping:
                self.rpc_rank = rank_mapping[self.rpc_rank]

        def execute_method(self, method: str | bytes, *args, **kwargs):
            """执行方法。

            Args:
                method: 方法名或字节码
                *args: 位置参数
                **kwargs: 关键字参数

            Returns:
                方法执行结果

            Raises:
                Exception: 方法执行异常
            """
            try:
                return run_method(self, method, args, kwargs)
            except Exception as e:
                # 如果 driver worker 也执行方法，
                # 其他 worker 的异常可能导致 rpc 死锁
                # 参考 https://github.com/vllm-project/vllm/issues/3455
                msg = (
                    f"Error executing method {method!r}. "
                    "This might cause deadlock in distributed execution."
                )
                logger.exception(msg)
                raise e

        def get_node_ip(self) -> str:
            """获取节点 IP 地址。

            Returns:
                节点 IP 地址
            """
            return get_ip()

        def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
            """获取节点 ID 和 GPU ID 列表。

            Returns:
                二元组：(node_id, gpu_ids)
            """
            node_id = ray.get_runtime_context().get_node_id()
            device_key = vllm.platforms.current_platform.ray_device_key
            if not device_key:
                raise RuntimeError(
                    "current platform %s does not support ray.",
                    vllm.platforms.current_platform.device_name,
                )
            gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
            return node_id, gpu_ids

        def setup_device_if_necessary(self):
            """必要时设置设备。

            TODO(swang): 这是必需的，因为 Ray CG 在后台线程上执行，
            所以我们需要重置 torch 的当前设备。
            在编译图中修复后可以移除此 API。
            """
            assert self.worker is not None, "Worker is not initialized"
            if not self.compiled_dag_cuda_device_set:
                if current_platform.is_tpu():
                    # 不需要
                    pass
                else:
                    assert self.worker.device is not None
                    current_platform.set_device(self.worker.device)

                self.compiled_dag_cuda_device_set = True

        def execute_model_ray(
            self,
            execute_model_input: tuple["SchedulerOutput", "GrammarOutput"]
            | tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ) -> Union[
            "ModelRunnerOutput",
            tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ]:
            """使用 Ray Compiled Graph 执行模型。

            此方法被 Ray Compiled Graph 用于执行模型，
            需要特殊的 self.setup_device_if_necessary() 逻辑。

            Args:
                execute_model_input: 执行模型输入，包含调度器输出、语法输出和可选的中间张量

            Returns:
                模型 runner 输出或 (scheduler_output, grammar_output, intermediate_tensors)
            """
            self.setup_device_if_necessary()
            assert self.worker is not None, "Worker is not initialized"
            if len(execute_model_input) == 3:
                scheduler_output, grammar_output, intermediate_tensors = (
                    execute_model_input
                )
            else:
                scheduler_output, grammar_output = execute_model_input
                intermediate_tensors = None
            assert self.worker.model_runner is not None
            output = self.worker.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if self._is_intermediate_tensors(output):
                if (
                    self.worker.model_runner.supports_mm_inputs
                    and get_pp_group().is_first_rank
                ):
                    # 在 Ray 转发到下一个 PP Stage 之前剥离 mm_features
                    # PP Stage>0 只需要中间张量，不需要预处理的多模态数据

                    # scheduled_new_reqs 是 SchedulerOutput 的必需字段，
                    # 所以如果缺少它，访问它会引发 AttributeError
                    for req in scheduler_output.scheduled_new_reqs:
                        req.mm_features = []
                return scheduler_output, grammar_output, output

            if isinstance(output, AsyncModelRunnerOutput):
                output = output.get_output()
            if not self._is_last_rank():
                # 没有调度请求但仍可能有已完成请求的情况
                assert not output or not output.req_ids
                output = scheduler_output, grammar_output, None
            elif output is None:
                output = self.worker.model_runner.sample_tokens(grammar_output)
                # 确保跨越 Ray Compiled DAG 的输出是可序列化的
                # AsyncModelRunnerOutput 持有 CUDA 事件且无法 pickled
                if isinstance(output, AsyncModelRunnerOutput):
                    output = output.get_output()
            return output

        def override_env_vars(self, vars: dict[str, str]):
            """覆盖环境变量。

            Args:
                vars: 环境变量字典
            """
            os.environ.update(vars)

        def _is_intermediate_tensors(self, output) -> bool:
            """检查输出是否为中间张量。

            Args:
                output: 输出

            Returns:
                是否为中间张量
            """
            return isinstance(output, IntermediateTensors)

        def _is_last_rank(self) -> bool:
            """是否是最后一个 rank。

            Returns:
                是否是最后一个 rank
            """
            return get_pp_group().is_last_rank

    ray_import_err = None

except ImportError as e:
    ray = None  # type: ignore
    # 仅捕获字符串以避免在某些情况下阻止垃圾回收的 traceback 中的变量引用
    ray_import_err = str(e)
    RayWorkerWrapper = None  # type: ignore


class FutureWrapper(Future):
    """Ray 输出引用的包装器，用于满足 .execute_model() 的接口要求。

    顶层（核心忙循环）期望 .result() API 阻塞并返回单个输出。

    如果提供了 aggregator，则所有 workers 的输出将在 result() 调用时聚合。
    如果未提供，则仅返回第一个 worker 的输出。

    Attributes:
        ref_or_refs: Ray 引用或引用列表
        aggregator: KV 输出聚合器
    """

    def __init__(self, ref_or_refs, aggregator: KVOutputAggregator | None = None):
        super().__init__()
        self.ref_or_refs = ref_or_refs
        self.aggregator = aggregator

    def result(self, timeout=None):
        """获取结果。

        Args:
            timeout: 超时时间

        Returns:
            聚合后的输出
        """
        outputs = ray.get(self.ref_or_refs, timeout=timeout)
        if self.aggregator is None:
            return outputs

        return self.aggregator.aggregate(outputs, output_rank=0)


def ray_is_available() -> bool:
    """返回 Ray 是否可用。

    Returns:
        Ray 是否可用
    """
    return ray is not None


def assert_ray_available():
    """如果 Ray 不可用则抛出异常。

    Raises:
        ValueError: Ray 未安装
    """
    if ray is None:
        raise ValueError(
            f"Failed to import Ray: {ray_import_err}."
            "Please install Ray with `pip install ray`."
        )


def _verify_bundles(
    placement_group: "PlacementGroup", parallel_config: ParallelConfig, device_str: str
):
    """验证给定 placement group 的 bundles 位于正确位置。

    有 2 个规则：
    - 如果所有 tensor parallel workers 无法放入单个节点则发出警告
    - 如果 driver node 未包含在 placement group 中则失败

    Args:
        placement_group: 放置组
        parallel_config: 并行配置
        device_str: 设备字符串（如 "GPU"）

    Raises:
        RuntimeError: 如果 driver node 未包含在 placement group 中
    """
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray."
    )
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (例如，{"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> bundle 列表 (例如，{"GPU": 1})
    node_id_to_bundle: dict[str, list[dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    if driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` and `ray list nodes` to see if you have available "
            "GPUs in a node `{driver_node_id}` before starting an vLLM engine."
        )

    for node_id, bundles in node_id_to_bundle.items():
        if len(bundles) < parallel_config.tensor_parallel_size:
            logger.warning(
                "tensor_parallel_size=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Tensor parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.",
                parallel_config.tensor_parallel_size,
                device_str,
                len(bundles),
                device_str,
                node_id,
                parallel_config.tensor_parallel_size,
            )


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """等待 placement group 就绪。

    如果 placement group 未在规定时间内创建，
    它打印信息性日志消息。

    Args:
        current_placement_group: 当前放置组

    Raises:
        ValueError: 如果无法在规定时间内提供 placement group
    """
    # 等待 PG 就绪 - 这将阻塞直到所有
    # 请求的资源可用，如果无法配置将超时
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # 指数退避警告打印
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check `ray status` and "
            "`ray list nodes` to see if you have enough resources,"
            " and make sure the IP addresses used by ray cluster"
            " are the same as VLLM_HOST_IP environment variable"
            " specified in each node if you are running on a multi-node.",
            int(time.time() - s),
            placement_group_specs,
        )

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        # 当 GPU 数量超额时提供更有用的错误消息
        total_gpu_required = sum(spec.get("GPU", 0) for spec in placement_group_specs)
        # 如果需要多个 GPU，提供更具体的错误消息
        # 我们在这里使用 >1，因为多 GPU（tensor parallel）作业更可能
        # 因集群资源不足而失败，用户可能需要调整 tensor_parallel_size
        if total_gpu_required > 1:
            raise ValueError(
                f"Cannot provide a placement group requiring "
                f"{total_gpu_required} GPUs "
                f"(placement_group_specs={placement_group_specs}) within "
                f"{PG_WAIT_TIMEOUT} seconds.\n"
                f"Tensor parallel size may exceed available GPUs in your "
                f"cluster. Check resources with `ray status` and "
                f"`ray list nodes`.\n"
                f"If running on K8s with limited GPUs, consider reducing "
                f"--tensor-parallel-size to match available GPU resources."
            ) from None
        else:
            raise ValueError(
                "Cannot provide a placement group of "
                f"{placement_group_specs=} within "
                f"{PG_WAIT_TIMEOUT} seconds. See "
                "`ray status` and `ray list nodes` to make sure the cluster "
                "has enough resources."
            ) from None


def _wait_until_pg_removed(current_placement_group: "PlacementGroup"):
    """等待 placement group 被移除。

    Args:
        current_placement_group: 当前放置组
    """
    ray.util.remove_placement_group(current_placement_group)
    s = time.time()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        pg = ray.util.get_current_placement_group()
        if pg is None:
            break

        # 指数退避警告打印
        wait_interval *= 2
        logger.info(
            "Waiting for removing a placement group of specs for %d seconds.",
            int(time.time() - s),
        )
        time.sleep(wait_interval)


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: str | None = None,
):
    """使用 Ray 初始化分布式集群。

    连接到 Ray 集群并为 workers 创建放置组，
    其中包括每个分布式 worker 的资源规格。

    Args:
        parallel_config: 并行执行配置
        ray_address: Ray 集群地址，如果为 None 则使用默认 Ray 集群地址
    """
    assert_ray_available()
    from vllm.platforms import current_platform

    # 在 Ray 处理之前预验证 GPU 需求
    if current_platform.is_cuda() and parallel_config.world_size > 1:
        from vllm.utils.torch_utils import cuda_device_count_stateless

        available_gpus = cuda_device_count_stateless()
        if parallel_config.world_size > available_gpus:
            logger.warning(
                "Tensor parallel size (%d) exceeds available GPUs (%d). "
                "This may result in Ray placement group allocation failures. "
                "Consider reducing tensor_parallel_size to %d or less, "
                "or ensure your Ray cluster has %d GPUs available.",
                parallel_config.world_size,
                available_gpus,
                available_gpus,
                parallel_config.world_size,
            )

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # 尝试连接现有 ray 实例，如果找不到则创建新实例
        try:
            ray.init("auto")
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources."
            )
            ray.init(
                address=ray_address,
                num_gpus=parallel_config.world_size,
                runtime_env=parallel_config.ray_runtime_env,
            )
    else:
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)

    device_str = current_platform.ray_device_key
    if not device_str:
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )

    # 为 worker 进程创建或获取放置组
    if parallel_config.placement_group:
        current_placement_group = parallel_config.placement_group
    else:
        current_placement_group = ray.util.get_current_placement_group()

    if current_placement_group:
        logger.info("Using the existing placement group")

        # 我们在放置组中
        bundles = current_placement_group.bundle_specs
        # 验证我们可以使用放置组
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    f"Placement group bundle cannot have more than 1 {device_str}."
                )
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}."
            )
    else:
        logger.info("No current placement group found. Creating a new placement group.")
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # 记录警告消息并延迟资源分配失败响应
        # 避免立即拒绝以允许用户创建的放置组并等待集群就绪
        if parallel_config.world_size > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.",
                device_str,
                device_str,
            )
        # 创建新放置组
        placement_group_specs: list[dict[str, float]] = [
            {device_str: 1.0} for _ in range(parallel_config.world_size)
        ]

        # vLLM engine 也是使用加速器执行模型的 worker，
        # 所以它需要在当前节点中有设备。检查当前节点
        # 是否至少有一个设备
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}."
            )
        # 这样，至少需要在当前节点中创建 bundle
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # 默认情况下，Ray 尽可能打包资源
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK"
        )
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # 在并行配置中设置放置组
    parallel_config.placement_group = current_placement_group


def get_num_tpu_nodes() -> int:
    """获取 TPU 节点数量。

    Returns:
        TPU 节点数量
    """
    from ray._private.accelerators import TPUAcceleratorManager

    cluster_resources = ray.cluster_resources()
    total_tpus = int(cluster_resources["TPU"])
    tpus_per_node = TPUAcceleratorManager.get_current_node_num_accelerators()
    assert total_tpus % tpus_per_node == 0
    return total_tpus // tpus_per_node


def get_num_nodes_in_placement_group() -> int:
    """获取放置组中的节点数量。

    Returns:
        放置组中的节点数量
    """
    pg_table = ray.util.placement_group_table()
    current_pg = ray.util.get_current_placement_group()
    num_nodes = 0

    if current_pg:
        nodes_in_pg = set()
        for pg_key, pg in pg_table.items():
            if pg_key == current_pg.id.hex():
                for _, node in pg["bundles_to_node_id"].items():
                    nodes_in_pg.add(node)
        num_nodes = len(nodes_in_pg)

    return num_nodes
