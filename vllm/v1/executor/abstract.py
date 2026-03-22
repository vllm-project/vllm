# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor 抽象基类模块。

本模块定义了 vLLM V1 执行器的抽象基类，负责：
- 定义执行器接口和通用功能
- 管理 worker 的生命周期
- 执行模型推理和采样
- 支持分布式执行

主要类：
- Executor: 执行器抽象基类

执行器是 vLLM 的核心组件，负责在一个或多个设备上执行模型。
它可以是单设备执行器，也可以是支持多设备分布式执行的执行器。
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class Executor(ABC):
    """vLLM 执行器的抽象基类。

    执行器负责在一个设备上执行模型，
    或者作为分布式执行器在多个设备上执行模型。

    主要功能：
    - 初始化和配置 worker
    - 执行模型推理（execute_model）
    - 执行采样（sample_tokens）
    - 管理 LoRA 适配器
    - 支持睡眠/唤醒功能
    - 健康检查

    Attributes:
        uses_ray: 是否使用 Ray 进行编排
        supports_pp: 是否支持流水线并行（Pipeline Parallelism）
        vllm_config: vLLM 配置
        is_sleeping: 是否处于睡眠状态
        sleeping_tags: 睡眠标签集合
        kv_output_aggregator: KV 输出聚合器
    """

    uses_ray: bool = False  # 执行器是否使用 Ray 进行编排
    supports_pp: bool = False  # 执行器是否支持流水线并行

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """根据配置获取执行器类。

        根据 distributed_executor_backend 配置选择对应的执行器类。

        Args:
            vllm_config: vLLM 配置

        Returns:
            执行器类

        Raises:
            TypeError: 如果 distributed_executor_backend 不是 Executor 的子类
            ValueError: 如果 distributed_executor_backend 未知
        """
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend 必须在 VllmConfig.__post_init__ 中设置
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_executor import RayDistributedExecutor

            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: 使 v1 调度确定性以支持外部启动器
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        """初始化执行器。

        Args:
            vllm_config: vLLM 配置
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator: KVOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        """初始化执行器。子类必须实现此方法。"""
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """从配置初始化 KV 缓存并开始模型执行循环。

        初始化底层 worker 的 KV 缓存并开始模型执行循环。

        Args:
            kv_cache_configs: KV 缓存配置列表
        """
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        compilation_times: list[float] = self.collective_rpc("compile_or_warm_up_model")
        # 将编译时间从 worker 传播回主进程
        # 当 TP>1 时，编译在 worker 进程中进行，因此主进程配置不会被更新
        # 使用 worker 中的最大值，因为它们是并行编译的
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """注册失败回调函数。

        当执行器进入永久失败状态时调用此函数。

        Args:
            callback: 失败回调函数
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        """确定可用内存（字节）。

        Returns:
            可用内存列表（字节）
        """
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """获取 KV 缓存规格。

        Returns:
            KV 缓存规格列表
        """
        return self.collective_rpc("get_kv_cache_spec")

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        在所有 worker 上执行 RPC 调用。

        Args:
            method: 要执行的 worker 方法名称或可序列化并发送到所有 worker 的 callable
                如果是 callable，应接受额外的 `self` 参数，以及 `args` 和 `kwargs` 中传入的参数
            timeout: 最大等待时间（秒），超时抛出 TimeoutError，None 表示无限等待
            args: 传递给 worker 方法的位置参数
            kwargs: 传递给 worker 方法的关键字参数
            non_block: 如果为 True，返回 Future 列表而不是等待结果

        Returns:
            包含每个 worker 结果的列表

        Note:
            建议使用此 API 仅传递控制消息，并设置数据平面通信来传递数据。
        """
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        """在所有 worker 上执行 RPC 调用。子类必须实现此方法。"""
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[int, KVConnectorHandshakeMetadata]]:
        """获取 KV 连接器握手元数据。

        Returns:
            KV 连接器握手元数据列表
        """
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """执行模型推理。

        Args:
            scheduler_output: 调度器输出
            non_block: 如果为 True，返回 Future 而不是等待结果

        Returns:
            模型 runner 输出或 None
        """
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """执行 token 采样。

        Args:
            grammar_output: 语法输出（用于结构化输出）
            non_block: 如果为 True，返回 Future 而不是等待结果

        Returns:
            模型 runner 输出
        """
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        """执行虚拟批次（用于预热和测试）。"""
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """获取草稿 token ID（用于推测解码）。

        Returns:
            草稿 token ID 或 None
        """
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        """最大并发批次数量。

        Returns:
            最大并发批次数量
        """
        return 1

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """启动或停止性能分析。

        Args:
            is_start: True 启动分析，False 停止分析
            profile_prefix: 分析文件前缀
        """
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        """保存分片状态。

        Args:
            path: 保存路径
            pattern: 文件名模式（可选）
            max_size: 每个分片的最大大小（可选）
        """
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """检查执行器是否健康。如果不健康，应抛出异常。"""
        raise NotImplementedError

    def shutdown(self) -> None:
        """关闭执行器。"""
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """初始化 KVOutputAggregator。

        Args:
            connector: KV 连接器
        """
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    @cached_property  # 避免不必要的 RPC 调用
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取支持的任务类型。

        Returns:
            支持的任务元组
        """
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """添加 LoRA 适配器。

        Args:
            lora_request: LoRA 请求

        Returns:
            是否成功添加
        """
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        """移除 LoRA 适配器。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功移除
        """
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        """固定 LoRA 适配器（防止被驱逐）。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功固定
        """
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        """列出所有 LoRA ID。

        Returns:
            LoRA ID 集合
        """
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """重置每个 worker 中的多模态缓存。"""
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """重置每个 worker 中的编码器缓存以清除缓存的编码器输出。"""
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        """使执行器进入睡眠状态（释放资源）。

        Args:
            level: 睡眠级别
        """
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        """唤醒执行器（恢复资源）。

        Args:
            tags: 要唤醒的标签列表，None 表示唤醒所有
        """
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        """重新初始化分布式配置。

        Args:
            reconfig_request: 重新配置请求
        """
        raise NotImplementedError

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        是否支持异步调度。

        Returns:
            是否支持异步调度
        """
        return False


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# 向后兼容
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
