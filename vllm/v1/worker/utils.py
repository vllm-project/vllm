# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker 工具函数模块。

本模块提供了 Worker 相关的通用工具函数和接口定义，负责：
- 定义 Worker 基础接口
- 处理 Worker 包装和初始化
- 支持多模态缓存管理
- 提供分布式训练支持

主要类：
- WorkerBase: Worker 基础接口
- WorkerWrapperBase: Worker 包装器
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """Worker 接口，允许 vLLM 清晰分离不同硬件的实现。

    同时抽象了控制平面通信，例如向其他 Worker 传播请求元数据。

    这是所有具体 Worker 实现的基类，定义了 Worker 必须实现的接口。

    Attributes:
        vllm_config: 完整的 vLLM 配置
        model_config: 模型配置
        cache_config: KV 缓存配置
        lora_config: LoRA 配置
        load_config: 加载配置
        parallel_config: 并行配置
        scheduler_config: 调度器配置
        device_config: 设备配置
        speculative_config: 推测解码配置
        observability_config: 可观测性配置
        kv_transfer_config: KV 传输配置
        compilation_config: 编译配置
        local_rank: 本地设备索引
        rank: 分布式全局索引
        distributed_init_method: 分布式初始化方法
        is_driver_worker: 是否为 driver worker
        device: 设备
        model_runner: 模型运行器
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """初始化 Worker 基础组件。

        Args:
            vllm_config: 完整的 vLLM 配置
            local_rank: 本地设备索引
            rank: 分布式全局索引
            distributed_init_method: 分布式初始化方法
            is_driver_worker: 是否负责 driver 职责
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
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # 设备和模型状态
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """获取 KV 缓存实现的规格说明。

        Returns:
            KVCacheSpec 字典，键为层名
        """
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> float:
        """通过编译/预热准备模型执行。

        Returns:
            累积的编译时间（秒）
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """基本健康检查（可覆盖以添加设备特定检查）。"""
        return

    def init_device(self) -> None:
        """初始化设备状态，如加载模型或其他设备内存分配。"""
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        """重置多模态缓存。"""
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        """获取模型实例。"""
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """对模型应用函数。

        Args:
            fn: 应用于模型的函数

        Returns:
            函数返回值
        """
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """返回 transformers 风格的层次化模型视图。

        Returns:
            模型检查字符串
        """
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """将模型加载到目标设备。

        Args:
            load_dummy_weights: 是否加载虚拟权重（用于测试）
        """
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """执行模型前向传播。

        如果此方法返回 None，则应立即调用 sample_tokens 以获取 ModelRunnerOutput。

        注意：如果重新架构结构化输出并行，此设计可能会改变。

        Args:
            scheduler_output: 调度器输出

        Returns:
            ModelRunnerOutput、AsyncModelRunnerOutput 或 None
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """在 execute_model 返回 None 后立即调用。

        Args:
            grammar_output: 语法输出

        Returns:
            ModelRunnerOutput 或 AsyncModelRunnerOutput
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """返回单个缓存块的大小（字节）。

        用于推测解码。

        Returns:
            缓存块大小（字节）
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """添加 LoRA 适配器。

        Args:
            lora_request: LoRA 请求

        Returns:
            是否成功添加
        """
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        """移除 LoRA 适配器。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功移除
        """
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        """固定 LoRA 适配器（防止被驱逐）。

        Args:
            lora_id: LoRA ID

        Returns:
            是否成功固定
        """
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        """列出所有活跃的 LoRA ID。

        Returns:
            LoRA ID 集合
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """从模型配置获取词汇表大小。

        Returns:
            词汇表大小
        """
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """清理 Worker 持有的资源。"""
        return


class WorkerWrapperBase:
    """Worker 包装器类。

    此类代表 executor/engine 中的一个进程。负责延迟初始化 Worker
    并处理 Worker 的生命周期。

    工作流程：
    1. 实例化 WorkerWrapper，记住 Worker 模块和类名
    2. 调用 update_environment_variables 设置环境变量
    3. 调用 init_worker 进行实际的 Worker 初始化

    Attributes:
        rpc_rank: Worker 在 executor 中的索引
        global_rank: Worker 在分布式组中的全局索引
        worker: Worker 实例
        vllm_config: vLLM 配置
        mm_receiver_cache: 多模态接收器缓存
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """使用给定的 vllm_config 和 rpc_rank 初始化 Worker 包装器。

        注意：rpc_rank 是 Worker 在 executor 中的索引。在大多数情况下，
        它也是 Worker 在分布式组中的索引。但是当多个 executor 一起工作时，
        它们可能不同。

        例如：在 SPMD 风格的离线推理中，TP=2，
        用户可以启动 2 个 engines/executors，每个只有 1 个 worker。
        所有 worker 的 rpc_rank=0，但它们在 TP 组中有不同的索引。

        Args:
            rpc_rank: executor 中的 Worker 索引
            global_rank: 分布式全局索引（可选）
        """
        self.rpc_rank = rpc_rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank

        # 在调用 init_worker 后初始化
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        """关闭 Worker。"""
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        """更新环境变量。

        Args:
            envs_list: 环境变量列表，每个元素对应一个 rank
        """
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """初始化 Worker。

        在这里我们注入一些常见的逻辑，然后初始化 Worker。
        参数传递给 Worker 类构造函数。

        Args:
            all_kwargs: 所有 Worker 的参数列表
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "初始化 Worker 需要提供 vllm_config"
        )
        self.vllm_config = vllm_config

        # 启用线程的跟踪函数调用
        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "不再支持传递 worker_cls。"
                "请将类保持在单独的模块中，"
                "并以字符串形式传递类的限定名。"
            )

        # 处理 Worker 扩展类
        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # 检查 worker 和 worker_extension_cls 之间的冲突
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker 类 {worker_class} 已经有属性"
                        f" {attr}，这与 Worker"
                        f" 扩展类 {worker_extension_cls} 冲突。"
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # 动态继承 Worker 扩展类
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "将 %s 注入到 %s 以扩展 collective_rpc 调用 %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        # 处理多模态缓存
        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "缺少 `shared_worker_lock` 参数，这是 executor 需要的。"
                "此参数用于 mm_processor_cache_type='shm'。"
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # 使 vLLM 配置在 Worker 初始化期间可用
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        """从配置初始化 Worker。

        Args:
            kv_cache_configs: KV 缓存配置列表
        """
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        """初始化设备。"""
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # 使 vLLM 配置在设备初始化期间可用
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        """委托给底层 Worker。"""
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        """应用多模态缓存到调度器输出。

        Args:
            scheduler_output: 调度器输出
        """
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """执行模型。

        Args:
            scheduler_output: 调度器输出

        Returns:
            ModelRunnerOutput 或 AsyncModelRunnerOutput
        """
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        """重置多模态缓存。"""
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
