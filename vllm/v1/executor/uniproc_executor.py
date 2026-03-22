# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""单进程执行器模块。

本模块实现了单进程执行器，负责：
- 在单个进程中执行模型
- 支持同步和异步调度
- 支持外部启动器（如 torchrun）

主要类：
- UniProcExecutor: 单进程执行器
- ExecutorWithExternalLauncher: 使用外部启动器的执行器

UniProcExecutor 适用于单卡场景或调试场景，无需多进程或 Ray。
"""

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(Executor):
    """单进程执行器。

    在单个进程中执行模型，适用于单卡场景或调试场景。
    无需多进程或 Ray 编排。

    主要功能：
    - 初始化单个 worker
    - 执行模型推理和采样
    - 支持异步调度（可选）
    - 支持外部启动器

    Attributes:
        driver_worker: 驱动 worker 包装器
        async_output_thread: 异步输出线程池
    """

    def _init_executor(self) -> None:
        """初始化 worker 并加载模型。"""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        self.async_output_thread: ThreadPoolExecutor | None = None
        if self.max_concurrent_batches > 1:
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
            )

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)

    def _distributed_args(self) -> tuple[str, int, int]:
        """返回分布式初始化参数 (distributed_init_method, rank, local_rank)。

        Returns:
            三元组：
                - distributed_init_method: 分布式初始化方法
                - rank: 全局 rank
                - local_rank: 本地 rank
        """
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # 如果指定了设备索引，将 local_rank 设置为设备索引
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        """最大并发批次数量。

        Returns:
            如果使用异步调度则为 2，否则为 1
        """
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        """在 worker 上执行 RPC 调用。

        Args:
            method: 方法名或 callable
            timeout: 超时时间（未使用）
            args: 位置参数
            kwargs: 关键字参数
            non_block: 是否非阻塞
            single_value: 是否返回单个值（而不是列表）

        Returns:
            RPC 结果或 Future
        """
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    if single_value:
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]:
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """执行模型推理。

        Args:
            scheduler_output: 调度器输出
            non_block: 是否非阻塞

        Returns:
            模型 runner 输出或 Future
        """
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # 在非阻塞模式下，尽早暴露任何异常
        if non_block and output.done():
            # 如果任务失败，原地抛出异常
            output.result()
        return output

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """执行 token 采样。

        Args:
            grammar_output: 语法输出
            non_block: 是否非阻塞

        Returns:
            模型 runner 输出或 Future
        """
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """获取草稿 token ID。

        Returns:
            草稿 token ID 或 None
        """
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        """检查健康状态。

        UniProcExecutor 只要运行中就始终健康。
        """
        # UniProcExecutor 将始终健康
        return

    def shutdown(self) -> None:
        """关闭执行器。"""
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """是否支持异步调度。

        Returns:
            True（支持异步调度）
        """
        return True


class ExecutorWithExternalLauncher(UniProcExecutor):
    """使用外部启动器的执行器。

    专门设计用于 torchrun 兼容的启动器，用于
    带有张量并行的离线推理。

    参考 https://github.com/vllm-project/vllm/issues/11400 了解
    动机，以及 examples/offline_inference/torchrun_example.py 了解
    用法示例。

    关键点：虽然是张量并行推理，但我们每个 executor 只创建
    一个 worker，用户将使用 torchrun 兼容的启动器启动多个
    engine，所有这些 engine 共同处理相同的 prompts。当调度是
    确定性时，所有 engine 将生成相同的 outputs，
    它们不需要相互同步状态。

    Attributes:
        继承自 UniProcExecutor
    """

    def _init_executor(self) -> None:
        """初始化 worker 并加载模型。

        Raises:
            AssertionError: 如果 VLLM_ENABLE_V1_MULTIPROCESSING 未设置为 0
        """
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        """返回分布式初始化参数。

        engine 使用 torchrun 兼容的启动器启动
        因此我们可以使用 env:// 方法。
        必需的环境变量：
        - RANK
        - LOCAL_RANK
        - MASTER_ADDR
        - MASTER_PORT

        Returns:
            三元组：
                - distributed_init_method: "env://"
                - rank: 从环境变量读取
                - local_rank: 从环境变量读取
        """
        # engine 使用 torchrun 兼容的启动器启动
        # 所以我们可以使用 env:// 方法
        # 必需的环境变量：
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        """确定可用内存（字节）。

        需要在所有 ranks 之间获取最小值。

        Returns:
            可用内存列表（所有 ranks 的最小值）
        """
        # 我们需要获取所有 ranks 之间的最小值
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
