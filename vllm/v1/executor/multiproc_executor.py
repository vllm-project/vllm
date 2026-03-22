# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""多进程执行器模块。

本模块实现了基于多进程的分布式执行器，负责：
- 使用 multiprocessing 创建 worker 进程
- 通过共享内存消息队列进行通信
- 支持流水线并行和数据并行
- 监控 worker 健康状态

主要类：
- MultiprocExecutor: 多进程执行器
- WorkerProc: 在独立进程中运行 worker 的包装器
- FutureWrapper: Future 包装器
- WorkerProcHandle: Worker 进程句柄
- UnreadyWorkerProcHandle: 未就绪 Worker 进程句柄

主要函数：
- set_multiprocessing_worker_envs: 设置多进程环境变量

MultiprocExecutor 适用于单机多卡场景，使用共享内存进行高效通信。
"""

import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class FutureWrapper(Future):
    """Future 包装器，用于处理异步 RPC 调用。

    该类包装了 Future 以支持队列中的顺序处理。

    Attributes:
        futures_queue: Future 队列
        aggregate: 聚合函数
    """

    def __init__(
        self,
        futures_queue: deque[tuple["FutureWrapper", Callable]],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.aggregate = aggregate
        super().__init__()

    def result(self, timeout=None):
        """获取结果。

        Args:
            timeout: 超时时间（未实现）

        Returns:
            结果

        Raises:
            RuntimeError: 如果指定了 timeout
        """
        if timeout is not None:
            raise RuntimeError("timeout not implemented")
        # 清空队列中在我们前面的任何 future
        while not self.done():
            future, get_response = self.futures_queue.pop()
            future.wait_for_response(get_response)
        return super().result()

    def wait_for_response(self, get_response: Callable):
        """等待响应。

        Args:
            get_response: 获取响应的函数
        """
        try:
            response = self.aggregate(get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    """多进程执行器。

    使用 multiprocessing 创建多个 worker 进程，通过共享内存消息队列进行通信。
    适用于单机多卡场景。

    主要功能：
    - 创建和管理 worker 进程
    - 通过共享内存消息队列广播调度器输出
    - 收集 worker 输出
    - 监控 worker 健康状态
    - 支持流水线并行

    Attributes:
        supports_pp: True（支持流水线并行）
        monitor_workers: 是否监控 worker
        _finalizer: 析构函数
        is_failed: 是否失败
        failure_callback: 失败回调
        workers: worker 进程句柄列表
        rpc_broadcast_mq: RPC 广播消息队列
        response_mqs: 响应消息队列列表
        futures_queue: Future 队列
        output_rank: 输出 rank
    """

    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        """初始化多进程执行器。

        Args:
            vllm_config: vLLM 配置
            monitor_workers: 是否监控 worker 进程
        """
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        """初始化执行器。"""
        # 在退出时调用 self.shutdown 进行清理
        # 并确保 workers 将被终止
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # 设置多进程环境变量
        set_multiprocessing_worker_envs()

        # 使用 loopback 地址进行通信
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # 初始化 worker 并设置消息队列用于 SchedulerOutputs 和 ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # 对于每个 dp rank 内的 leader node，
            # 每个 dp 将有自己的 leader multiproc executor
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # 创建 workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # 使用 fork 时，跟踪 worker 继承的 socket 文件描述符，
            # 以便我们可以在后续 workers 中关闭它们
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers 必须在 wait_for_ready 之前创建以避免死锁，
            # 因为 worker.init_device() 执行设备同步

            # 等待所有本地 workers 就绪
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # 如果不在 headless 模式下，启动后台线程监控 worker 健康状态
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # 只有 leader node 有远程响应 mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # 确保消息队列就绪，如果重新排序将死锁
            # 必须与 WorkerProc 保持一致

            # 等待所有输入 mqs 就绪
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # 等待所有远程响应 mqs 就绪
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # 如果失败，清理 worker 进程
                # 首先关闭 death_writers 以信号 workers 退出
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        """获取并行大小。

        Returns:
            三元组：(tp_size, pp_size, pcp_size)
        """
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        """执行器后初始化。"""
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        """是否是 driver worker。

        Args:
            rank: rank

        Returns:
            是否是 driver worker
        """
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        """启动 worker 监控。

        Args:
            inline: 是否内联运行
        """
        workers = self.workers
        self_ref = weakref.ref(self)

        # 监控 worker 进程活性，如果任何进程意外死亡，
        # 记录错误，关闭执行器并调用失败回调通知引擎
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, shutting down executor.", proc_name
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        """注册失败回调。

        Args:
            callback: 失败回调函数
        """
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

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
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
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
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self) -> None:
        """执行虚拟批次。"""
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """获取草稿 token ID。

        Returns:
            草稿 token ID 或 None
        """
        # 优化：仅从单个 worker（output_rank）获取输出
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """执行集体 RPC 调用。

        如果提供了 unique_reply_rank 和/或 kv_output_aggregator 则返回单个结果，
        否则返回列表。

        Args:
            method: 方法名或 callable
            timeout: 超时时间
            args: 位置参数
            kwargs: 关键字参数
            non_block: 是否非阻塞
            unique_reply_rank: 唯一回复 rank
            kv_output_aggregator: KV 输出聚合器

        Returns:
            RPC 结果或 Future

        Raises:
            AssertionError: 如果 rpc_broadcast_mq 为 None
            RuntimeError: 如果执行器失败
            TimeoutError: 如果 RPC 超时
        """
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        if kv_output_aggregator is not None:
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        if non_block:
            future = FutureWrapper(self.futures_queue, aggregate=aggregate)
            self.futures_queue.appendleft((future, get_response))
            return future

        # 首先清空队列中任何挂起的 futures
        while self.futures_queue:
            future, get_fut_response = self.futures_queue.pop()
            future.wait_for_response(get_fut_response)

        return aggregate(get_response())

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """确保所有 worker 进程被终止。

        假设 workers 已收到终止请求。等待处理，然后在需要时发送
        终止和 kill 信号。

        Args:
            worker_procs: worker 进程列表
        """

        def wait_for_termination(procs, timeout):
            if not time:
                # 如果在关闭的后期阶段，解释器可能将 `time` 替换为 `None`
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        # 首先给进程时间正确清理自己
        logger.debug("Worker Termination: allow workers to gracefully shutdown")
        if wait_for_termination(active_procs(), 4):
            return

        # 如果仍在运行，发送 SIGTERM
        logger.debug("Worker Termination: workers still running sending SIGTERM")
        for p in active_procs():
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # 如果仍在运行，发送 SIGKILL
            logger.debug(
                "Worker Termination: resorting to SIGKILL to take down workers"
            )
            for p in active_procs():
                p.kill()

    def shutdown(self):
        """正确关闭执行器及其 workers。"""
        if not getattr(self, "shutting_down", False):
            logger.debug("Triggering shutdown of workers")
            self.shutting_down = True

            # 确保首先终止所有 worker 进程
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # 关闭 death_writer 以信号子进程退出
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # 关闭响应队列
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

    def check_health(self) -> None:
        """检查健康状态。"""
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        """最大并发批次数量。

        PP 需要 PP-size 个并发批次来填充流水线。

        Returns:
            最大并发批次数量
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def _get_output_rank(self) -> int:
        """获取输出 rank。

        仅从 TP rank=0 和 PP rank=-1 返回 ModelRunnerOutput
        （最后一个 PP stage 的第一个 TP worker）。

        示例：
        假设 TP=8, PP=4, 则 world_size=32
        0-7, PP rank 0
        8-15, PP rank 1
        16-23, PP rank 2
        24-31, PP rank 3
        所以 world_size - tp_size = 32 - 8 = 24 应该是 PP rank = -1（即 3）

        Returns:
            输出 rank
        """
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """是否支持异步调度。

        Returns:
            True（支持异步调度）
        """
        return True


@dataclass
class UnreadyWorkerProcHandle:
    """READY 之前的 WorkerProcess 句柄。

    Attributes:
        proc: 进程
        rank: rank
        ready_pipe: 就绪管道
        death_writer: 死亡管道写入端
    """

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    """Worker 进程句柄。

    Attributes:
        proc: 进程
        rank: rank
        worker_response_mq: worker 响应消息队列（单节点模式）
        peer_worker_response_mqs: 对等 worker 响应消息队列列表
        death_writer: 死亡管道写入端
    """

    proc: BaseProcess
    rank: int
    # 在单节点模式下，worker 进程写入此 MQ
    worker_response_mq: MessageQueue | None
    # 这仅在 driver node 上非空，
    # 对等 worker 进程 i 写入 MQ `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        """从未就绪句柄创建句柄。

        Args:
            unready_handle: 未就绪句柄
            worker_response_mq: worker 响应消息队列
            peer_worker_response_mqs: 对等 worker 响应消息队列

        Returns:
            WorkerProcHandle
        """
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """在独立进程中运行一个 Worker 的包装器。

    主要功能：
    - 在独立进程中初始化 worker
    - 通过消息队列接收和发送数据
    - 监控死亡管道以检测父进程退出
    - 执行忙循环处理 RPC 调用

    Attributes:
        READY_STR: 就绪字符串
        rpc_broadcast_mq: RPC 广播消息队列
        worker_response_mq: worker 响应消息队列
        worker: worker 包装器
        rank: rank
        use_async_scheduling: 是否使用异步调度
        async_output_queue: 异步输出队列
        async_output_copy_thread: 异步输出复制线程
    """

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        """初始化消息队列。

        Args:
            input_shm_handle: 输入共享内存句柄
            vllm_config: vLLM 配置
        """
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # 初始化 MessageQueue 用于接收 SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # 初始化消息队列用于发送模型输出
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # 初始化远程 MessageQueue 用于跨节点接收 SchedulerOutput
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # 由于 executor proc 有 external_writer_handle，
                # 来自实际 writer 的就绪信号在 create_mq_broadcaster 方法之外发送
                # 并且在此设置之后，我们使其非阻塞。握手将在
                # worker.rpc_broadcast_mq.wait_until_ready() 被调用时触发
                blocking=False,
            )
            # 初始化远程消息队列用于将模型输出发送到 driver worker，
            # 为 driver worker 公开 peer_response_handles，包括所有 ranks 的句柄
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        """初始化 WorkerProc。

        Args:
            vllm_config: vLLM 配置
            local_rank: 本地 rank
            rank: 全局 rank
            distributed_init_method: 分布式初始化方法
            input_shm_handle: 输入共享内存句柄
            shared_worker_lock: 共享 worker 锁
            is_driver_worker: 是否是 driver worker
        """
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # TODO: 将 `init_worker` 移动到执行器级别作为集体 rpc 调用
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # 加载模型
        self.worker.init_device()
        # 现在并行组已初始化，更新进程标题
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # 根据注意力后端设置块大小
        current_platform.update_block_size_for_backend(vllm_config)

        # 在 init_device() 之后初始化消息队列，因为多节点设置
        # (nnodes_within_dp > 1) 需要初始化分布式组
        self._init_message_queues(input_shm_handle, vllm_config)

        # 启用环境变量缓存（例如，假设此后不再有
        # 环境变量覆盖）
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        """创建 worker 进程。

        Args:
            vllm_config: vLLM 配置
            local_rank: 本地 rank
            rank: 全局 rank
            distributed_init_method: 分布式初始化方法
            input_shm_handle: 输入共享内存句柄
            shared_worker_lock: 共享 worker 锁
            is_driver_worker: 是否是 driver worker
            inherited_fds: 继承的文件描述符

        Returns:
            UnreadyWorkerProcHandle
        """
        context = get_mp_context()
        # 就绪管道用于从子进程向父进程通信就绪状态
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # 死亡管道用于让子进程检测父进程退出
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # 让 worker 也关闭此 worker 管道的父端
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # 在后台进程中运行 EngineCore 忙循环
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        # 在这里关闭父进程中的子端管道
        ready_writer.close()
        death_reader.close()
        # 在父进程中保持 death_writer 打开 - 当父进程退出时，
        # 子进程中的 death_reader 将获得 EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        """等待响应句柄就绪。

        Args:
            handles: 句柄字典
            proc_handle: 进程句柄

        Returns:
            WorkerProcHandle
        """
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        """等待 workers 就绪。

        Args:
            unready_proc_handles: 未就绪进程句柄列表

        Returns:
            WorkerProcHandle 列表

        Raises:
            Exception: WorkerProc 初始化失败
        """
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # 等待 WorkerProc 就绪
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # 关闭连接
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        """关闭 worker。"""
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        """监控死亡管道。

        Args:
            death_pipe: 死亡管道
            shutdown_requested: 关闭请求事件
        """
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                # 这将阻塞直到父进程退出（管道关闭）
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # 直接传递队列引用以避免传递 self 时的 gc 问题
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker 初始化和执行循环。

        这在后台进程中运行。

        信号处理程序用于优雅终止。
        SystemExit 异常仅抛出一次以允许此和 worker
        进程无错误终止。
        """
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # SIGTERM 或 SIGINT 将终止 worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # 关闭从父进程继承的管道（包括其他 worker 管道）
        # 显式传递现有管道并关闭它们使管道在使用 fork 时行为正常
        # 否则，管道的隐藏引用存在于子进程中并阻止 EOF 关闭
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            # 初始化 tracer
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # 一旦我们知道所有内容已加载，发送 READY
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # 确保消息队列就绪，如果重新排序将死锁
            # 必须与 Executor 保持一致
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # 注意：如果 Exception 在 busy_loop 中引发，我们通过 MQ RPC 发送
            # FAILURE 消息通知 Executor，触发系统关闭
            # TODO(rob): 处理 MQ 本身损坏的情况

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # 如果任何 worker 死亡，父进程向所有 worker 进程发送 SIGTERM
            # 设置此值以便我们不会重新抛出 SystemExit() 以避免 __del__ 中的 zmq 异常
            shutdown_requested.set()

        except SystemExit as e:
            # SystemExit 在 SIGTERM 或 SIGKILL 时抛出，通常表示
            # 优雅关闭过程未成功
            logger.warning("WorkerProc was terminated")
            # SystemExit 绝不能被忽略
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # 一旦 worker 退出忙循环，清理
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        """响应状态枚举。"""
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        """准备 worker 输出并将其入队到 worker_response_mq。

        如果输出是 Exception，则转换为 FAILURE 响应。

        Args:
            output: 输出
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """处理 worker 输出。

        如果启用了异步调度，则传递给 async_output_busy_loop 线程。
        否则，直接入队到 worker_response_mq。

        Args:
            output: 输出
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """异步处理输出的线程入口点。

        设置线程的设备为 worker 设备。
        线程不会继承主线程的上下文。
        调用任何 cuda 运行时函数时，它将隐式
        在设备 0 上创建新的 cuda 上下文，消耗额外内存。
        这里我们为线程设置设备为 worker 设备，
        强制上下文与主线程相同。
        """
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self):
        """多进程 Worker 的主忙循环。"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)
            except Exception as e:
                # Notes 在 python 3.11 中引入
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # 异常可能不可序列化，所以我们将其转换为字符串，仅用于日志记录
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        """设置进程标题和日志前缀。

        Args:
            enable_ep: 是否启用 expert parallel
        """
        # 首先检查并行组是否已初始化
        if not model_parallel_is_initialized():
            # 并行组尚未初始化，使用默认进程名
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """设置在多进程环境中有 workers 时应使用的环境变量。

    这应由父进程在创建 worker 进程之前调用。

    主要功能：
    - 强制使用 spawn 启动方法
    - 配置线程并行度以避免 CPU 争用
    """
    _maybe_force_spawn()

    # 如果未设置 OMP_NUM_THREADS，配置线程并行度
    #
    # 有助于避免 CPU 争用。每个核心生成一个线程的默认值
    # 与每个 GPU 的多进程结合可能对性能产生负面影响
    # 当在容器中运行时，争用被放大，其中 CPU 限制可能导致节流
    default_omp_num_threads = 1
    if (
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning_once(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
            scope="local",
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
