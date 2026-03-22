# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""微批次（Ubatching）上下文管理模块。

本模块实现了用于微批次同步的上下文管理器，负责：
- 管理 DBO（Dual Batch Overlap）执行的线程同步
- 提供通信流和计算流的切换机制
- 使用 threading.Barrier 和 Event 进行 CPU-GPU 同步
- 支持多个微批次上下文的并发管理

主要类：
- UBatchContext: 微批次上下文管理器

主要函数：
- make_ubatch_contexts: 创建多个微批次上下文
- dbo_enabled: 检查 DBO 是否启用
- dbo_current_ubatch_id: 获取当前微批次 ID
"""

import threading

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# 线程 ID 到上下文 ID 的映射
_THREAD_ID_TO_CONTEXT: dict = {}
# 微批次数量，默认为 2
_NUM_UBATCHES: int = 2
# 当前上下文列表
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


class UBatchContext:
    """使用 threading event 进行微批次同步的上下文管理器。

    此类管理 DBO 执行中的同步，提供：
    - 通信流和计算流的分离
    - CPU 和 GPU 事件的协调
    - 线程安全的上下文切换

    Attributes:
        id: 微批次 ID
        comm_stream: 通信 CUDA 流
        compute_stream: 计算 CUDA 流
        forward_context: 前向传播上下文
        ready_barrier: 就绪屏障
        cpu_wait_event: CPU 等待事件
        cpu_signal_event: CPU 信号事件
        current_stream: 当前 CUDA 流
        gpu_comm_done_event: GPU 通信完成事件
        gpu_compute_done_event: GPU 计算完成事件
        schedule: 调度策略
        recv_hook: 接收钩子函数
    """

    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
    ):
        """初始化微批次上下文。

        Args:
            id: 微批次 ID
            comm_stream: 通信 CUDA 流
            compute_stream: 计算 CUDA 流
            forward_context: 前向传播上下文
            ready_barrier: 就绪屏障
            cpu_wait_event: CPU 等待事件
            cpu_signal_event: CPU 信号事件
            gpu_comm_done_event: GPU 通信完成事件
            gpu_compute_done_event: GPU 计算完成事件
            schedule: 调度策略
        """
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule
        self.recv_hook = None

    def __enter__(self):
        """进入上下文。"""
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        # _NUM_UBATCHES 在 make_ubatch_contexts 中设置
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # 假设我们要从计算流开始
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文。"""
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def _restore_context(self):
        """恢复上下文。"""
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        """更新当前流。

        Args:
            stream: 新的 CUDA 流
        """
        self.current_stream = stream
        if current_stream() != self.current_stream:
            torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        """信号：通信完成。"""
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        """信号：计算完成。"""
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        """等待：计算完成。"""
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        """等待：通信完成。"""
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self):
        """CPU 让出执行权。

        重要的是只有一个线程在运行。这些断言确保
        在唤醒另一个线程并进入睡眠之前，这是唯一运行的线程。
        """
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def switch_to_comm(self):
        """切换到通信流。"""
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        """切换到计算流。"""
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self):
        """切换到通信流（同步）。"""
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        """切换到计算流（同步）。"""
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        """可能运行接收钩子。"""
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        """让出执行权。"""
        self.current_stream = current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        """让出并从计算流切换到通信流。"""
        assert current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        """让出并从通信流切换到计算流。"""
        assert current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()


def dbo_enabled() -> bool:
    """检查 DBO 是否启用。

    Returns:
        DBO 是否启用
    """
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    """获取当前微批次 ID。

    Returns:
        当前微批次 ID，如果 DBO 未启用则返回 0
    """
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):
    """注册微批次函数装饰器。

    此装饰器将函数注册为微批次感知函数，
    在 DBO 启用时自动传入当前上下文。

    Args:
        func: 要注册的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


# 注册的微批次函数
dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm
)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute
)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(UBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync
)


def dbo_register_recv_hook(recv_hook):
    """注册接收钩子。

    Args:
        recv_hook: 接收钩子函数
    """
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    """在前一个微批次的事件上执行函数。

    Args:
        func: 要执行的函数
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        函数返回值
    """
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # 在 ubatch 计算流上执行可调用对象以记录/等待事件
        with torch.cuda.stream(ctx.compute_stream):
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
    """创建微批次同步上下文管理器。

    Args:
        num_micro_batches: 微批次数量
        compute_stream: 计算 CUDA 流
        comm_stream: 通信 CUDA 流
        forward_contexts: 前向传播上下文列表
        ready_barrier: 就绪屏障
        schedule: 调度策略

    Returns:
        UBatchContext 列表

    Raises:
        AssertionError: 如果 num_micro_batches 不大于 1
    """
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1, "num_micro_batches 必须大于 1"

    _NUM_UBATCHES = num_micro_batches
    # 确保全局上下文列表足够大
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    # 创建 CPU 和 GPU 事件
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)

    return ctxs
