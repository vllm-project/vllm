# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import Optional

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.utils import current_stream

_THREAD_ID_TO_CONTEXT: dict = {}
_CURRENT_CONTEXTS: list[Optional['UBatchContext']] = [None, None]


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(self,
                 id: int,
                 comm_stream: torch.cuda.Stream,
                 compute_stream: torch.cuda.Stream,
                 forward_context: ForwardContext,
                 ready_barrier: threading.Barrier,
                 cpu_wait_event: threading.Event,
                 cpu_signal_event: threading.Event,
                 gpu_comm_done_event: torch.cuda.Event,
                 gpu_compute_done_event: torch.cuda.Event,
                 schedule: str = "default"):
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
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # Assume we start on the compute stream
        assert current_stream() == self.compute_stream
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        self.current_stream = self.compute_stream
        torch.cuda.set_stream(self.current_stream)
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context
        torch.cuda.set_stream(self.current_stream)

    def update_stream(self, stream):
        self.current_stream = stream
        torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def switch_to_comm_sync(self):
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        self.current_stream = current_stream()
        self._cpu_yield()
        if self.current_stream != current_stream():
            self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        assert current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        assert current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()


def dbo_enabled() -> bool:
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):

    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_maybe_run_recv_hook = _register_ubatch_function(
    UBatchContext.maybe_run_recv_hook)
dbo_switch_to_comm_sync = _register_ubatch_function(
    UBatchContext.switch_to_comm_sync)


def dbo_register_recv_hook(recv_hook):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % 2]
        next_ctx.recv_hook = recv_hook


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
    assert num_micro_batches == 2, "only been tested with 2 micro-batches"
    """
    Create a context manager for micro-batching synchronization.
    """
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [
        torch.cuda.Event() for _ in range(num_micro_batches)
    ]
    gpu_compute_done_events = [
        torch.cuda.Event() for _ in range(num_micro_batches)
    ]

    assert len(forward_contexts) == 2

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(id=i,
                            compute_stream=compute_stream,
                            comm_stream=comm_stream,
                            forward_context=forward_contexts[i],
                            ready_barrier=ready_barrier,
                            cpu_wait_event=cpu_events[i],
                            cpu_signal_event=cpu_events[(i + 1) %
                                                        num_micro_batches],
                            gpu_comm_done_event=gpu_comm_done_events[i],
                            gpu_compute_done_event=gpu_compute_done_events[i],
                            schedule=schedule)
        ctxs.append(ctx)

    return ctxs
