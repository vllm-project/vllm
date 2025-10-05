# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from enum import Enum
from functools import lru_cache
from typing import Callable, Optional

import torch
from torch.library import Library

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.utils import current_stream

_THREAD_ID_TO_CONTEXT: dict[int, int] = {}
_CURRENT_CONTEXTS: list[Optional['UBatchContext']] = [None, None]
Schedules = tuple['Schedule', ...]


class Schedule(Enum):
    # Schedule notation legend:
    #    S = Shared expert
    #    A0 = MLA qkv pro,
    #    A1 = Core attn + out proj + MoE gate
    #    D = Dispatch
    #    C = Combine

    # Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
    # Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
    # Order: D₁ send, A0₀, A1₀, D₁ recv, D₀ send, MLP₁, D₀ recv,
    #        C₁ send, S₁, MLP₀, C₁ recv, C₀ send, S₀, A0₁, A1₁, C₀ recv.
    MLP_SHARED_OVERLAP = "mlp_shared_overlap"

    # Comp: |-S₀-A0₁-|-MLP₀-|-A1₁-||-S₁-A0₀-|-MLP₁-|-A1₀-|
    # Comm: |---D₀---|      |-C₀--||---D₁---|      |-C₁--|
    # Order: D₀ send, S₀, A0₁, D₀ recv, MLP₀, C₀ send, A1₁, C₀ recv,
    #        D₁ send, S₁, A0₀, D₁ recv, MLP₁, C₁ send, A1₀, C₁ recv.
    ATTN_SHARED_OVERLAP = "attn_shared_overlap"


_SCHEDULE_WAIT_STAGES = {  # Default is 1
    Schedule.ATTN_SHARED_OVERLAP: 2,
}


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
                 started: bool = True,
                 schedule: Schedule = Schedule.MLP_SHARED_OVERLAP):
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
        self.started = started
        self.recv_hook: Optional[Callable[[], None]] = None

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        if self.id > 0:
            wait_stages = _SCHEDULE_WAIT_STAGES.get(self.schedule, 1)
            for _ in range(wait_stages - 1):
                self._cpu_yield(check_context=False)

        self._restore_context()
        # Assume we start on the compute stream
        assert current_stream() == self.compute_stream
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        if self.id == 0:
            # Keep advancing the next micro-batch
            wait_stages = _SCHEDULE_WAIT_STAGES.get(self.schedule, 1)
            for _ in range(wait_stages - 1):
                self.yield_()
                # Cleanup and trailing recv hooks
                self.maybe_run_recv_hook()

        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()

        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        self.current_stream = stream
        if current_stream() != self.current_stream:
            torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self, check_context: bool = True):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        if check_context:
            assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def switch_to_comm(self):
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self):
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        self.current_stream = current_stream()
        self._cpu_yield()
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


def dbo_start():
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        assert ctx is not None
        ctx.started = True


def dbo_current_schedule() -> Optional[Schedule]:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return None
    ctx = _CURRENT_CONTEXTS[dbo_current_ubatch_id()]
    assert ctx is not None
    return ctx.schedule


def _register_ubatch_function(func: Callable[['UBatchContext'], None],
                              all_schedules_default: bool = False):

    def wrapper(schedules: Schedules = (),
                all_schedules: bool = all_schedules_default) -> None:
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            assert ctx is not None
            if not ctx.started:
                return
            if all_schedules or ctx.schedule in schedules:
                func(ctx)

    return wrapper


dbo_maybe_run_recv_hook = _register_ubatch_function(
    UBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(
    UBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(
    UBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync)

# DBO start needs to be callable from inside the torch compile region so
# we register it as a custom op.
lib = Library("vllm_dbo", "DEF")
lib.define("start(Tensor! x) -> ()")  # in-place, returns x


@torch.library.impl("vllm_dbo::start", "CompositeImplicitAutograd")
def _dbo_start_impl(x: torch.Tensor):
    dbo_start()
    return None


@lru_cache(maxsize=1)
def dbo_debug_annotate():
    return True


def dbo_register_recv_hook(recv_hook: Callable[[], None],
                           schedules: Schedules = (),
                           all_schedules: bool = False) -> bool:
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        assert ctx is not None
        if all_schedules or ctx.schedule in schedules:
            next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % 2]
            # Next context may have already exited
            if next_ctx is not None:
                next_ctx.recv_hook = recv_hook
                return True
    return False


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: Schedule = Schedule.MLP_SHARED_OVERLAP,
    delayed_start: bool = False,
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
                            started=not delayed_start,
                            schedule=schedule)
        ctxs.append(ctx)

    return ctxs
