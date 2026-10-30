# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
_NUM_UBATCHES: int = 2
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


class UBatchAbortedError(RuntimeError):
    """A sibling failed before handing control back to this microbatch."""


class _UBatchAbortState:
    def __init__(
        self, cpu_events: list[threading.Event], ready_barrier: threading.Barrier
    ):
        self.cpu_events = cpu_events
        self.ready_barrier = ready_barrier
        self.lock = threading.Lock()
        self.failure: tuple[int, BaseException] | None = None

    def abort(self, ubatch_id: int, error: BaseException) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = (ubatch_id, error)
            for event in self.cpu_events:
                event.set()
            self.ready_barrier.abort()

    def check(self) -> None:
        failure = self.failure
        if failure is not None:
            ubatch_id, error = failure
            raise UBatchAbortedError(f"Microbatch {ubatch_id} failed") from error


class UBatchContext:
    """Context manager for micro-batching synchronization using threading events."""

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
        abort_state: _UBatchAbortState,
        schedule: str = "default",
        token_offset: int = 0,
    ):
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
        self.token_offset = token_offset
        self.recv_hook = None
        self.abort_state = abort_state

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        try:
            _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
            _CURRENT_CONTEXTS[self.id] = self
            self.ready_barrier.wait()
            self._wait_for_turn()
            self._restore_context()
            # Assume we want to start on the compute stream
            self.update_stream(self.compute_stream)
        except BaseException as error:
            # A failed __enter__ does not invoke __exit__ automatically.
            self.abort_state.abort(self.id, error)
            self._unregister()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.abort_state.abort(self.id, exc_val)
        self._unregister()
        try:
            self.maybe_run_recv_hook()
        except BaseException as error:
            self.abort_state.abort(self.id, error)
            raise
        finally:
            self.cpu_wait_event.clear()
            self.cpu_signal_event.set()
        return False

    def _unregister(self):
        _CURRENT_CONTEXTS[self.id] = None
        _THREAD_ID_TO_CONTEXT.pop(threading.get_ident(), None)

    def _wait_for_turn(self):
        self.abort_state.check()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self.abort_state.check()

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

    def _cpu_yield(self):
        self.abort_state.check()
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self._wait_for_turn()
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


def dbo_current_ubatch_token_offset() -> int:
    """Row offset of the calling ubatch inside the parent batch.

    Returns 0 when no DBO context is active on this thread.
    """
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    ctx = _CURRENT_CONTEXTS[_THREAD_ID_TO_CONTEXT[threading.get_ident()]]
    return ctx.token_offset


def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


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
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        with torch.cuda.stream(ctx.compute_stream):
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
    token_offsets: list[int] | None = None,
) -> list[UBatchContext]:
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    _NUM_UBATCHES = num_micro_batches
    if ready_barrier.broken:
        ready_barrier.reset()
    # Ensure the global context list is large enough
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    """
    Create a context manager for micro-batching synchronization.
    """
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    abort_state = _UBatchAbortState(cpu_events, ready_barrier)
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
            abort_state=abort_state,
            schedule=schedule,
            token_offset=token_offsets[i] if token_offsets else 0,
        )
        ctxs.append(ctx)

    return ctxs
