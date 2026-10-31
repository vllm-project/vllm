# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import torch

from vllm import forward_context
from vllm.distributed import parallel_state
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
_NUM_UBATCHES: int = 2
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


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

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        # _NUM_UBATCHES is set in make_ubatch_contexts
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # Assume we want to start on the compute stream
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
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
    # Ensure the global context list is large enough
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    """
    Create a context manager for micro-batching synchronization.
    """
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
            token_offset=token_offsets[i] if token_offsets else 0,
        )
        ctxs.append(ctx)

    return ctxs


# DBO under pure TP: each TP all-reduce is a hand-off point and runs on the comm
# stream, on a per-microbatch communicator, while the other microbatch computes.
_TP_UBATCH_COMMS: list | None = None


def init_dbo_tp_comms(parallel_config) -> None:
    """Build the per-microbatch TP communicators and install the all-reduce hook
    (DBO under pure TP only). Collective: every rank calls it in the same order."""
    global _TP_UBATCH_COMMS
    if not parallel_config.dbo_tp_only or _TP_UBATCH_COMMS is not None:
        return
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    tp = parallel_state.get_tp_group()
    comms = [
        PyNcclCommunicator(group=tp.cpu_group, device=tp.device)
        for _ in range(parallel_config.num_ubatches)
    ]
    if any(c.disabled for c in comms):
        logger.warning(
            "DBO: the NCCL library is unavailable, the TP all-reduces of a "
            "microbatch stay on the compute stream (no overlap)"
        )
        return
    _TP_UBATCH_COMMS = comms
    parallel_state.set_dbo_tp_all_reduce_hook(_dbo_tp_all_reduce)
    logger.info(
        "DBO: %d per-microbatch TP communicators for the comm-stream all-reduce",
        len(comms),
    )


def dbo_comm_stream_priority() -> int:
    """High on ROCm: at the default priority comm and compute can share one
    hardware queue and serialize."""
    from vllm.platforms import current_platform

    return -1 if current_platform.is_rocm() else 0


def dbo_tp_collective(issue) -> bool:
    """Run ``issue(comm, stream)`` as a microbatch hand-off on its comm stream;
    False outside a microbatch."""
    if len(_THREAD_ID_TO_CONTEXT) == 0 or _TP_UBATCH_COMMS is None:
        return False
    ctx_idx = _THREAD_ID_TO_CONTEXT.get(threading.get_ident())
    if ctx_idx is None:
        return False
    ctx = _CURRENT_CONTEXTS[ctx_idx]
    ctx.yield_and_switch_from_compute_to_comm()
    issue(_TP_UBATCH_COMMS[ctx_idx], ctx.comm_stream)
    ctx.switch_to_compute_sync()
    return True


def _dbo_tp_all_reduce(group, input_: torch.Tensor) -> torch.Tensor | None:
    """The TP all-reduce of a microbatch, on the comm stream."""
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return None
    ctx_idx = _THREAD_ID_TO_CONTEXT.get(threading.get_ident())
    if ctx_idx is None or group is not parallel_state._TP:
        return None
    assert _TP_UBATCH_COMMS is not None
    ctx = _CURRENT_CONTEXTS[ctx_idx]
    logger.info_once("DBO: TP all-reduces of a microbatch go to the comm stream")
    # No record_stream: the stream waits already order the buffers, and it would
    # pin every block until the GPU catches up (out of memory at a 32k step).
    out = torch.empty_like(input_)
    ctx.yield_and_switch_from_compute_to_comm()
    _TP_UBATCH_COMMS[ctx_idx].all_reduce(input_, out_tensor=out, stream=ctx.comm_stream)
    ctx.switch_to_compute_sync()
    return out
