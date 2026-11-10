# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import torch

import vllm.envs as envs
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
# stream while the other microbatch computes.
# VLLM_DBO_TP_DMA_AR=1 (TP2 only, ROCm): a microbatch's all-reduce runs as a
# one-shot INT4 all-reduce over the SDMA copy engines, which needs no CUs while
# the other microbatch computes; QuickReduce and RCCL are CU kernels that contend
# with the GEMMs under DBO. The helper lives on the TP group's CudaCommunicator.
def enable_dbo_tp_overlap(
    parallel_config, comm_stream: torch.cuda.Stream, vllm_config=None
) -> None:
    """Collective: every rank calls it in the same order."""
    if not parallel_config.dbo_tp_only:
        return
    tp = parallel_state.get_tp_group()
    pynccl_comm = tp.device_communicator.pynccl_comm
    if pynccl_comm is not None:
        pynccl_comm.bind_stream(comm_stream)
    tp.dbo_overlap = True
    if envs.VLLM_DBO_TP_DMA_AR:
        _enable_dma_all_reduce(tp, vllm_config)


def _enable_dma_all_reduce(tp, vllm_config) -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        logger.warning("DBO: VLLM_DBO_TP_DMA_AR=1 is ROCm only; ignored")
        return
    if tp.world_size != 2:
        logger.warning(
            "DBO: VLLM_DBO_TP_DMA_AR=1 needs TP 2, got %d; ignored", tp.world_size
        )
        return
    if vllm_config is None:
        logger.warning("DBO: VLLM_DBO_TP_DMA_AR=1 needs the model config; ignored")
        return
    from vllm.distributed.device_communicators.dma_all_reduce import DmaAllReduce

    # Largest all-reduce: one full batch of hidden states, 2 bytes per value.
    max_elems = (
        vllm_config.scheduler_config.max_num_batched_tokens
        * vllm_config.model_config.get_hidden_size()
    )
    # Raises on a ROCm runtime without the needed HIP calls.
    tp.device_communicator.dma_ar = DmaAllReduce(
        tp.cpu_group, tp.device, max_bytes=max_elems * 2
    )
    logger.info(
        "DBO: microbatch all-reduces up to %d MiB run as one-shot INT4 over SDMA",
        max_elems * 2 >> 20,
    )


def dbo_comm_stream_priority() -> int:
    """High on ROCm: at the default priority comm and compute can share one
    hardware queue and serialize."""
    from vllm.platforms import current_platform

    return -1 if current_platform.is_rocm() else 0


def dbo_tp_all_reduce(
    device_communicator, input_: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """The TP all-reduce of a microbatch, issued on the comm stream.

    With VLLM_DBO_TP_DMA_AR set (TP2, ROCm) and ``input_`` small enough, it runs
    over the SDMA engines. Both microbatches share one comm stream and only one
    microbatch thread runs at a time, so the calls are issued in the same order
    on every rank and never overlap.
    """
    dma = getattr(device_communicator, "dma_ar", None)
    # Outside a microbatch (no split, warmup, decode) this is the regular TP
    # all-reduce: the SDMA path only pays off when it overlaps compute. Stream
    # write/wait operations cannot be captured into a graph either.
    if (
        dma is not None
        and threading.get_ident() in _THREAD_ID_TO_CONTEXT
        and dma.should_reduce(input_)
        and not torch.cuda.is_current_stream_capturing()
    ):
        logger.info_once("DBO: microbatch all-reduces run over SDMA (one-shot INT4)")
        return dma.all_reduce(input_, out=out)
    return device_communicator.all_reduce(input_)
