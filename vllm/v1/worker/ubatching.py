# SPDX-License-Identifier: Apache-2.0
import threading
from typing import Optional

import torch
import torch._dynamo
from torch.library import custom_op

from vllm import forward_context
from vllm.utils import current_stream
from vllm.distributed.parallel_state import get_dp_group


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(
            self,
            id: int,
            comm_stream: torch.cuda.Stream,
            compute_stream: torch.cuda.Stream,
            #fwd_ctx: forward_context.ForwardContext,
            cpu_wait_event: threading.Event,
            cpu_signal_event: threading.Event,
            gpu_comm_done_event: torch.cuda.Event,
            gpu_compute_done_event: torch.cuda.Event,
            schedule: str = "default"):
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.original_stream = current_stream()
        self.forward_context = None  #fwd_ctx
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule

    def __enter__(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = self

        self.cpu_wait_event.clear()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # Assume we start on the compute stream
        assert current_stream() == self.compute_stream
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        # print("Finishing ubatch %d\n" % self.id, flush=True)
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        self.current_stream = self.compute_stream
        torch.cuda.set_stream(self.original_stream)
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context
        torch.cuda.set_stream(self.current_stream)

    def update_stream(self, stream):
        self.current_stream = stream
        torch.cuda.set_stream(self.current_stream)

    def ctx_valid_state(self):
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()
        pass

    def _signal_comm_done(self):
        self.ctx_valid_state()
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.ctx_valid_state()
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        # print(f"{self.id} Waiting on COMPUTE stream", flush=True)
        self.ctx_valid_state()
        self.comm_stream.wait_event(self.gpu_compute_done_event)
        # print("Compute stream done", flush=True)

    def _wait_comm_done(self):
        # print(f"{self.id} Waiting on COMM stream", flush=True)
        self.ctx_valid_state()
        self.compute_stream.wait_event(self.gpu_comm_done_event)
        # print("Comm stream done", flush=True)

    def stream_string(self):
        if current_stream() == self.compute_stream:
            assert self.current_stream == self.compute_stream
            return "COMPUTE"
        elif current_stream() == self.comm_stream:
            assert self.current_stream == self.comm_stream
            return "COMM"

    def _cpu_yield(self):
        # print(f"UBatchContext: {self.id} yielding CPU", flush=True)
        self.ctx_valid_state()
        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        self.ctx_valid_state()
        # print(f"UBatchContext: {self.id} resuming CPU", flush=True)

    def yield_and_switch_from_compute_to_comm(self):
        assert current_stream() == self.compute_stream
        # dp_rank = get_dp_group().rank_in_group
        # print(f"DP: {dp_rank} UB: {self.id} "
        #       f"Yield and switch from {self.stream_string()}", flush=True)
        self.ctx_valid_state()
        self._signal_compute_done()
        self._cpu_yield()
        self.ctx_valid_state()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        # print(f"DP: {dp_rank} UB: {self.id} "
        #       f"Resuming on stream {self.stream_string()}", flush=True)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        assert current_stream() == self.comm_stream
        # dp_rank = get_dp_group().rank_in_group
        # print(f"DP: {dp_rank} UB: {self.id} "
        #       f"Yield and switch from {self.stream_string()}", flush=True)
        self.ctx_valid_state()
        self._signal_comm_done()
        self._cpu_yield()
        self.ctx_valid_state()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        # print(f"DP: {dp_rank} UB: {self.id} "
        #       f"Resuming on stream {self.stream_string()}", flush=True)
        self._wait_comm_done()


_CURRENT_CONTEXT: dict = {}


def get_current_ubatch_context() -> Optional[UBatchContext]:
    global _CURRENT_CONTEXT
    """
    Get the current UBatchContext for the current thread.
    """
    return _CURRENT_CONTEXT.get(threading.get_ident(), None)


def yield_and_switch_from_compute_to_comm_impl(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context()
    #print("you are in yield_impl", ctx)
    if ctx is not None and ctx.schedule == schedule:
        ctx.yield_and_switch_from_compute_to_comm()


def yield_and_switch_from_comm_to_compute_impl(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context()
    if ctx is not None and ctx.schedule == schedule:
        ctx.yield_and_switch_from_comm_to_compute()


# 2) Register kernel for CUDA, mark as mutating to prevent the compiler from
#    optimizing it away (TODO: see if this is actually needed)
@custom_op("vllm::yield_and_switch_from_compute_to_comm", mutates_args=("x", ))
def yield_and_switch_from_compute_to_comm(x: torch.Tensor,
                                          schedule: str = "default") -> None:
    yield_and_switch_from_compute_to_comm_impl(schedule)


# 3) Fake implementation for shape prop and FX tracing
@yield_and_switch_from_compute_to_comm.register_fake
def yield_and_switch_from_compute_to_comm_fake(x: torch.Tensor,
                                               schedule: str = "default"
                                               ) -> None:
    pass


@custom_op("vllm::yield_and_switch_from_comm_to_compute", mutates_args=("x", ))
def yield_and_switch_from_comm_to_compute(x: torch.Tensor,
                                          schedule: str = "default") -> None:
    yield_and_switch_from_comm_to_compute_impl(schedule)


@yield_and_switch_from_comm_to_compute.register_fake
def yield_and_switch_from_comm_to_compute_fake(x: torch.Tensor,
                                               schedule: str = "default"
                                               ) -> None:
    pass


def dump_ubatching_state():
    pass


"""
"""


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    device: Optional[torch.device] = None,
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
    device = device or torch.cuda.current_device()
    comm_stream = torch.cuda.Stream(device)

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(id=i,
                            compute_stream=compute_stream,
                            comm_stream=comm_stream,
                            cpu_wait_event=cpu_events[i],
                            cpu_signal_event=cpu_events[(i + 1) %
                                                        num_micro_batches],
                            gpu_comm_done_event=gpu_comm_done_events[i],
                            gpu_compute_done_event=gpu_compute_done_events[i],
                            schedule=schedule)
        ctxs.append(ctx)

    return ctxs
