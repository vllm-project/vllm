# SPDX-License-Identifier: Apache-2.0
import threading
from typing import Optional

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.utils import current_stream


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(
            self,
            id: int,
            comm_stream: torch.cuda.Stream,
            compute_stream: torch.cuda.Stream,
            forward_context: ForwardContext,
            cpu_wait_event: threading.Event,
            cpu_signal_event: threading.Event,
            gpu_comm_done_event: torch.cuda.Event,
            gpu_compute_done_event: torch.cuda.Event,
            schedule: str = "default"):
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
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

    def stream_string(self):
        if current_stream() == self.compute_stream:
            assert self.current_stream == self.compute_stream
            return "COMPUTE"
        elif current_stream() == self.comm_stream:
            assert self.current_stream == self.comm_stream
            return "COMM"

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


_CURRENT_CONTEXT: dict = {}


def get_current_ubatch_context() -> Optional[UBatchContext]:
    global _CURRENT_CONTEXT
    """
    Get the current UBatchContext for the current thread.
    """
    return _CURRENT_CONTEXT.get(threading.get_ident(), None)


def yield_and_switch_from_compute_to_comm(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context()
    if ctx is not None and ctx.schedule == schedule:
        ctx.yield_and_switch_from_compute_to_comm()


def yield_and_switch_from_comm_to_compute(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context()
    if ctx is not None and ctx.schedule == schedule:
        ctx.yield_and_switch_from_comm_to_compute()

def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
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

    assert len(forward_contexts) == 2

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(id=i,
                            compute_stream=compute_stream,
                            comm_stream=comm_stream,
                            forward_context=forward_contexts[i],
                            cpu_wait_event=cpu_events[i],
                            cpu_signal_event=cpu_events[(i + 1) %
                                                        num_micro_batches],
                            gpu_comm_done_event=gpu_comm_done_events[i],
                            gpu_compute_done_event=gpu_compute_done_events[i],
                            schedule=schedule)
        ctxs.append(ctx)

    return ctxs
