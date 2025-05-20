# SPDX-License-Identifier: Apache-2.0
import threading
import torch
import torch._dynamo
import torch.profiler as profiler
from torch.library import Library
from torch.library import custom_op, register_kernel

class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """
    def __init__(self,
                 stream: torch.cuda.Stream,
                 wait_event: threading.Event, 
                 signal_event: threading.Event,
                 schedule="default"):
        self.wait_event = wait_event
        self.signal_event = signal_event
        self.schedule = schedule
        self.stream = stream
        self.original_stream = torch.cuda.current_stream()

    def __enter__(self):
        global _CURRENT_CONTEXT
        self.original_stream = torch.cuda.current_stream()
        _CURRENT_CONTEXT[threading.get_ident()] = self
        # Set micro-batch stream
        torch.cuda.set_stream(self.stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        # Restore the original stream
        torch.cuda.set_stream(self.original_stream)
        return False

    def yield_(self):
        # Signal that this batch reached the barrier and wait for the other
        self.signal_event.set()
        # Wait for the next batch to signal back
        self.wait_event.wait()
        self.wait_event.clear()
        # When we resume switch back to the microbatch stream
        torch.cuda.set_stream(self.stream)

_CURRENT_CONTEXT: dict = {}

def yield_impl(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = _CURRENT_CONTEXT.get(threading.get_ident(), None)
    if ctx is not None and ctx.schedule == schedule:
        ctx.yield_()


# 2) Register kernel for CUDA
@custom_op("vllm::yield_", mutates_args=("x",))
def yield_(x: torch.Tensor, schedule="default") -> None:
    yield_impl(schedule)

# 3) Fake implementation for shape prop and FX tracing
@yield_.register_fake
def yield_(x: torch.Tensor, schedule="default") -> None:
    pass

"""

"""
def make_ubatch_context_chain(num_micro_batches: int) -> list[UBatchContext]:
    """
    Create a context manager for micro-batching synchronization.
    """
    events = [threading.Event() for _ in range(num_micro_batches)]
    
    ctxs = []
    for i in range(num_micro_batches):
        wait_event = events[i]
        signal_event = events[(i + 1) % num_micro_batches]
        ctx = UBatchContext(torch.Stream(), wait_event, signal_event)
        ctxs.append(ctx)

    return ctxs