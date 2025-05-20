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
    def __init__(self, wait_event: threading.Event, signal_event: threading.Event):
        self.wait_event = wait_event
        self.signal_event = signal_event

    def __enter__(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        return False

    def yield_(self):
        # Signal that this batch reached the barrier and wait for the other
        self.signal_event.set()
        self.wait_event.wait()
        # Reset for reuse
        self.wait_event.clear()

_CURRENT_CONTEXT: dict = {}

def yield_impl():
    # Perform the barrier if a context exists for this thread
    ctx = _CURRENT_CONTEXT.get(threading.get_ident(), None)
    if ctx is not None:
        ctx.yield_()


# 2) Register kernel for CUDA
@custom_op("custom::yield_", mutates_args=("x",))
def yield_(x: torch.Tensor) -> None:
    yield_impl()

# 3) Fake implementation for shape prop and FX tracing
@yield_.register_fake
def yield_(x: torch.Tensor):
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
        ctx = UBatchContext(wait_event, signal_event)
        ctxs.append(ctx)

    # Create the context manager
    ctx = UBatchContext(wait_event, signal_event)

    return ctx