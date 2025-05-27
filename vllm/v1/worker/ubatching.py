# SPDX-License-Identifier: Apache-2.0
import threading
import torch
import torch._dynamo
import torch.profiler as profiler
import os
from typing import Optional
from torch.library import Library
from torch.library import custom_op, register_kernel

from vllm.utils import current_stream
from vllm import forward_context

class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """
    def __init__(self,
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
        self.forward_context = None #fwd_ctx
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule

    def __enter__(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = self
        self._restore_context()
        # Assume we start on the compute stream
        assert current_stream() == self.compute_stream, \
            "Expected to start on the compute stream, but found %s" % current_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        torch.cuda.set_stream(self.original_stream)
        print("Finishing ubatch %d" % self.id)
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)
    
    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self, gpu_wait: bool = True):
        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def yield_and_switch_from_compute_to_comm(self):
        self._signal_compute_done()
        self._cpu_yield()
        torch.cuda.set_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        self._signal_comm_done()
        self._cpu_yield()
        torch.cuda.set_stream(self.compute_stream)
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
    if ctx is not None:
        ctx.yield_and_switch_from_compute_to_comm()

def yield_and_switch_from_comm_to_compute_impl(schedule="default"):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context()
    if ctx is not None:
        ctx.yield_and_switch_from_comm_to_compute()

# 2) Register kernel for CUDA, mark as mutating to prevent the compiler from
#    optimizing it away (TODO: see if this is actually needed)
@custom_op("vllm::yield_and_switch_from_compute_to_comm", mutates_args=("x",))
def yield_and_switch_from_compute_to_comm(x: torch.Tensor, schedule: str="default") -> None:
    yield_and_switch_from_compute_to_comm_impl(schedule)

# 3) Fake implementation for shape prop and FX tracing
@yield_and_switch_from_compute_to_comm.register_fake
def yield_and_switch_from_compute_to_comm(x: torch.Tensor, schedule: str="default") -> None:
    pass

@custom_op("vllm::yield_and_switch_from_comm_to_compute", mutates_args=("x",))
def yield_and_switch_from_comm_to_compute(x: torch.Tensor, schedule: str="default") -> None:
    yield_and_switch_from_comm_to_compute_impl(schedule)

@yield_and_switch_from_comm_to_compute.register_fake
def yield_and_switch_from_comm_to_compute(x: torch.Tensor, schedule: str="default") -> None:
    pass

def dump_ubatching_state():
    pass
    # """
    # Dump the current UBatchContext state for debugging.
    # """
    
    # dp_rank = os.getenv("VLLM_DP_RANK", None)
    
    # for ctx in _CURRENT_CONTEXT.values():
    #     print(f"UBatchContext: {ctx.id} (dp_rank {dp_rank})\n"
    #           f" Stream: {ctx.stream}, ({ctx.stream.query()})\n"
    #           f" Original Stream: {ctx.original_stream}, ({ctx.original_stream.query()})\n"
    #           f" CPU Wait Event: {ctx.cpu_wait_event}\n"
    #           f" GPU Wait Event: {ctx.gpu_wait_event}  ({ctx.gpu_wait_event.query()})\n"
    #           f" CPU Signal Event: {ctx.cpu_signal_event}\n"
    #           f" GPU Signal Event: {ctx.gpu_signal_event} ({ctx.gpu_signal_event.query()})\n")

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
                            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
                            gpu_comm_done_event=gpu_comm_done_events[i],
                            gpu_compute_done_event=gpu_compute_done_events[i],
                            schedule=schedule
                )
        ctxs.append(ctx)

    return ctxs