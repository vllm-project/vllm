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
                 stream: torch.cuda.Stream,
                 #fwd_ctx: forward_context.ForwardContext,
                 cpu_wait_event: threading.Event, 
                 cpu_signal_event: threading.Event,
                 gpu_wait_event: torch.cuda.Event,
                 gpu_signal_event: torch.cuda.Event,
                 gpu_wait_on_launch: bool = False,
                 schedule="default"):
        self.id = id
        self.stream = stream
        self.original_stream = current_stream()
        self.forward_context = None #fwd_ctx
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.gpu_wait_event = gpu_wait_event
        self.gpu_signal_event = gpu_signal_event
        self.schedule = schedule
        self.done_event = torch.cuda.Event()
        self.gpu_wait_on_launch = gpu_wait_on_launch

    def __enter__(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = self
        self._cpu_wait()
        # start_event = torch.cuda.Event()
        # self.original_stream.record_event(start_event)
        # self.stream.wait_event(start_event)
        print("Starting ubatch %d" % self.id)
        # if self.gpu_wait_on_launch:
        self.gpu_stream_wait()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        torch.cuda.set_stream(self.original_stream)
        print("Finishing ubatch %d" % self.id)
        self._signal()
        self._signal()
        self._signal()
        return False

    def _restore_context(self):
        # When we resume i.e. switch back to this micro-batch, we make sure
        # we have the correct stream and forward context
        torch.cuda.set_stream(self.stream)
        forward_context._forward_context = self.forward_context

    # Seperate GPU wait so we can do
    #  ubatch0 
    #   1) work
    #   2) dispatch
    #   3) yield
    #  ubatch1
    #   1) work
    #   2) gpu wait 
    #   3) dispatch
    #   4) yield
    #
    #  This way we can have the CPU schedule ubatch1-dispatch while ubatch0
    #  before yielding back to ubatch1 but ensure we wont start the dispatch
    #  until ubatch0-dispatch is done avoiding overlapping dispatches that
    #  might share underlying buffers
    def gpu_stream_wait(self):
        print("Waiting ubatch %d on %s in stream %s" % (self.id, self.gpu_wait_event, self.stream))
        self.stream.wait_event(self.gpu_wait_event)

    def _yield(self, gpu_wait: bool = True):
        #print("Yielding ubatch %d" % self.id)
        self._signal()
        self._cpu_wait()
        #print("Resuming ubatch %d" % self.id)
        if gpu_wait:
            self.gpu_stream_wait()

    def _signal(self):
        # Wait for the next batch to signal back
        print(f"signaling ubatch {self.id} to {self.gpu_signal_event} on {self.stream}")
        self.gpu_signal_event.record(self.stream)
        # Signal that this batch reached the barrier
        self.cpu_signal_event.set()

    def _cpu_wait(self):
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

_CURRENT_CONTEXT: dict = {}

def get_current_ubatch_context() -> Optional[UBatchContext]:
    global _CURRENT_CONTEXT
    """
    Get the current UBatchContext for the current thread.
    """
    return _CURRENT_CONTEXT.get(threading.get_ident(), None)

def yield_impl(schedule="default", gpu_wait: bool = True):
    # Perform the barrier if a context exists for this thread
    ctx = get_current_ubatch_context() 
    #print("you are in yield_impl", ctx)
    if ctx is not None:
        ctx._yield(gpu_wait=gpu_wait)


# 2) Register kernel for CUDA, mark as mutating to prevent the compiler from
#    optimizing it away (TODO: see if this is actually needed)
@custom_op("vllm::yield_", mutates_args=("x",))
def yield_(x: torch.Tensor, schedule: str="default") -> None:
    yield_impl(schedule)

# 3) Fake implementation for shape prop and FX tracing
@yield_.register_fake
def yield_(x: torch.Tensor, schedule: str="default") -> None:
    pass

def dump_ubatching_state():
    """
    Dump the current UBatchContext state for debugging.
    """
    
    dp_rank = os.getenv("VLLM_DP_RANK", None)
    
    for ctx in _CURRENT_CONTEXT.values():
        print(f"UBatchContext: {ctx.id} (dp_rank {dp_rank})\n"
              f" Stream: {ctx.stream}, ({ctx.stream.query()})\n"
              f" Original Stream: {ctx.original_stream}, ({ctx.original_stream.query()})\n"
              f" CPU Wait Event: {ctx.cpu_wait_event}\n"
              f" GPU Wait Event: {ctx.gpu_wait_event}  ({ctx.gpu_wait_event.query()})\n"
              f" CPU Signal Event: {ctx.cpu_signal_event}\n"
              f" GPU Signal Event: {ctx.gpu_signal_event} ({ctx.gpu_signal_event.query()})\n")



"""

"""
def make_ubatch_context_chain(
    num_micro_batches: int,
    #fwd_ctxs: forward_context.ForwardContext,
    streams: Optional[list[torch.Stream]] = None,
    device: Optional[torch.device] = None
) -> list[UBatchContext]:
    assert num_micro_batches == 2, "only been tested with 2 micro-batches"
    
    """
    Create a context manager for micro-batching synchronization.
    """
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_events = [torch.cuda.Event(blocking=True) for _ in range(num_micro_batches)]
    device = device or torch.cuda.current_device()
    
    ctxs = []
    for i in range(num_micro_batches):
        stream = (streams[i] if streams else None) or torch.cuda.Stream(device)
        ctx = UBatchContext(id=i,
                            stream=stream, 
                            #fwd_ctx=fwd_ctxs[i],
                            cpu_wait_event=cpu_events[i],
                            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
                            gpu_wait_event=gpu_events[i],
                            gpu_signal_event=gpu_events[(i + 1) % num_micro_batches],
                            gpu_wait_on_launch=(i > 0),
                )
        ctxs.append(ctx)
        
    def start_hook(from_stream: torch.cuda.Stream):
        ctxs[0].gpu_wait_event.record(from_stream)
        print('singal to ubatch %d event %s from stream %s' % (ctxs[0].id, ctxs[0].gpu_wait_event, from_stream))
        ctxs[0].cpu_wait_event.set()        

    return ctxs, start_hook