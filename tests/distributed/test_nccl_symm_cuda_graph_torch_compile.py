import torch
import os
import torch.distributed as dist
import logging
import tempfile
from contextlib import nullcontext

import torch.distributed
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import GroupCoordinator, init_world_group

from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils import cpp_extension

nccl_allocator_source = """
#include <nccl.h>
extern "C" {
void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return ptr;
}
void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
}
}
"""

_allocator = None
_mem_pool = None
_registered_base_addrs = set()
_graph_pool_id = None
_symmetric_memory_enabled = False


def enable_symmetric_memory():
    global _symmetric_memory_enabled
    _symmetric_memory_enabled = True


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _allocator, _mem_pool
    if _mem_pool is None:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so", "nccl_alloc_plug", "nccl_free_plug"
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


class use_symmetric_memory:
    def __init__(self, group_coordinator: GroupCoordinator):
        self.group_coordinator = group_coordinator
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool()) if _symmetric_memory_enabled else nullcontext()
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()

    def __enter__(self):
        if not _symmetric_memory_enabled:
            return self
        assert (
            self.group_coordinator.device_communicator.pynccl_comm is not None
        ), f"Symmetric memory requires pynccl to be enabled in group '{self.group_coordinator.group_name}'"
        assert (
            self.group_coordinator.device_communicator.pynccl_comm.nccl_version >= 22703
        ), "NCCL version 2.27.3 or higher is required for NCCL symmetric memory"
        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            # Pause graph memory pool to use symmetric memory with cuda graph
            torch._C._cuda_endAllocateCurrentStreamToPool(self.device, _graph_pool_id)
        self._mem_pool_ctx.__enter__()
        return self

    def tag(self, tensor: torch.Tensor):
        if not _symmetric_memory_enabled:
            return
        tensor.symmetric_memory = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _symmetric_memory_enabled:
            return
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        for segment in get_nccl_mem_pool().snapshot():
            if segment['address'] not in _registered_base_addrs:
                if segment['stream'] == 0:
                    # PyTorch version < 2.8.0 has a multi-thread MemPool bug
                    # See https://github.com/pytorch/pytorch/issues/152861
                    # Fixed at https://github.com/pytorch/pytorch/commit/f01e628e3b31852983ab30b25bf251f557ba9c0b
                    # WAR is to skip allocations on the default stream since the forward_pass thread always runs on a custom stream
                    continue
                self.group_coordinator.device_communicator.pynccl_comm.register_comm_window_raw(segment['address'], segment['total_size'])
                _registered_base_addrs.add(segment['address'])

        if self.is_graph_capture:
            torch._C._cuda_beginAllocateToPool(self.device, _graph_pool_id)

class simple_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y =torch.add(x,x)
        z = torch.add(y,x)
        with use_symmetric_memory(_WORLD) as sm:
            k = torch.empty_like(z)
        torch.add(z,x,out=k)
        sm.tag(k)
        _WORLD.device_communicator.pynccl_comm.all_reduce(k,k)
        a = torch.add(k,k)
        b = torch.add(a,a)
        return b

# compiled_computation = torch.compile(simple_model, backend="inductor", fullgraph=True)

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
print(f"local_rank: {local_rank}")

device = torch.device(f"cuda:{local_rank}")
dist.init_process_group(backend="nccl", device_id=device)

ranks = [i for i in range(world_size)]
cpu_group = torch.distributed.new_group(ranks, backend="gloo")

_WORLD = GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=True,
        group_name="world",
        )

torch._logging.set_logs(dynamo=logging.DEBUG)  # See Dynamo tracing
torch._logging.set_logs(graph=True)           # See traced graph
torch._logging.set_logs(output_code=True)     # See generated code
torch._logging.set_logs(fusion=True)          # See fusion decisions
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo,+inductor"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = "/tmp/torch_compile_debug"

enable_symmetric_memory()
use_compiled = True
model = simple_model()
if use_compiled:
    get_nccl_mem_pool() # otherwise load_inline fails with dynamo
    compiled_model = torch.compile(model, backend="inductor", fullgraph=False)
x = torch.full([128], local_rank, dtype=torch.float16, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

dist.destroy_process_group()