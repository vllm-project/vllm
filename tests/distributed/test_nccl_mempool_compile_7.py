import torch
import os
import torch.distributed as dist
import logging
import tempfile
from contextlib import nullcontext
from typing import List
from packaging import version
import argparse

import torch.distributed
from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils import cpp_extension

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import GroupCoordinator, init_world_group
from vllm.utils import direct_register_custom_op
from torch.distributed._functional_collectives import all_reduce as f_all_reduce

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


def is_symmetric_memory_enabled():
    return True


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
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


class use_symmetric_memory:
    def __init__(self, group_coordinator: GroupCoordinator):
        if not is_symmetric_memory_enabled():
            self.group_coordinator = None
            self._mem_pool_ctx = None
            self.is_graph_capture = None
            self.device = None
            self.pre_2_8_0 = None
        else:
            #print("IN INIT")
            self.group_coordinator = group_coordinator
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.is_graph_capture = torch.cuda.is_current_stream_capturing()
            self.device = torch.cuda.current_device()
            self.pre_2_8_0 = version.parse(torch.__version__) < version.parse("2.8.0")
            #print(self.pre_2_8_0, version.parse(torch.__version__))
    def __enter__(self):
        if not is_symmetric_memory_enabled():
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
            if self.pre_2_8_0:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    self.device, _graph_pool_id
                )
            else:
                torch._C._cuda_endAllocateToPool(self.device, _graph_pool_id)
        #print("IN enter and calling mem pool enter")
        self._mem_pool_ctx.__enter__()
        return self

    def tag(self, tensor: torch.Tensor):
        if not is_symmetric_memory_enabled():
            return
        #print("IN tag")
        tensor.symmetric_memory = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_symmetric_memory_enabled():
            return
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        #print("IN exit and calling mem pool exit")
        for segment in get_nccl_mem_pool().snapshot():
            #print("there is a segment")
            if segment["address"] not in _registered_base_addrs:
                if segment["stream"] == 0 and self.pre_2_8_0:
                    # PyTorch version < 2.8.0 has a multi-thread MemPool bug
                    # See https://github.com/pytorch/pytorch/issues/152861
                    # Fixed at https://github.com/pytorch/pytorch/commit/f01e628e3b31852983ab30b25bf251f557ba9c0b
                    # WAR is to skip allocations on the default stream since the forward_pass thread always runs on a custom stream
                    continue
                print("IN register comm window raw")
                self.group_coordinator.device_communicator.pynccl_comm.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs.add(segment["address"])

        if self.is_graph_capture:
            if self.pre_2_8_0:
                torch._C._cuda_beginAllocateToPool(self.device, _graph_pool_id)
            else:
                torch._C._cuda_beginAllocateCurrentThreadToPool(
                    self.device, _graph_pool_id
                )

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
# x = torch.full([1024,1024], local_rank, dtype=torch.float16, device='cuda')
# symm_input = torch.ops.nccl_symm_cache.nccl_malloc_tensor(
#             [1024,1024],
#             torch.float16,
#             'cuda',
#             )
# symm_input.copy_(x)
# #_WORLD.device_communicator.pynccl_comm.all_reduce(symm_input,symm_input)
# _WORLD.device_communicator.pynccl_comm.register_comm_window(symm_input)
# _WORLD.device_communicator.pynccl_comm.all_reduce(symm_input,symm_input)
# print(symm_input)

def pynccl_all_reduce_impl(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    """Implementation function for the custom NCCL all_reduce operator."""
    print(local_rank ,"ar ptr    :", hex(input_tensor.data_ptr()), hex(output_tensor.data_ptr()))
    return _WORLD.device_communicator.pynccl_comm.all_reduce(input_tensor, output_tensor)


def pynccl_all_reduce_fake(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    output_tensor = f_all_reduce(input_tensor, reduceOp="sum", group=_WORLD.device_group)
    return output_tensor

# Register the custom operator
direct_register_custom_op(
    op_name="pynccl_all_reduce",
    op_func=pynccl_all_reduce_impl,
    mutates_args=["output_tensor"],
    fake_impl=pynccl_all_reduce_fake,
)

def get_symm_input_impl(template: torch.Tensor) -> torch.Tensor:
    # with use_symmetric_memory(_WORLD) as sm:
    #     mem = torch.empty_like(template)
    #     sm.tag(mem)
    with torch.cuda.use_mem_pool(get_nccl_mem_pool()):
        mem = torch.empty_like(template)
        mem.symmetric_memory = True
        _WORLD.device_communicator.pynccl_comm.register_comm_window(mem)
        print(local_rank,"alloc ptr :", hex(mem.data_ptr()))
    return mem

def get_symm_input_fake(template: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(template)

direct_register_custom_op(
    op_name="get_symm_input",
    op_func=get_symm_input_impl,
    mutates_args=["template"],
    fake_impl=get_symm_input_fake,
    tags=(torch.Tag.maybe_aliasing_or_mutating,),
)

def symm_all_reduce_with_copy_impl(input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.cuda.use_mem_pool(get_nccl_mem_pool()):
        symm_input = torch.empty_like(input_tensor)
        symm_input.symmetric_memory = True
        symm_output = torch.empty_like(input_tensor)
        symm_output.symmetric_memory = True
        _WORLD.device_communicator.pynccl_comm.register_comm_window(symm_input)
        _WORLD.device_communicator.pynccl_comm.register_comm_window(symm_output)
    symm_input.copy_(input_tensor)
    _WORLD.device_communicator.pynccl_comm.all_reduce(symm_input, symm_output)
    return symm_output

def symm_all_reduce_with_copy_fake(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input_tensor)

direct_register_custom_op(
    op_name="symm_all_reduce_with_copy",
    op_func=symm_all_reduce_with_copy_impl,
    mutates_args=[],#"output_tensor"
    fake_impl=symm_all_reduce_with_copy_fake,
    #tags=(torch.Tag.maybe_aliasing_or_mutating,),
)

def outplace_add_impl(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a,b,out=out)

def outplace_add_fake(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    return

direct_register_custom_op(
    op_name="outplace_add",
    op_func=outplace_add_impl,
    mutates_args=["out"],
    fake_impl=outplace_add_fake,
    tags=(torch.Tag.maybe_aliasing_or_mutating,),
)

class simple_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y =torch.add(x,x)
        z = torch.add(y,x)
        # symm_input = torch.ops.vllm.get_symm_input(z)
        # torch.add(z,x,out=symm_input)
        #torch.ops.vllm.outplace_add(z,x,symm_input)
        # torch.ops.vllm.pynccl_all_reduce(
        #     symm_input,
        #     symm_input,
        # )
        symm_input = torch.add(z,x)
        symm_input = torch.ops.vllm.symm_all_reduce_with_copy(symm_input)
        a = torch.add(symm_input,symm_input)
        b = torch.add(a,a)
        return b

# warm up idea
# with torch.cuda.use_mem_pool(get_nccl_mem_pool()):
#     mem = torch.full([1024,1024], local_rank/1000, dtype=torch.float16, device="cuda")
# for i in range(1000):
#     torch.ops.vllm.pynccl_all_reduce(mem,mem)
# print(mem)
# print("WARM UP DONE")
# torch.distributed.barrier()

# model and settings
parser = argparse.ArgumentParser(description='NCCL symmetric cache torch compile test')
parser.add_argument('--compile', action='store_true', default=False,
                    help='Use torch.compile for the model (default: False)')
args = parser.parse_args()

use_compiled = args.compile

model = simple_model()
if use_compiled:
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)

# first test
size = [2048,2048]
dtype = torch.float16
x = torch.full(size, local_rank, dtype=dtype, device="cuda")
for _ in range(1):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

# continue second test
size = [4096,4096]
dtype = torch.float16
x = torch.full(size, local_rank*2, dtype=dtype, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

# continue third test
size = [8192,8192]
dtype = torch.float16
x = torch.full(size, local_rank*4, dtype=dtype, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

# forth test
size = [512,512]
dtype = torch.float16
x = torch.full(size, local_rank, dtype=dtype, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

print("done")
dist.destroy_process_group()