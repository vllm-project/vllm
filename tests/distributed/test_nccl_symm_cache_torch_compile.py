import torch
import os
import torch.distributed as dist
import logging
import tempfile
from contextlib import nullcontext
from typing import List

import torch.distributed
from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils import cpp_extension

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import GroupCoordinator, init_world_group
from vllm.utils import direct_register_custom_op


nccl_mem_alloc_source = """
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <nccl.h>

at::Tensor nccl_malloc_tensor(
    const std::vector<int64_t>& shape,
    c10::ScalarType dtype,
    const c10::Device& device) {
  size_t size = c10::elementSize(dtype) * c10::multiply_integers(shape);
  void* ptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return at::from_blob(
      ptr, shape, [](void* ptr) { ncclMemFree(ptr); },
      at::TensorOptions().dtype(dtype).device(device));
}
TORCH_LIBRARY(nccl_symm_cache, m) {
  m.def("nccl_malloc_tensor", &nccl_malloc_tensor);
}

"""
out_dir = tempfile.gettempdir()
nccl_mem_alloc_libname = "nccl_mem_alloc"
torch.utils.cpp_extension.load_inline(
    name=nccl_mem_alloc_libname,
    cpp_sources=nccl_mem_alloc_source,
    with_cuda=True,
    extra_ldflags=["-lnccl"],
    verbose=True,
    is_python_module=False,
    build_directory=out_dir,
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
    return _WORLD.device_communicator.pynccl_comm.all_reduce(input_tensor, output_tensor)


def pynccl_all_reduce_fake(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return output_tensor

# Register the custom operator
direct_register_custom_op(
    op_name="pynccl_all_reduce",
    op_func=pynccl_all_reduce_impl,
    mutates_args=["output_tensor"],
    fake_impl=pynccl_all_reduce_fake,
)

buffer_cache = {}
def get_symm_input_impl(size: List[int], dtype: torch.dtype) -> torch.Tensor:
    if buffer_cache.get(size, dtype) is None:
      buffer_cache[size, dtype] = torch.ops.nccl_symm_cache.nccl_malloc_tensor(
          size,
          dtype,
          'cuda',
      )
      _WORLD.device_communicator.pynccl_comm.register_comm_window(
          buffer_cache[size, dtype])
    return buffer_cache[size, dtype]

def get_symm_input_fake(size: List[int], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(size, dtype=dtype, device='cuda')

direct_register_custom_op(
    op_name="get_symm_input",
    op_func=get_symm_input_impl,
    mutates_args=[],
    fake_impl=get_symm_input_fake,
)

class simple_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.symm_input = get_symm_input(size, dtype)
    def forward(self, x):
        y =torch.add(x,x)
        z = torch.add(y,x)
        symm_input = torch.ops.vllm.get_symm_input(size, dtype)
        torch.add(z,x,out=symm_input)
        torch.ops.vllm.pynccl_all_reduce(
            symm_input,
            symm_input,
        )
        a = torch.add(symm_input,symm_input)
        b = torch.add(a,a)
        return b

use_compiled = True
size = [1024,1024]
dtype = torch.float16
model = simple_model()
if use_compiled:
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
x = torch.full(size, local_rank, dtype=dtype, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

size = [512,512]
dtype = torch.float16
x = torch.full(size, local_rank*2, dtype=dtype, device="cuda")
for _ in range(10):
    y = compiled_model(x) if use_compiled else model(x)
print(y)

dist.destroy_process_group()