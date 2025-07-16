import torch
import os
import torch.distributed as dist

import torch.distributed
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group


local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local_rank)
print(f'local_rank: {local_rank}')

device = torch.device(f"cuda:{local_rank}")
dist.init_process_group(backend="nccl", device_id=device)

ranks = [i for i in range(world_size)]
cpu_group = torch.distributed.new_group(ranks, backend="gloo")




pynccl_comm = PyNcclCommunicator(
    group=cpu_group,
    device=device,
)
pynccl_comm.disabled = False

# create allocator
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
nccl_allocator_libname = "nccl_allocator"
nccl_allocator = torch.utils.cpp_extension.load_inline(
    name=nccl_allocator_libname,
    cpp_sources=nccl_allocator_source,
    with_cuda=True,
    extra_ldflags=["-lnccl"],
    verbose=True,
    is_python_module=False,
    build_directory="./",
)

allocator = CUDAPluggableAllocator(
    f"./{nccl_allocator_libname}.so", "nccl_alloc_plug", "nccl_free_plug"
).allocator()
pool = torch.cuda.MemPool(allocator)

default_pg = _get_default_group()
backend = default_pg._get_backend(device)

size = 1024*1024
input = torch.full([size], local_rank, dtype=torch.float16, device=device)

with torch.cuda.use_mem_pool(pool):
    # tensor gets allocated with ncclMemAlloc passed in the pool
    symm_input = torch.full([size], local_rank, dtype=torch.float16, device=device)


pynccl_comm.register_comm_window(symm_input)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

stream = torch.cuda.default_stream()
# Warmup
torch.distributed.barrier()
pynccl_comm.all_reduce(input, stream=stream)

start.record()
for _ in range(100):
    pynccl_comm.all_reduce(input, stream=stream)
end.record()
torch.cuda.synchronize()

if local_rank == 0:
    print(f'default={start.elapsed_time(end)/100:.3f}')

# Warmup
torch.distributed.barrier()
pynccl_comm.all_reduce(symm_input, stream=stream)

start.record()
for _ in range(100):
    pynccl_comm.all_reduce(symm_input, stream=stream)
end.record()
torch.cuda.synchronize()

if local_rank == 0:
    print(f'symm_memory={start.elapsed_time(end)/100:.3f}')

dist.destroy_process_group()