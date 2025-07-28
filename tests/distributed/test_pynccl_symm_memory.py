import torch
import os
import torch.distributed as dist
import logging
import tempfile

import torch.distributed
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils import cpp_extension

nccl_allocator_source = """
#include <nccl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <iostream>
extern "C" {

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  std::cout << "Using ncclMemAlloc" << std::endl;
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    std::cerr << "nccl_alloc_plug: in graph capture" << std::endl;
    assert(false);
  }
  void* ptr;
  //at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return ptr;

}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  std::cout << "Using ncclMemFree" << std::endl;
  //at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""


_allocator = None
_mem_pool = None
_registered_base_addrs = set()
_graph_pool_id = None


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
    def __init__(self, pynccl):
        self.pynccl = pynccl
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()

    def __enter__(self):
        if self.is_graph_capture:
            print(
                f"use_symmetric_memory: endAllocateCurrentStreamToPool ={torch.cuda.MemPoolContext.active_pool()}"
            )
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            torch._C._cuda_endAllocateCurrentStreamToPool(
                self.device, _graph_pool_id
            )
        self._mem_pool_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        for segment in get_nccl_mem_pool().snapshot():
            if segment["address"] not in _registered_base_addrs:
                self.pynccl.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs.add(segment["address"])
        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            torch._C._cuda_beginAllocateToPool(self.device, _graph_pool_id)


def test(size, pynccl_comm):
    with use_symmetric_memory(pynccl_comm):
        symm_input = torch.full([size], 0, dtype=torch.float16, device="cuda")

    if torch.distributed.get_rank() == 0:
        print(
            f"total_size={sum(segment['total_size'] for segment in get_nccl_mem_pool().snapshot())} allocated={sum(segment['allocated_size'] for segment in get_nccl_mem_pool().snapshot())}"
        )
        print(f"{get_nccl_mem_pool().snapshot()=}")
    pynccl_comm.all_reduce(symm_input,symm_input)


local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
print(f"local_rank: {local_rank}")

device = torch.device(f"cuda:{local_rank}")
dist.init_process_group(backend="nccl", device_id=device)

ranks = [i for i in range(world_size)]
cpu_group = torch.distributed.new_group(ranks, backend="gloo")


pynccl_comm = PyNcclCommunicator(
    group=cpu_group,
    device=device,
)


# Warmup
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for s in range(1, 6):
        test(s * 1024 * 512, pynccl_comm)
torch.cuda.current_stream().wait_stream(stream)
torch.distributed.barrier()

print("After warmup")

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for s in range(1, 6):
        test(s * 1024 * 512, pynccl_comm)
torch.cuda.current_stream().wait_stream(stream)
torch.distributed.barrier()
g = torch.cuda.CUDAGraph()
p = torch.cuda.graph_pool_handle()
set_graph_pool_id(p)
print("Start graph")
if torch.distributed.get_rank() == 0:
    print(f"before graph total_size={sum(segment['total_size'] for segment in get_nccl_mem_pool().snapshot())} allocated={sum(segment['allocated_size'] for segment in get_nccl_mem_pool().snapshot())}")
with torch.cuda.graph(g, pool=p, stream=stream):
    if torch.distributed.get_rank() == 0:
        print(f"in graph total_size={sum(segment['total_size'] for segment in get_nccl_mem_pool().snapshot())} allocated={sum(segment['allocated_size'] for segment in get_nccl_mem_pool().snapshot())}")
    for s in range(1, 6):
        test(s*1024*512, pynccl_comm)

torch.cuda.synchronize()
torch.distributed.barrier()
g.replay()

torch.cuda.synchronize()
torch.distributed.barrier()

print("Done")

dist.destroy_process_group()
