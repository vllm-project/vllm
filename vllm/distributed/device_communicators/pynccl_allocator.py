import tempfile
import torch
from torch.cuda.memory import CUDAPluggableAllocator
from vllm.distributed.parallel_state import GroupCoordinator

nccl_allocator_source = """
#include <nccl.h>
#include <c10/cuda/CUDAGuard.h>
extern "C" {

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return ptr;

}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""

_allocator = None
_mem_pool = None


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
