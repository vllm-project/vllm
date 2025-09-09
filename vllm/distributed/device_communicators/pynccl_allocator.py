from packaging import version
import tempfile

import torch
from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import load_inline

from vllm import envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger


logger = init_logger(__name__)

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
_nccl_allocator_disabled = False
_cached_pool_snapshot = None

def is_symmetric_memory_enabled():
    global _nccl_allocator_disabled
    return envs.VLLM_USE_NCCL_SYMM_MEM and not _nccl_allocator_disabled


def is_symmetric_memory_tensor(tensor: torch.Tensor):
    if not is_symmetric_memory_enabled() or _cached_pool_snapshot is None:
        return False
    for segment in _cached_pool_snapshot:
        for block in segment["blocks"]:
            if block["address"] == tensor.untyped_storage().data_ptr():
                return True
    return False


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _allocator, _mem_pool, _nccl_allocator_disabled
    if _mem_pool is None and not _nccl_allocator_disabled:
        try: 
            out_dir = tempfile.gettempdir()
            nccl_allocator_libname = "nccl_allocator"
            load_inline(
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
        except Exception as e:
            _nccl_allocator_disabled = True
            logger.warning(
                "Failed to compile NCCL memory allocator. "
                "Symmetric memory will be disabled. "
                "This is expected if NCCL headers are not available. "
                "Error: %s", str(e)
            )
            _mem_pool = None
    return _mem_pool


class use_symmetric_memory:
    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        disabled: bool = False,
    ):
        self.disabled = (
            disabled
            or not is_symmetric_memory_enabled()
            or pynccl_comm.world_size == 1
            or get_nccl_mem_pool() is None
            or version.parse(torch.__version__) < version.parse("2.8.0.a0")
        )
        if self.disabled:
            self.pynccl_comm = None
            self._mem_pool_ctx = None
            self.is_graph_capture = None
            self.device = None
        else:
            self.pynccl_comm = pynccl_comm
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.is_graph_capture = torch.cuda.is_current_stream_capturing()
            self.device = torch.cuda.current_device()

    def __enter__(self):
        if self.disabled:
            return self
        assert (
            self.pynccl_comm is not None
        ), "Symmetric memory requires pynccl to be initalized"
        assert (
            self.pynccl_comm.nccl_version >= 22703
        ), "NCCL version 2.27.3 or higher is required for NCCL symmetric memory"
        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            # Pause graph memory pool to use symmetric memory with cuda graph
            torch._C._cuda_endAllocateToPool(self.device, _graph_pool_id)
        self._mem_pool_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled:
            return
        global _cached_pool_snapshot
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        _cached_pool_snapshot = get_nccl_mem_pool().snapshot()
        for segment in _cached_pool_snapshot:
            if segment["address"] not in _registered_base_addrs:
                self.pynccl_comm.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs.add(segment["address"])
        if self.is_graph_capture:
            torch._C._cuda_beginAllocateCurrentThreadToPool(
                self.device, _graph_pool_id
            )
