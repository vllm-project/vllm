import logging
from packaging import version
import tempfile
from typing import Any, TYPE_CHECKING

import torch
from torch.cuda.memory import CUDAPluggableAllocator

from vllm.config import ParallelConfig
from vllm.utils import direct_register_custom_op
if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = logging.getLogger(__name__)

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
_registered_tensor_addrs = set()
_graph_pool_id = None
_nccl_allocator_disabled = False

def is_symmetric_memory_enabled():
    return ParallelConfig.enable_nccl_symm_mem

def is_symmetric_memory_tensor(tensor: torch.Tensor):
    return tensor.untyped_storage().data_ptr() in _registered_tensor_addrs


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _allocator, _mem_pool, _nccl_allocator_disabled
    if _mem_pool is None and not _nccl_allocator_disabled:
        try: 
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
        pynccl_comm: "PyNcclCommunicator",
        disabled: bool = False,
        disable_war: bool = False,
    ):
        self.disabled = (
            disabled
            or not is_symmetric_memory_enabled()
            or pynccl_comm.world_size == 1
            or get_nccl_mem_pool() is None
        )
        if self.disabled:
            self.pynccl_comm = None
            self._mem_pool_ctx = None
            self.is_graph_capture = None
            self.device = None
            self.pre_2_8_0 = None
        else:
            self.pynccl_comm = pynccl_comm
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.is_graph_capture = torch.cuda.is_current_stream_capturing()
            self.device = torch.cuda.current_device()
            self.pre_2_8_0 = version.parse(torch.__version__) < version.parse(
                "2.8.0"
            )
        self.disable_war = disable_war

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
            if self.pre_2_8_0:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    self.device, _graph_pool_id
                )
            else:
                torch._C._cuda_endAllocateToPool(self.device, _graph_pool_id)
        self._mem_pool_ctx.__enter__()
        return self

    def tag(self, tensor: torch.Tensor):
        if self.disabled:
            return
        global _registered_tensor_addrs
        _registered_tensor_addrs.add(tensor.untyped_storage().data_ptr())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled:
            return
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        for segment in get_nccl_mem_pool().snapshot():
            if segment["address"] not in _registered_base_addrs:
                if (
                    segment["stream"] == 0
                    and self.pre_2_8_0
                    and not self.disable_war
                ):
                    # PyTorch version < 2.8.0 has a multi-thread MemPool bug
                    # See https://github.com/pytorch/pytorch/issues/152861
                    # Fixed at https://github.com/pytorch/pytorch/commit/f01e628e3b31852983ab30b25bf251f557ba9c0b
                    # WAR is to skip allocations on the default stream since the
                    # forward_pass thread always runs on a custom stream
                    continue
                self.pynccl_comm.register_comm_window_raw(
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


def all_reduce_symmetric_with_copy_impl(
    input_tensor: torch.Tensor, pynccl_comm: object
) -> torch.Tensor:
    with use_symmetric_memory(pynccl_comm):
        symm_input = torch.empty_like(input_tensor)
        symm_output = torch.empty_like(input_tensor)
    symm_input.copy_(input_tensor)
    symm_output = pynccl_comm.all_reduce(symm_input, symm_output)
    return symm_output

def all_reduce_symmetric_with_copy_fake(
    input_tensor: torch.Tensor, pynccl_comm: object
) -> torch.Tensor:
    return torch.empty_like(input_tensor)


direct_register_custom_op(
    op_name="all_reduce_symmetric_with_copy",
    op_func=all_reduce_symmetric_with_copy_impl,
    mutates_args=[],
    fake_impl=all_reduce_symmetric_with_copy_fake,
)
