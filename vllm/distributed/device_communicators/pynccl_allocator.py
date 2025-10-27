# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import tempfile
from typing import Any

import torch
from packaging import version
from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import load_inline

from vllm import envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.nccl import find_nccl_include_paths

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
_allocator_wrapper = None
_mem_pool = None
_registered_base_addrs = set()
_graph_pool_id = None
_nccl_allocator_failed_to_compile = False
_cached_pool_snapshot = None


def is_symmetric_memory_enabled():
    global _nccl_allocator_failed_to_compile
    return envs.VLLM_USE_NCCL_SYMM_MEM and not _nccl_allocator_failed_to_compile


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


def compile_nccl_allocator():
    global _allocator, _allocator_wrapper, _nccl_allocator_failed_to_compile
    if not current_platform.is_cuda():
        _nccl_allocator_failed_to_compile = True
        return
    try:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        nccl_include_paths = find_nccl_include_paths()
        load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=envs.VLLM_LOGGING_LEVEL == "DEBUG",
            is_python_module=False,
            build_directory=out_dir,
            extra_include_paths=nccl_include_paths,
        )
        _allocator_wrapper = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        )
        _allocator = _allocator_wrapper.allocator()
    except Exception as e:
        _nccl_allocator_failed_to_compile = True
        logger.warning(
            "Failed to compile NCCL memory allocator. "
            "Symmetric memory will be disabled. "
            "This is expected if NCCL headers are not available. "
            "optionally set VLLM_NCCL_INCLUDE_PATH to point to a directory "
            "containing the NCCL header. "
            "Error: %s",
            str(e),
        )


def get_nccl_mem_pool():
    global _mem_pool, _nccl_allocator_failed_to_compile
    if _mem_pool is None and not _nccl_allocator_failed_to_compile:
        compile_nccl_allocator()
        if _allocator is not None:
            _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


def _cleanup_nccl_mem_pool():
    global _mem_pool
    _mem_pool = None


def _cleanup_nccl_allocator_wrapper():
    global _allocator_wrapper
    _allocator_wrapper = None


atexit.register(_cleanup_nccl_mem_pool)
atexit.register(_cleanup_nccl_allocator_wrapper)


class nccl_symm_mem_context:
    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        disabled: bool = False,
    ):
        self.disabled = (
            disabled
            or not is_symmetric_memory_enabled()
            or pynccl_comm.world_size == 1
            or not current_platform.is_cuda()
            or get_nccl_mem_pool() is None
            or version.parse(torch.__version__) < version.parse("2.8.0.a0")
        )
        if self.disabled:
            self.pynccl_comm: PyNcclCommunicator | None = None
            self._mem_pool_ctx: contextlib.AbstractContextManager[Any] = (
                contextlib.nullcontext()
            )
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
        assert self.pynccl_comm is not None, (
            "Symmetric memory requires pynccl to be initalized"
        )
        assert self.pynccl_comm.nccl_version >= 22703, (
            "NCCL version 2.27.3 or higher is required for NCCL symmetric memory"
        )
        if self.is_graph_capture:
            assert _graph_pool_id is not None, (
                "graph_pool_id is not set under graph capture"
            )
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
        _pool = get_nccl_mem_pool()
        assert _pool is not None
        _cached_pool_snapshot = _pool.snapshot()
        assert self.pynccl_comm is not None
        for segment in _cached_pool_snapshot:
            if segment["address"] not in _registered_base_addrs:
                self.pynccl_comm.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs.add(segment["address"])
        if self.is_graph_capture:
            torch._C._cuda_beginAllocateCurrentThreadToPool(self.device, _graph_pool_id)
