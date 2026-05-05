# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
from typing import Any

import torch
from packaging import version

from vllm import envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Previously this module compiled a C++ shim at runtime (load_inline) that
# wrapped ncclMemAlloc/ncclMemFree in a CUDAPluggableAllocator. PyTorch 2.11
# ships NCCLSymmetricMemoryAllocator natively (torch/csrc/distributed/c10d/
# symm_mem/NCCLSymmetricMemory.cu) which does the same thing. We now use:
#
#   set_backend("NCCL") + get_mem_pool(device)
#
# This eliminates runtime C++ compilation and the NCCL header dependency.
# Requires torch >= 2.11.0 (pytorch/pytorch#168129 fixed a dead get_group_info
# call in alloc() that made the NCCL backend unusable in torch 2.10).

_mem_pool = None
_registered_base_addrs: set = set()
_graph_pool_id = None
_nccl_symm_mem_init_failed = False
_backend_initialized = False
_cached_pool_snapshot = None


def _ensure_nccl_symm_mem_backend() -> None:
    """Initialize PyTorch's NCCL symmetric memory backend (once).

    Must be called before any torch.distributed._symmetric_memory allocations
    so that set_backend("NCCL") takes effect before SymmMemCommunicator or
    any other caller can lock in a different backend.
    """
    global _backend_initialized, _nccl_symm_mem_init_failed
    if _backend_initialized:
        return
    _backend_initialized = True

    if not current_platform.is_cuda():
        _nccl_symm_mem_init_failed = True
        return

    try:
        from torch.distributed._symmetric_memory import set_backend
        set_backend("NCCL")
    except Exception as e:
        _nccl_symm_mem_init_failed = True
        logger.warning(
            "Failed to initialize NCCL symmetric memory backend. "
            "Symmetric memory will be disabled. Error: %s",
            str(e),
        )


def is_symmetric_memory_enabled() -> bool:
    if not envs.VLLM_USE_NCCL_SYMM_MEM:
        return False
    # Ensure set_backend("NCCL") runs before SymmMemCommunicator allocates,
    # since is_symmetric_memory_enabled() is called first in CudaCommunicator.
    _ensure_nccl_symm_mem_backend()
    return not _nccl_symm_mem_init_failed


def is_symmetric_memory_tensor(tensor: torch.Tensor) -> bool:
    if not is_symmetric_memory_enabled() or _cached_pool_snapshot is None:
        return False
    for segment in _cached_pool_snapshot:
        for block in segment["blocks"]:
            if block["address"] == tensor.untyped_storage().data_ptr():
                return True
    return False


def set_graph_pool_id(graph_pool_id: Any) -> None:
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _mem_pool, _nccl_symm_mem_init_failed
    if _mem_pool is None and not _nccl_symm_mem_init_failed:
        _ensure_nccl_symm_mem_backend()
        if not _nccl_symm_mem_init_failed:
            try:
                from torch.distributed._symmetric_memory import get_mem_pool
                device = torch.cuda.current_device()
                _mem_pool = get_mem_pool(device)
            except Exception as e:
                _nccl_symm_mem_init_failed = True
                logger.warning(
                    "Failed to get NCCL symmetric memory pool. "
                    "Symmetric memory will be disabled. Error: %s",
                    str(e),
                )
    return _mem_pool


def _cleanup_nccl_mem_pool() -> None:
    global _mem_pool
    _mem_pool = None


atexit.register(_cleanup_nccl_mem_pool)


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
            or version.parse(torch.__version__) < version.parse("2.11.0")
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
            self.device = torch.accelerator.current_device_index()

    def __enter__(self):
        if self.disabled:
            return self
        assert self.pynccl_comm is not None, (
            "Symmetric memory requires pynccl to be initialized"
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
            torch._C._cuda_beginAllocateCurrentThreadToPool(
                self.device, _graph_pool_id
            )
