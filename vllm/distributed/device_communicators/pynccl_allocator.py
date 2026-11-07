# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import hashlib
import logging
import os
import stat
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
from vllm.utils.nccl import find_nccl_include_paths, find_nccl_library_paths

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
_registered_base_addrs: dict[bytes, set] = {}
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


def set_graph_pool_id(graph_pool_id: Any) -> None:
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def _current_uid() -> int:
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else -1


def _verify_private_entry(info: os.stat_result, path: str, *, expect_dir: bool) -> None:
    """Check a file entry is safe to trust before it is used or loaded.

    Follows the weight cache protocol
    (vllm/model_executor/model_loader/weight_cache/protocol.py): reject
    symlinks and entries owned by a different user, and require the
    directory to be inaccessible to group and world. Permission bits are
    only meaningful on POSIX; on Windows isolation comes from the
    per-profile directory.
    """
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"Refusing to use symlinked path {path}")
    if expect_dir:
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{path} is not a directory")
        if os.name == "posix" and info.st_mode & 0o077:
            raise RuntimeError(f"{path} is group/world accessible")
    elif not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{path} is not a regular file")
    uid = _current_uid()
    if uid != -1 and info.st_uid != uid:
        raise RuntimeError(f"{path} is not owned by the current user")


def _verify_private_dir(directory: str) -> None:
    """Verify a dir is a real dir owned by us, not group/world accessible."""
    _verify_private_entry(os.lstat(directory), directory, expect_dir=True)


def _verify_owned_file(path: str) -> None:
    """Verify a file is ours and lives in a private dir before dlopen.

    Called on the compiled allocator library so this process never loads
    a library a different local user could have planted.
    """
    _verify_private_dir(os.path.dirname(path) or os.curdir)
    _verify_private_entry(os.lstat(path), path, expect_dir=False)


def _cache_key(
    libname: str,
    ldflags: list[str],
    include_paths: list[str] | None,
) -> str:
    """Hash everything that can change the compiled library.

    Mirrors torch's extension versioner (sources plus build flags), so a
    cached artifact is never reused under a key that no longer matches.
    """
    material = "\0".join(
        (
            nccl_allocator_source,
            libname,
            repr(ldflags),
            repr(include_paths),
            torch.__version__,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _prepare_build_dir(
    libname: str,
    ldflags: list[str],
    include_paths: list[str] | None,
) -> str:
    """Return a private build directory for the JIT-compiled allocator.

    The build artifacts include the .so that is dlopen'ed into this
    process, so they must never live in a shared sticky directory under
    a predictable name: a local user could pre-plant a malicious library
    that ninja then skips rebuilding. Prefer a per-user cache directory
    keyed by the build inputs (so the library is compiled once and
    reused across processes), locked down and verified following
    ensure_private_socket_dir. Fall back to a fresh private temp
    directory when the cache root is unusable.
    """
    try:
        out_dir = os.path.join(
            envs.VLLM_CACHE_ROOT, libname, _cache_key(libname, ldflags, include_paths)
        )
        os.makedirs(out_dir, mode=0o700, exist_ok=True)
        os.chmod(out_dir, 0o700)
        _verify_private_dir(out_dir)
        return out_dir
    except (OSError, RuntimeError, AttributeError) as e:
        logger.warning(
            "Could not use a build directory under VLLM_CACHE_ROOT (%s) "
            "for the NCCL allocator, falling back to a private temp "
            "directory. Error: %s",
            envs.VLLM_CACHE_ROOT,
            str(e),
        )
        return tempfile.mkdtemp(prefix=f"vllm-{libname}-")


def compile_nccl_allocator():
    global _allocator, _allocator_wrapper, _nccl_allocator_failed_to_compile
    if not current_platform.is_cuda():
        _nccl_allocator_failed_to_compile = True
        return
    try:
        nccl_allocator_libname = "nccl_allocator"
        nccl_include_paths = find_nccl_include_paths()
        ldflags = ["-l:libnccl.so.2"]
        nccl_lib_paths = find_nccl_library_paths()
        if nccl_lib_paths:
            ldflags = [f"-L{p}" for p in nccl_lib_paths] + ldflags
        out_dir = _prepare_build_dir(
            nccl_allocator_libname, ldflags, nccl_include_paths
        )
        load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=ldflags,
            verbose=logger.isEnabledFor(logging.DEBUG),
            is_python_module=False,
            build_directory=out_dir,
            extra_include_paths=nccl_include_paths,
        )
        nccl_allocator_lib_path = os.path.join(out_dir, f"{nccl_allocator_libname}.so")
        # The library is dlopen'ed into this process; refuse anything a
        # different local user could have planted.
        _verify_owned_file(nccl_allocator_lib_path)
        _allocator_wrapper = CUDAPluggableAllocator(
            nccl_allocator_lib_path,
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
        comm_key = bytes(self.pynccl_comm.unique_id.internal)
        if comm_key not in _registered_base_addrs:
            _registered_base_addrs[comm_key] = set()
        for segment in _cached_pool_snapshot:
            if segment["address"] not in _registered_base_addrs[comm_key]:
                self.pynccl_comm.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs[comm_key].add(segment["address"])
        if self.is_graph_capture:
            torch._C._cuda_beginAllocateCurrentThreadToPool(self.device, _graph_pool_id)
