# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-memory helpers: GIL-releasing ``madvise`` and chunked host pinning.

``MADV_POPULATE_WRITE`` over a multi-hundred-GB region spends minutes in the
kernel faulting in pages. CPython calls the syscall without
``Py_BEGIN_ALLOW_THREADS`` (``Modules/mmapmodule.c``) — unlike ``mmap()`` in the
same module, which does release it — so ``mmap.madvise()`` on a background
thread would still stall the engine. ``ctypes.CDLL`` releases the GIL around
foreign calls, so we invoke ``madvise(2)`` directly instead.

Note that ``cudaHostRegister`` needs no such treatment: PyTorch already wraps it
in ``py::gil_scoped_release``.
"""

import ctypes
import mmap
import time
from functools import cache
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

logger = init_logger(__name__)

# cudaHostRegister serializes other CUDA API calls; keep each one short.
HOST_REGISTER_CHUNK_SIZE_BYTES = 256 * MiB_bytes
_HOST_REGISTER_YIELD_SECONDS = 0.001


@cache
def _get_madvise() -> Any:
    library = ctypes.CDLL(None, use_errno=True)
    function = library.madvise
    function.restype = ctypes.c_int
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    )
    return function


def madvise(data_ptr: int, nbytes: int, advice: int) -> None:
    """Apply memory advice through a ctypes call that releases the GIL."""
    if _get_madvise()(data_ptr, nbytes, advice) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "madvise failed")


class HostRegisterError(RuntimeError):
    """A cudaHostRegister call failed; the memory is usable but unpinned."""


def host_register_chunked(
    base_ptr: int, nbytes: int, chunk_size_bytes: int | None = None
) -> list[int]:
    """Page-lock ``[base_ptr, base_ptr + nbytes)`` via cudaHostRegister.

    When ``chunk_size_bytes`` is set (a positive page multiple), register
    incrementally and yield between chunks so inference can submit CUDA work
    in the gaps. Interior cuts land on absolute page boundaries even for an
    unaligned ``base_ptr``: cudaHostRegister locks whole pages and rejects a
    range overlapping an already-registered page, so a mid-page cut would
    fail.

    Returns:
        The registered chunk base pointers, for `host_unregister` on teardown.

    Raises:
        HostRegisterError: A registration call failed. Chunks registered so
            far are unregistered first, as they are for any other exception.
    """
    page_size = mmap.PAGESIZE
    if chunk_size_bytes is not None and (
        chunk_size_bytes <= 0 or chunk_size_bytes % page_size != 0
    ):
        raise ValueError("chunk_size_bytes must be a positive page multiple")

    registered: list[int] = []
    end = base_ptr + nbytes
    pos = base_ptr
    cudart = torch.cuda.cudart()
    try:
        while pos < end:
            if chunk_size_bytes is None:
                boundary = end
            else:
                boundary = min(end, pos - pos % page_size + chunk_size_bytes)
            result = cudart.cudaHostRegister(pos, boundary - pos, 0).value
            if result != 0:
                raise HostRegisterError(f"cudaHostRegister failed (code={result})")
            registered.append(pos)
            pos = boundary
            if pos < end:
                time.sleep(_HOST_REGISTER_YIELD_SECONDS)
    except BaseException:
        host_unregister(registered)
        raise
    return registered


def host_unregister(ptrs: list[int]) -> None:
    """Release page-locked ranges returned by `host_register_chunked`."""
    cudart = torch.cuda.cudart()
    for ptr in reversed(ptrs):
        result = cudart.cudaHostUnregister(ptr).value
        if result != 0:
            logger.warning("cudaHostUnregister failed (code=%d)", result)
