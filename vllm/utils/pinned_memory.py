# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-size pinned host memory.

``pin_memory()`` and ``pin_memory=True`` go through torch's caching host
allocator, which rounds every block up to the next power of two. A buffer that
lives for the whole process gains nothing from that cache, and one sized just
above a power of two pins up to twice its size in page-locked RAM.
"""

import ctypes
import mmap
import weakref
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

# cudaHostAllocPortable | cudaHostAllocMapped (hipHostMallocPortable |
# hipHostMallocMapped have the same values). Mapped keeps the buffer usable
# by get_accelerator_view_from_cpu_tensor.
_HOST_ALLOC_FLAGS = 0x1 | 0x2
# cudaHostRegisterPortable | cudaHostRegisterMapped, the register-path
# equivalents of the two flags above.
_HOST_REGISTER_FLAGS = 0x1 | 0x2


def empty_pinned(
    size: int | Sequence[int],
    dtype: torch.dtype,
    *,
    stride: Sequence[int] | None = None,
    register: bool = False,
    fallback: bool = False,
) -> torch.Tensor:
    """Return an uninitialized pinned CPU tensor that pins exactly its size.

    Meant for buffers that live as long as the process. On CUDA and ROCm the
    memory comes from cudaHostAlloc / hipHostMalloc and is freed when the last
    tensor viewing it is collected; the caller must not drop it while an
    asynchronous copy still reads or writes it. Elsewhere this falls back to
    torch's pinned allocator (still rounded), or to pageable memory when
    pinning is unavailable.

    Args:
        size: Shape of the tensor, or its length if an int.
        dtype: Element type of the tensor.
        stride: Strides in elements; contiguous for ``size`` when omitted.
        register: Page-lock an anonymous mapping with cudaHostRegister /
            hipHostRegister instead of allocating with cudaHostAlloc. Both pin
            exactly ``size``; registered memory is reported differently by the
            driver and is faster to read from the device on some links, so
            callers that have measured a difference can ask for it.
        fallback: Fall back to torch's pinned allocator, with a warning naming
            the requested size, when exact pinning fails. Off by default: a
            failing cudaHostAlloc usually means page-locked memory has run out,
            and the fallback asks for the same bytes rounded up to the next
            power of two, which is the waste this helper exists to avoid.
            Registration can fail for reasons unrelated to free memory, so
            callers using ``register`` may prefer it.

    """
    shape = (size,) if isinstance(size, int) else tuple(size)
    strides = (
        tuple(stride)
        if stride is not None
        else torch.empty(shape, device="meta").stride()
    )
    if not is_pin_memory_available():
        return torch.empty_strided(shape, strides, dtype=dtype, device="cpu")
    from vllm.platforms import current_platform

    numel = 0
    if all(shape):
        numel = 1 + sum((dim - 1) * step for dim, step in zip(shape, strides))
    nbytes = numel * dtype.itemsize
    if not current_platform.is_cuda_alike() or nbytes == 0:
        return torch.empty_strided(
            shape, strides, dtype=dtype, device="cpu", pin_memory=True
        )

    try:
        flat = _register(nbytes) if register else _alloc(nbytes)
    except Exception:
        if not fallback:
            raise
        logger.warning(
            "Could not pin %.2f GB of host memory exactly; falling back to "
            "torch's pinned allocator, which rounds the request up to the "
            "next power of two.",
            nbytes / 1e9,
            exc_info=True,
        )
        return torch.empty_strided(
            shape, strides, dtype=dtype, device="cpu", pin_memory=True
        )
    return flat.view(dtype).as_strided(shape, strides)


def _cudart():
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    # Otherwise is_pinned() reports False for this memory.
    torch.cuda.init()
    return CudaRTLibrary()


def _alloc(nbytes: int) -> torch.Tensor:
    """Allocate with cudaHostAlloc / hipHostMalloc; free with the last view."""
    cudart = _cudart()
    ptr = cudart.cudaHostAlloc(nbytes, _HOST_ALLOC_FLAGS)
    buffer = (ctypes.c_uint8 * nbytes).from_address(ptr)
    release = weakref.finalize(buffer, cudart.cudaFreeHost, ptr)
    # The OS reclaims everything at exit, after the runtime may be gone.
    release.atexit = False  # type: ignore[misc]  # typeshed omits it from slots
    return torch.frombuffer(buffer, dtype=torch.uint8)


def _register(nbytes: int) -> torch.Tensor:
    """Page-lock pageable host memory with cudaHostRegister / hipHostRegister.

    The registered range is page-aligned, which registration wants, and is
    unregistered when the last view of it is collected. The finalizer hangs off
    the backing allocation rather than off this tensor, because the caller gets
    a view and this tensor is collected as soon as it returns.
    """
    cudart = _cudart()
    page = mmap.PAGESIZE
    backing = (ctypes.c_uint8 * (nbytes + page))()
    base = ctypes.addressof(backing)
    offset = -base % page
    ptr = base + offset
    status = cudart.cudaHostRegister(ptr, nbytes, _HOST_REGISTER_FLAGS)
    if status != 0:
        cudart.cudaGetLastError()  # or the next launch check raises this
        raise RuntimeError(
            f"cudaHostRegister/hipHostRegister failed with status {status} "
            f"for {nbytes / 1e9:.2f} GB"
        )
    view = (ctypes.c_uint8 * nbytes).from_buffer(backing, offset)
    release = weakref.finalize(backing, _unregister, cudart, ptr)
    # The OS reclaims everything at exit, after the runtime may be gone.
    release.atexit = False  # type: ignore[misc]  # typeshed omits it from slots
    return torch.frombuffer(view, dtype=torch.uint8)


def _unregister(cudart, ptr: int) -> None:
    cudart.cudaHostUnregister(ptr)
    cudart.cudaGetLastError()
