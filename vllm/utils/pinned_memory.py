# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-size pinned host memory.

``pin_memory()`` and ``pin_memory=True`` go through torch's caching host
allocator, which rounds every block up to the next power of two. A buffer that
lives for the whole process gains nothing from that cache, and one sized just
above a power of two pins up to twice its size in page-locked RAM.
"""

import ctypes
import weakref
from collections.abc import Sequence

import torch

from vllm.utils.platform_utils import is_pin_memory_available

# cudaHostAllocPortable | cudaHostAllocMapped (hipHostMallocPortable |
# hipHostMallocMapped have the same values). Mapped keeps the buffer usable
# by get_accelerator_view_from_cpu_tensor.
_HOST_ALLOC_FLAGS = 0x1 | 0x2


def empty_pinned(
    size: int | Sequence[int],
    dtype: torch.dtype,
    *,
    stride: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return an uninitialized pinned CPU tensor that pins exactly its size.

    Meant for buffers that live as long as the process. On CUDA and ROCm the
    memory comes from cudaHostAlloc / hipHostMalloc and is freed when the last
    tensor viewing it is collected; the caller must not drop it while an
    asynchronous copy still reads or writes it. Elsewhere this falls back to
    torch's pinned allocator (still rounded), or to pageable memory when
    pinning is unavailable.
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

    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    # Otherwise is_pinned() reports False for this memory.
    torch.cuda.init()
    cudart = CudaRTLibrary()
    ptr = cudart.cudaHostAlloc(nbytes, _HOST_ALLOC_FLAGS)
    buffer = (ctypes.c_uint8 * nbytes).from_address(ptr)
    release = weakref.finalize(buffer, cudart.cudaFreeHost, ptr)
    # The OS reclaims everything at exit, after the runtime may be gone.
    release.atexit = False  # type: ignore[misc]  # typeshed omits it from slots
    flat = torch.frombuffer(buffer, dtype=torch.uint8)
    return flat.view(dtype).as_strided(shape, strides)
