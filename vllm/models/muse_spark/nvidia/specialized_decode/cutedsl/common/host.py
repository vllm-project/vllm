# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side helpers for the CuTe providers of the specialized decode."""

import math
import struct
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_tensor

from vllm.platforms import current_platform

from .primitives import to_cute, to_cute_dynamic

_NAMES = ("BFloat16", "Float32", "Uint8", "Uint32", "Int32", "Int64")
_DTYPES = {getattr(torch, name.lower()): getattr(cutlass, name) for name in _NAMES}


def layout(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def stream(device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def on_device(index: int, launch: Callable[..., None], *args) -> None:
    """``launch(*args)`` with CUDA device ``index`` current, as a C++ op's
    device guard runs it: inputs on another GPU than the current one launch
    there. For a compiled TVM FFI call that takes no GPU tensor (addresses
    only); one that does switches to its first GPU tensor's device itself."""
    if index == torch.accelerator.current_device_index():
        launch(*args)
    else:
        with torch.accelerator.device_index(index):
            launch(*args)


def fp32(x: float) -> float:
    """``x`` rounded to FP32 (to nearest even), as a Float32 argument takes
    it."""
    return struct.unpack("f", struct.pack("f", x))[0]


def architecture(device):
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def fake(shape, stride=None, dtype=torch.bfloat16, align=16):
    """Row-major unless ``stride`` is given."""
    stride = stride or tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    dtype = _DTYPES.get(dtype, dtype)
    return make_fake_tensor(dtype, shape, stride, assumed_align=align)


def dynamic_fake(rank, dtype=torch.bfloat16):
    shape = tuple(cute.sym_int() for _ in range(rank))
    return fake(shape, (*(cute.sym_int(64) for _ in range(rank - 1)), 1), dtype)


def dynamic_view(tensor, align=16):
    return to_cute(tensor, align).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def pdl_supported() -> bool:
    """Whether kernels launch with programmatic dependent launch here."""
    return current_platform.is_cuda() and current_platform.is_arch_support_pdl()


def rows_view(tensor: torch.Tensor, width: int, alignment: int = 16) -> cute.Tensor:
    """The dynamic CuTe view of contiguous rows of ``width`` (any row count)
    that the row kernels take, ``alignment``-byte aligned."""
    return to_cute_dynamic(tensor.flatten(), alignment, divisibility=width)


def check_operand(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    align: int,
    name: str,
) -> None:
    """Raise ValueError unless ``tensor`` is a contiguous ``align``-byte
    aligned ``shape`` ``dtype`` tensor (what a kernel's vector accesses
    assume)."""
    if (
        tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
        or tensor.data_ptr() % align
    ):
        raise ValueError(
            f"{name} must be a contiguous {align}-byte aligned {list(shape)} {dtype} "
            "tensor"
        )


# Bytes of the per-device buffer behind ``unread_operand``: an [8192] FP32 row.
_UNREAD_BYTES = 8192 * 4
_UNREAD: dict[torch.device, torch.Tensor] = {}


def unread_operand(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """A contiguous 32 KiB [n] ``dtype`` tensor for an operand that a kernel
    variant neither reads nor writes (a norm weight it does not apply, a row it
    does not store). Every caller on a device gets a view of the same buffer,
    allocated once and kept for the process, so the operand costs no memory
    per op. The buffer is all ones bits (NaN in FP32 and BF16), so a kernel
    that read it would show in its output."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    buffer = _UNREAD.get(device)
    if buffer is None:
        buffer = torch.full((_UNREAD_BYTES,), 0xFF, dtype=torch.uint8, device=device)
        _UNREAD[device] = buffer
    return buffer.view(dtype)


def assert_not_capturing(what: str) -> None:
    """Kernels compile at build: raise if one would compile during capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{what} must be compiled before CUDA graph capture")


def _checked(result, operation="CUDA driver call"):
    """The values of a cuda-python driver result, raising on a failed status."""
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    return tuple(values)


def resolve_kernel(compiled, device):
    executor = compiled.to(device)
    library = compiled.library
    (count,) = _checked(cuda.cuLibraryGetKernelCount(library))
    if count != 1:  # every prepared module compiles one kernel
        raise RuntimeError(
            f"expected one kernel in the compiled library, found {count}"
        )
    (kernels,) = _checked(cuda.cuLibraryEnumerateKernels(count, library))
    (function,) = _checked(cuda.cuKernelGetFunction(kernels[0]))
    return executor, function


def check_residency(compiled, device_index: int, grid: int, threads: int, what: str):
    """Grids whose CTAs wait on one another (a Lamport publish-then-poll, not
    only within a cluster) must all be resident at once: raise when the device
    cannot hold ``grid`` CTAs of ``compiled`` (``threads`` per CTA, its dynamic
    shared memory) at the same time."""
    _, function = resolve_kernel(compiled, device_index)
    attribute = cuda.CUfunction_attribute
    max_dynamic = attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    (smem,) = _checked(cuda.cuFuncGetAttribute(max_dynamic, function))
    (per_sm,) = _checked(
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, threads, smem)
    )
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    if grid > per_sm * sms:
        raise RuntimeError(
            f"{what}: {grid} CTAs cannot be co-resident ({per_sm} per SM x {sms} SMs)"
        )
