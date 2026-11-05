# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side helpers for the prepared CuTe providers."""

import ctypes
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_tensor

from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute

_NAMES = ("BFloat16", "Float32", "Uint8", "Uint32", "Int32", "Int64")
_DTYPES = {getattr(torch, name.lower()): getattr(cutlass, name) for name in _NAMES}


def layout(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def stream(device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


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


def tensor_maps(specs, device):
    """TMA descriptors (128 B each) in one device tensor.
    Each spec: (pointer, dtype, dims, strides, box, swizzle, l2 promotion).

    The bytes are uploaded with a copy on the current stream: the new tensor may
    reuse a block freed by a kernel still pending on that stream (e.g. a warmup
    launch's output), which a copy outside the stream would race with."""
    encoded = bytearray()
    with torch.accelerator.device_index(torch.device(device).index):
        for pointer, dtype, dims, strides, box, swizzle, l2 in specs:
            u64 = ([cuda.cuuint64_t(v) for v in values] for values in (dims, strides))
            u32 = ([cuda.cuuint32_t(v) for v in box], [cuda.cuuint32_t(1)] * len(dims))
            interleave = cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
            fill = cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            status, tensor_map = cuda.cuTensorMapEncodeTiled(
                dtype, len(dims), pointer, *u64, *u32, interleave, swizzle, l2, fill
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuTensorMapEncodeTiled failed: {status}")
            encoded += (ctypes.c_ubyte * 128).from_address(tensor_map.getPtr())
    return torch.frombuffer(encoded, dtype=torch.uint8).to(device)


def _checked(result, operation="CUDA driver call"):
    """The values of a cuda-python driver result, raising on a failed status."""
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    return tuple(values)


class PreparedLaunchAttributes:
    def __init__(self, compiled, executor):
        self.compiled, self.executor = compiled, executor

    def __call__(self, *args, **kwargs):
        return self.compiled(*args, **kwargs)


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
    """Persistent grids whose CTAs wait on one another (not only within a cluster) must
    all be resident at once: raise when the device cannot hold ``grid`` CTAs of
    ``compiled`` (``threads`` per CTA, its dynamic shared memory) at the same time."""
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


def prepare_shared_carveout(compiled, device_index: int, value: int):
    """Restore ``value`` (CuTe's launcher drops preferred_smem_carveout) as a driver
    function attribute; call once before capture."""
    executor, function = resolve_kernel(compiled, device_index)
    attribute = cuda.CUfunction_attribute
    attribute = attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    _checked(cuda.cuFuncSetAttribute(function, attribute, value))
    return PreparedLaunchAttributes(compiled, executor)
