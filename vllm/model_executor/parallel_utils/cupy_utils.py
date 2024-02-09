"""CuPy utilities for all-reduce.

We use CuPy all-reduce instead of torch.distributed.all_reduce when capturing
CUDA graphs, because torch.distributed.all_reduce causes errors when capturing
CUDA graphs.

TODO: Remove this file when torch.distributed.all_reduce is fixed.
"""
import contextlib
from unittest.mock import patch

import torch
from torch.distributed import ReduceOp

try:
    import cupy
    from cupy.cuda import nccl
    from cupyx.distributed._nccl_comm import NCCLBackend, _get_nccl_dtype_and_count
except ImportError as e:
    cupy = e
    nccl = None

    class NCCLBackend:
        ...


_OP_MAPPING = {
    ReduceOp.SUM: "sum",
    ReduceOp.PRODUCT: "prod",
    ReduceOp.MIN: "min",
    ReduceOp.MAX: "max",
}

_NCCL_BACKEND = None
_WORLD_SIZE = 0

_get_nccl_dtype_and_count_orginal = _get_nccl_dtype_and_count


def _get_nccl_dtype_and_count_bf16(*args, **kwargs):
    """Patch/hack to force bf16 dtype in cupy NCCL.

    cupy doesn't support bf16 by default, but the underlying NCCL
    kernels do. We can just force the dtype to be bf16 and it will
    work fine."""
    dtype, count = _get_nccl_dtype_and_count_orginal(*args, **kwargs)
    # Hardcoded to always return bf16 dtype
    dtype = nccl.NCCL_BFLOAT16
    return dtype, count


def is_initialized() -> bool:
    """Returns whether the NCCL backend is initialized."""
    return _NCCL_BACKEND is not None


@contextlib.contextmanager
def set_cupy_stream(stream: torch.cuda.Stream) -> None:
    """Set the cuda stream for communication"""
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream,
                                           stream.device_index)
    with cupy_stream:
        yield


def init_process_group(world_size: int, rank: int, host: str,
                       port: int) -> None:
    """Initializes the CuPy NCCL backend.

    # TODO: handle NCCL timeouts.
    """
    assert not is_initialized()

    if isinstance(cupy, Exception):
        raise ImportError("NCCLBackend is not available. Please install "
                          "cupy-cuda12x==13.0.0.") from cupy

    # TODO(woosuk): Create TP and PP process groups for CuPy.
    global _NCCL_BACKEND
    global _WORLD_SIZE
    assert world_size > 0, f"{world_size=} should be a positive integer"
    assert 0 <= rank < world_size, (
        f"{rank=} should be a integer between [0, {world_size})")

    cupy.cuda.runtime.setDevice(torch.cuda.current_device())
    _NCCL_BACKEND = NCCLBackend(world_size, rank, host, port)
    _WORLD_SIZE = world_size


def all_reduce(input_: torch.Tensor, op=ReduceOp.SUM) -> None:
    """All-reduces the input tensor across the process group."""
    assert input_.is_cuda, f"{input_} should be a cuda tensor"
    # Hack to support bfloat16
    torch_dtype = input_.dtype
    if torch_dtype is torch.bfloat16:
        # We need to view as float16, otherwise
        # cupy will fail. This will not change
        # the underlying data.
        input_ = input_.view(torch.float16)
        maybe_patch_cupy_bf16 = patch(
            "cupyx.distributed._nccl_comm._get_nccl_dtype_and_count",
            _get_nccl_dtype_and_count_bf16)
    else:
        maybe_patch_cupy_bf16 = contextlib.nullcontext()
    cupy_input = cupy.asarray(input_)
    with maybe_patch_cupy_bf16:
        _NCCL_BACKEND.all_reduce(in_array=cupy_input,
                                 out_array=cupy_input,
                                 op=_OP_MAPPING[op])


def destroy_process_group() -> None:
    """Destroys the NCCL backend."""
    global _NCCL_BACKEND
    global _WORLD_SIZE
    _NCCL_BACKEND = None
    _WORLD_SIZE = 0


def get_world_size() -> int:
    """Returns the world size."""
    return _WORLD_SIZE
