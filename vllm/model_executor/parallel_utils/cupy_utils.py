"""CuPy utilities for all-reduce.

We use CuPy all-reduce instead of torch.distributed.all_reduce when capturing
CUDA graphs, because torch.distributed.all_reduce causes errors when capturing
CUDA graphs.

NOTE: We use CuPy 12.3 since CuPy 13.0 does not support Python 3.8.
TODO: Remove this file when torch.distributed.all_reduce is fixed.
"""
import contextlib

import torch
from torch.distributed import ReduceOp

try:
    import cupy
    from cupy.cuda import nccl
    from cupyx.distributed import NCCLBackend
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


class NCCLBackendWithBFloat16(NCCLBackend):
    # This is enough to add bfloat16 support for most operations,
    # but broadcast will fail (will require changes in compiled
    # cupy code).
    def _get_nccl_dtype_and_count(self, array, count=None):
        nccl_dtype, count = super()._get_nccl_dtype_and_count(array, count)
        torch_dtype = getattr(array, "_torch_dtype", None)
        if torch_dtype is torch.bfloat16:
            nccl_dtype = nccl.NCCL_BFLOAT16
        return nccl_dtype, count

    def barrier(self) -> None:
        raise RuntimeError(
            "Currently, CuPy NCCL barrier is not supported since the TCP "
            "store is immediately stopped after the initialization.")


_NCCL_BACKEND = None
_WORLD_SIZE = 0


def is_initialized() -> bool:
    """Returns whether the NCCL backend is initialized."""
    return _NCCL_BACKEND is not None


@contextlib.contextmanager
def set_cupy_stream(stream: torch.cuda.Stream):
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
        raise ImportError(
            "NCCLBackend is not available. Please install cupy.") from cupy

    # TODO(woosuk): Create TP and PP process groups for CuPy.
    global _NCCL_BACKEND
    global _WORLD_SIZE
    assert world_size > 0, f"{world_size=} should be a positive integer"
    assert 0 <= rank < world_size, (
        f"{rank=} should be a integer between [0, {world_size})")

    cupy.cuda.runtime.setDevice(torch.cuda.current_device())
    _NCCL_BACKEND = NCCLBackendWithBFloat16(world_size, rank, host, port)
    _WORLD_SIZE = world_size

    # Stop the TCP store to prevent the deadlock issues at termination time.
    # FIXME(woosuk): This is hacky. Find a more robust solution.
    if rank == 0 and hasattr(_NCCL_BACKEND, "_store"):
        _NCCL_BACKEND._store.stop()


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
    cupy_input = cupy.asarray(input_)
    cupy_input._torch_dtype = torch_dtype  # pylint: disable=protected-access
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


def get_nccl_backend():
    return _NCCL_BACKEND
