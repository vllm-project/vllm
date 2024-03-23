from .pynccl import NCCLCommunicator, ncclGetVersion

import contextlib
import logging
import torch

from typing import Optional
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)

logger.info(f"vLLM is using nccl=={ncclGetVersion()}")

comm: Optional[NCCLCommunicator] = None


def is_initialized() -> bool:
    """Returns whether the NCCL backend is initialized."""
    return comm is not None


@contextlib.contextmanager
def set_cupy_stream(stream: torch.cuda.Stream):
    """Set the cuda stream for communication"""
    try:
        comm.stream = stream
        yield
    finally:
        pass


def init_process_group(world_size: int, rank: int, host: str,
                       port: int) -> None:
    assert not is_initialized()
    global comm
    comm = NCCLCommunicator(init_method=f"tcp://{host}:{port}",
                            world_size=world_size,
                            rank=rank)


def all_reduce(input_: torch.Tensor, op=ReduceOp.SUM) -> None:
    """All-reduces the input tensor across the process group."""
    assert input_.is_cuda, f"{input_} should be a cuda tensor"
    comm.all_reduce(input_, op)


def destroy_process_group() -> None:
    global comm
    comm = None


def get_world_size() -> int:
    """Returns the world size."""
    return comm.world_size


def get_nccl_backend():
    return comm
