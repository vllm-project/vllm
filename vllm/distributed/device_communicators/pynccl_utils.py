import contextlib
from typing import Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from vllm.distributed.device_communicators.pynccl import NCCLCommunicator
except Exception as e:
    # in non-NVIDIA environments, we can't import the nccl module
    # e.g. when running on machines with AMD GPUs
    logger.info("Failed to import NCCL library: %s", e)
    logger.info("It is expected if you are not running on NVIDIA GPUs.")
    pass

comm: Optional["NCCLCommunicator"] = None


@contextlib.contextmanager
def set_pynccl_stream(stream: torch.cuda.Stream):
    """Set the cuda stream for communication"""
    try:
        assert comm is not None
        comm.stream = stream
        yield
    finally:
        pass


def all_reduce(input_: torch.Tensor, op=ReduceOp.SUM) -> None:
    """All-reduces the input tensor across the process group."""
    assert input_.is_cuda, f"{input_} should be a cuda tensor"
    assert comm is not None
    comm.all_reduce(input_, op)


def get_world_size() -> int:
    """Returns the world size."""
    assert comm is not None
    return comm.world_size


def get_nccl_backend() -> Optional["NCCLCommunicator"]:
    return comm
