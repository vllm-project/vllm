# communicator interface, as proposed in
# https://github.com/vllm-project/vllm/issues/3587
# `Communicator` is responsible for communicating **large tensor data**
# between multiple devices. This functionality is usually provided by
# vendors, e.g. NCCL from NVIDIA, RCCL from AMD.
# Put it simple, this is for data-plane communication.

from typing import Any, Optional

import torch
from torch.distributed import ReduceOp

from vllm.interfaces.coordinator import Coordinator


class Communicator(object):
    """
    `coordinator` is the object used to initialize the communicator.
    `group` is the list of **global** ranks to identify communication groups.
    For functions with a `src` or `dst` argument, the rank is also global.
    If the communicator needs to know the local rank inside a group, it should
    convert the rank by searching over the group.

    The interfaces are designed to be as general as possible. They contain
    out-of-place operations, and stream arguments to launch the operations.
    However, subclasses are free to implement only in-place operations without
    the stream argument, i.e. raising NotImplementedError for the out-of-place
    operations and not-None stream argument, as long as they satisfy the
    requirements of the corresponding application.
    """

    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        assert self.coordinator.is_initialized()

    def broadcast(self,
                  tensor_in: torch.Tensor,
                  tensor_out: Optional[torch.Tensor] = None,
                  src: int = 0,
                  stream: Optional[Any] = None):
        raise NotImplementedError(
            f"broadcast is not implemented in {self.__class__.__name__}")

    def all_reduce(self,
                   tensor_in: torch.Tensor,
                   tensor_out: Optional[torch.Tensor] = None,
                   op: ReduceOp = ReduceOp.SUM,
                   stream: Optional[Any] = None):
        raise NotImplementedError(
            f"all_reduce is not implemented in {self.__class__.__name__}")

    def reduce(self,
               tensor_in: torch.Tensor,
               tensor_out: Optional[torch.Tensor] = None,
               dst: int = 0,
               op: ReduceOp = ReduceOp.SUM,
               stream: Optional[Any] = None):
        raise NotImplementedError(
            f"reduce is not implemented in {self.__class__.__name__}")

    def all_gather(self,
                   tensor_in: torch.Tensor,
                   tensor_out: Optional[torch.Tensor] = None,
                   stream: Optional[Any] = None):
        raise NotImplementedError(
            f"all_gather is not implemented in {self.__class__.__name__}")

    def __del__(self):
        pass
