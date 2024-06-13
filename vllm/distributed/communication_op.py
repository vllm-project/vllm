from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


@dataclass
class DistributedContext:
    communication_allowed: bool = True

    @staticmethod
    def get_current() -> "DistributedContext":
        """
        Get the singleton context.
        """
        global _default_context
        return _default_context


_default_context: DistributedContext = DistributedContext()


def disable_communication(fn):
    """
    Helper decorator to disable control plane communication, i.e.
    calling broadcast_tensor_dict will throw a RuntimeError. This can be used
    to ensure that decorated code stays worker-local.
    """

    def wrapper(*args, **kwargs):
        # Disallow control plane communication.
        comm_ctx = DistributedContext.get_current()
        original_comm_allowed = comm_ctx.communication_allowed
        comm_ctx.communication_allowed = False

        try:
            return fn(*args, **kwargs)
        finally:
            comm_ctx.communication_allowed = original_comm_allowed

    return wrapper


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    ctx = DistributedContext.get_current()
    if not ctx.communication_allowed:
        raise RuntimeError(
            "Control plane communication not allowed in functions decorated "
            "with @disable_communication")

    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
