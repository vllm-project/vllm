# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group

from vllm.multistream.context import get_multistream_comm_context

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    current_ms_metadata = get_multistream_comm_context()
    if current_ms_metadata is None:
        output =  get_tp_group().all_reduce(input_)
    else:
        current_ms_metadata.before_comm_event.record()
        with torch.cuda.stream(current_ms_metadata.comm_stream):
            current_ms_metadata.before_comm_event.wait()
            output = get_tp_group().all_reduce(input_)
            current_ms_metadata.after_comm_event.record()
    return output


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
