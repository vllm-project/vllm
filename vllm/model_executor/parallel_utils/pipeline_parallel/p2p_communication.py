from typing import List, Optional, Union

import torch
import torch.distributed as dist

from vllm.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_next_rank)

Shape = Union[List[int], torch.Size]


def send_to_next_pp_rank(tensor: torch.Tensor) -> None:
    dist.send(tensor, get_pipeline_model_parallel_next_rank())


def receive_from_prev_pp_rank(
    tensor_shape: Shape,
    tensor_dtype: torch.dtype,
    tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tensor is None:
        tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device='cuda')
    dist.recv(tensor, get_pipeline_model_parallel_prev_rank())
    return tensor
