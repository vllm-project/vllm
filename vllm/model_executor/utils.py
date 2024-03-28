"""Utils for model executor."""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)


def distribute_weights(param_data: torch.Tensor,
                       loaded_weight: Optional[torch.Tensor],
                       weight_owner: int,
                       use_scatter: bool,
                       shard_size: Optional[int] = None,
                       split_dim: Optional[int] = None) -> None:
    """Distribute weights from rank 'weight_owner' to other tensor parallel 
    ranks.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    tp_group = get_tensor_model_parallel_group()
    if use_scatter:  # use torch.distributed.scatter
        assert shard_size is not None and split_dim is not None, \
            "shard_size and split_dim must be specified when using scatter"
        if tp_rank == torch.distributed.get_group_rank(tp_group, weight_owner):
            # If split_dim > 0, the split weights is not contiguous.
            loaded_weight_list = [
                weight.contiguous().to(param_data.dtype) for weight in
                torch.split(loaded_weight, shard_size, dim=split_dim)
            ]
            assert len(loaded_weight_list) == tp_size
            assert param_data.shape == loaded_weight_list[0].shape
        else:
            loaded_weight_list = None
        torch.distributed.scatter(param_data, loaded_weight_list, weight_owner,
                                  tp_group)
    else:  # use torch.distributed.broadcast
        if tp_rank == torch.distributed.get_group_rank(tp_group, weight_owner):
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
        torch.distributed.broadcast(param_data, weight_owner, tp_group)
