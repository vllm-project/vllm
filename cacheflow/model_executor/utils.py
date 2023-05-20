"""Utils for model executor."""
import random

import numpy as np
import torch

from cacheflow.model_executor.parallel_utils.parallel_state import model_parallel_is_initialized
from cacheflow.model_executor.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)


def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def get_cache_block_size(
    block_size: int,
    num_heads: int,
    head_size: int,
    num_layers: int,
    dtype: torch.dtype,
) -> int:
    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_layers * (key_cache_block + value_cache_block)
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
