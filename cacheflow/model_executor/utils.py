"""Utils for model executor."""
import random
from typing import Union

import numpy as np
import torch

from cacheflow.model_executor.parallel_utils.parallel_state import model_parallel_is_initialized
from cacheflow.model_executor.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "float": torch.float,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, str):
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype.lower()]
    else:
        torch_dtype = dtype
    return torch_dtype


def get_dtype_size(dtype: Union[torch.dtype, str]) -> int:
    torch_dtype = get_torch_dtype(dtype)
    return torch.tensor([], dtype=torch_dtype).element_size()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)
