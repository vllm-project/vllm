from typing import Union

import random

import numpy as np
import psutil
import torch

_STR_DTYPE_TO_TORCH_DTYPE = {
    'half': torch.half,
    'float': torch.float,
    'float16': torch.float16,
    'float32': torch.float32,
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


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    return psutil.virtual_memory().total
