from typing import Union

import torch
import torch.nn as nn

from cacheflow.models.memory_analyzer import CacheFlowMemoryAnalyzer
from cacheflow.models.memory_analyzer import OPTMemoryAnalyzer
from cacheflow.models.opt import OPTForCausalLM
from cacheflow.models.utils import get_torch_dtype


_MODELS = {
    'opt': OPTForCausalLM,
}

_MEMORY_ANALYZERS = {
    'opt': OPTMemoryAnalyzer,
}


def get_model(
    model_name: str,
    dtype: Union[torch.dtype, str],
) -> nn.Module:
    torch_dtype = get_torch_dtype(dtype)
    for model_class, hf_model in _MODELS.items():
        if model_class in model_name:
            model = hf_model.from_pretrained(
                model_name, torch_dtype=torch_dtype)
            return model.eval()
    raise ValueError(f'Unsupported model name: {model_name}')


def get_memory_analyzer(
    model_name: str,
    block_size: int,
    dtype: Union[torch.dtype, str],
) -> CacheFlowMemoryAnalyzer:
    torch_dtype = get_torch_dtype(dtype)
    for model_class, memory_analyzer in _MEMORY_ANALYZERS.items():
        if model_class in model_name:
            return memory_analyzer(
                model_name, block_size, torch_dtype)
    raise ValueError(f'Unsupported model name: {model_name}')
