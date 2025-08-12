from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig

from cacheflow.models.memory_analyzer import CacheFlowMemoryAnalyzer
from cacheflow.models.memory_analyzer import LlamaMemoryAnalyzer
from cacheflow.models.memory_analyzer import OPTMemoryAnalyzer
from cacheflow.models.llama import LlamaForCausalLM
from cacheflow.models.opt import OPTForCausalLM
from cacheflow.models.utils import get_torch_dtype


_MODELS = {
    'llama': LlamaForCausalLM,
    'opt': OPTForCausalLM,
}

_MEMORY_ANALYZERS = {
    'llama': LlamaMemoryAnalyzer,
    'opt': OPTMemoryAnalyzer,
}


def get_model(
    model_name: str,
    dtype: Union[torch.dtype, str],
    path: str,
    use_dummy_weights: bool,
) -> nn.Module:
    torch_dtype = get_torch_dtype(dtype)
    torch.set_default_dtype(torch_dtype)
    config = AutoConfig.from_pretrained(model_name)
    for model_class_name, model_class in _MODELS.items():
        if model_class_name in model_name:
            if use_dummy_weights:
                # Create a model instance.
                # The weights will be initialized as empty tensors.
                model = model_class(config)
                model = model.cuda()
                # NOTE(woosuk): For precise performance evaluation, we assign
                # random values to the weights. 
                model.initialize_dummy_weights()
            else:
                # Download model weights if it's not cached.
                weights_dir = model_class.get_weights(model_name, path=path)
                # Create a model instance.
                model = model_class(config)
                # Load the weights from the cached or downloaded files.
                model.load_weights(weights_dir)
                model = model.cuda()
            return model.eval(), torch_dtype
    raise ValueError(f'Unsupported model name: {model_name}')


def get_memory_analyzer(
    model_name: str,
    block_size: int,
    dtype: Union[torch.dtype, str],
    gpu_memory: int,
    cpu_memory: int,
    tensor_parallel_size: int = 1,
) -> CacheFlowMemoryAnalyzer:
    torch_dtype = get_torch_dtype(dtype)
    for model_class, memory_analyzer in _MEMORY_ANALYZERS.items():
        if model_class in model_name:
            return memory_analyzer(
                model_name, block_size, torch_dtype, gpu_memory, cpu_memory,
                tensor_parallel_size)
    raise ValueError(f'Unsupported model name: {model_name}')
