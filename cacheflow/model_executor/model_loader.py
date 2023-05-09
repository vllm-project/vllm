from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import PretrainedConfig

from cacheflow.model_executor.memory_analyzer import (
    CacheFlowMemoryAnalyzer, GPT2MemoryAnalyzer, GPTNeoXMemoryAnalyzer,
    LlamaMemoryAnalyzer, OPTMemoryAnalyzer)
from cacheflow.model_executor.models import (
    GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM, OPTForCausalLM)
from cacheflow.model_executor.utils import get_torch_dtype
from cacheflow.model_executor.weight_utils import initialize_dummy_weights


# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}

_MEMORY_ANALYZERS = {
    "GPT2LMHeadModel": GPT2MemoryAnalyzer,
    "GPTNeoXForCausalLM": GPTNeoXMemoryAnalyzer,
    "LlamaForCausalLM": LlamaMemoryAnalyzer,
    "OPTForCausalLM": OPTMemoryAnalyzer,
}


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}"
    )


def _get_memory_analyzer(config: PretrainedConfig) -> CacheFlowMemoryAnalyzer:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MEMORY_ANALYZERS:
            return _MEMORY_ANALYZERS[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MEMORY_ANALYZERS.keys())}"
    )


def _get_dtype(config: PretrainedConfig, dtype: str) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32
    if dtype == "default":
        if config_dtype == torch.float32:
            # Following the common practice, we use float16 for float32 models.
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        torch_dtype = get_torch_dtype(dtype)
        if torch_dtype != config_dtype and config_dtype != torch.float32:
            # TODO(woosuk): Allow using float16 for bfloat16 models and
            # vice versa. Print a warning message and continue.
            raise ValueError(
                f"Cannot use {torch_dtype} for {config_dtype} model.")
    return torch_dtype


def get_model(
    model_name: str,
    dtype: str,
    cache_dir: Optional[str],
    use_dummy_weights: bool,
    use_np_cache: bool,
) -> nn.Module:
    config = AutoConfig.from_pretrained(model_name)
    torch_dtype = _get_dtype(config, dtype)
    torch.set_default_dtype(torch_dtype)
    model_class = _get_model_architecture(config)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(config)
    if use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_name, cache_dir, use_np_cache)
        model = model.cuda()
    return model.eval(), torch_dtype


def get_memory_analyzer(
    model_name: str,
    block_size: int,
    dtype: str,
    gpu_memory: int,
    cpu_memory: int,
    tensor_parallel_size: int = 1,
) -> CacheFlowMemoryAnalyzer:
    config = AutoConfig.from_pretrained(model_name)
    torch_dtype = _get_dtype(config, dtype)
    memory_analyzer = _get_memory_analyzer(config)
    return memory_analyzer(
        model_name, block_size, torch_dtype, gpu_memory, cpu_memory,
        tensor_parallel_size)
