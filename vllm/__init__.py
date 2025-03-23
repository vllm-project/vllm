# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import os
import torch
import importlib

# see https://github.com/NVIDIA/nccl/issues/1234
os.environ['NCCL_CUMEM_ENABLE'] = '0'
# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]

_lazy_modules = {
    "EngineArgs": "vllm.engine.arg_utils:EngineArgs",
    "AsyncEngineArgs": "vllm.engine.arg_utils:AsyncEngineArgs",
    "AsyncLLMEngine": "vllm.engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": "vllm.engine.llm_engine:LLMEngine",
    "LLM": "vllm.entrypoints.llm:LLM",
    "initialize_ray_cluster": "vllm.executor.ray_utils:initialize_ray_cluster",
    "PromptType": "vllm.inputs:PromptType",
    "TextPrompt": "vllm.inputs:TextPrompt",
    "TokensPrompt": "vllm.inputs:TokensPrompt",
    "ModelRegistry": "vllm.model_executor.models:ModelRegistry",
    "ClassificationOutput": "vllm.outputs:ClassificationOutput",
    "ClassificationRequestOutput": "vllm.outputs:ClassificationRequestOutput",
    "CompletionOutput": "vllm.outputs:CompletionOutput",
    "EmbeddingOutput": "vllm.outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": "vllm.outputs:EmbeddingRequestOutput",
    "PoolingOutput": "vllm.outputs:PoolingOutput",
    "PoolingRequestOutput": "vllm.outputs:PoolingRequestOutput",
    "RequestOutput": "vllm.outputs:RequestOutput",
    "ScoringOutput": "vllm.outputs:ScoringOutput",
    "ScoringRequestOutput": "vllm.outputs:ScoringRequestOutput",
    "SamplingParams": "vllm.sampling_params:SamplingParams",
    "PoolingParams": "vllm.pooling_params:PoolingParams",
}


def __getattr__(name):
    if name in _lazy_modules:
        module_spec = _lazy_modules[name]
        module_path, attr_name = module_spec.split(":")
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        return value
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return __all__ 
