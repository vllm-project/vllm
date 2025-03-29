# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import os

import torch

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/NVIDIA/nccl/issues/1234
os.environ['NCCL_CUMEM_ENABLE'] = '0'

# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

_lazy_imports_module_list = {
    "LLM": "vllm.entrypoints.llm.LLM",
    "ModelRegistry": "vllm.model_executor.models.ModelRegistry",
    "PromptType": "vllm.inputs.PromptType",
    "TextPrompt": "vllm.inputs.TextPrompt",
    "TokensPrompt": "vllm.inputs.TokensPrompt",
    "SamplingParams": "vllm.sampling_params.SamplingParams",
    "RequestOutput": "vllm.outputs.RequestOutput",
    "CompletionOutput": "vllm.outputs.CompletionOutput",
    "PoolingOutput": "vllm.outputs.PoolingOutput",
    "PoolingRequestOutput": "vllm.outputs.PoolingRequestOutput",
    "EmbeddingOutput": "vllm.outputs.EmbeddingOutput",
    "EmbeddingRequestOutput": "vllm.outputs.EmbeddingRequestOutput",
    "ClassificationOutput": "vllm.outputs.ClassificationOutput",
    "ClassificationRequestOutput": "vllm.outputs.ClassificationRequestOutput",
    "ScoringOutput": "vllm.outputs.ScoringOutput",
    "ScoringRequestOutput": "vllm.outputs.ScoringRequestOutput",
    "LLMEngine": "vllm.engine.llm_engine.LLMEngine",
    "EngineArgs": "vllm.engine.arg_utils.EngineArgs",
    "AsyncLLMEngine": "vllm.engine.async_llm_engine.AsyncLLMEngine",
    "AsyncEngineArgs": "vllm.engine.arg_utils.AsyncEngineArgs",
    "initialize_ray_cluster": "vllm.executor.ray_utils.initialize_ray_cluster",
    "PoolingParams": "vllm.pooling_params.PoolingParams",
}


def __getattr__(name: str):
    if name in _lazy_imports_module_list:
        import importlib
        module_path, attr = _lazy_imports_module_list[name].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
