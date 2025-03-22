# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

from .version import __version__, __version_tuple__  # isort:skip
from typing import TYPE_CHECKING
import torch
import os

from vllm.utils import LazyLoader

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

if TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.executor.ray_utils import initialize_ray_cluster
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (ClassificationOutput, ClassificationRequestOutput,
                            CompletionOutput, EmbeddingOutput,
                            EmbeddingRequestOutput, PoolingOutput,
                            PoolingRequestOutput, RequestOutput, ScoringOutput,
                            ScoringRequestOutput)
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
else:
    EngineArgs = LazyLoader("EngineArgs", globals(), "vllm.engine.arg_utils")
    AsyncEngineArgs = LazyLoader("AsyncEngineArgs", globals(), "vllm.engine.arg_utils")
    AsyncLLMEngine = LazyLoader("AsyncLLMEngine", globals(), "vllm.engine.async_llm_engine")
    LLMEngine = LazyLoader("LLMEngine", globals(), "vllm.engine.llm_engine")
    LLM = LazyLoader("LLM", globals(), "vllm.entrypoints.llm")
    initialize_ray_cluster = LazyLoader("initialize_ray_cluster", globals(), "vllm.executor.ray_utils")
    PromptType = LazyLoader("PromptType", globals(), "vllm.inputs")
    TextPrompt = LazyLoader("TextPrompt", globals(), "vllm.inputs")
    TokensPrompt = LazyLoader("TokensPrompt", globals(), "vllm.inputs")
    ModelRegistry = LazyLoader("ModelRegistry", globals(), "vllm.model_executor.models")
    ClassificationOutput = LazyLoader("ClassificationOutput", globals(), "vllm.outputs")
    ClassificationRequestOutput = LazyLoader("ClassificationRequestOutput", globals(), "vllm.outputs")
    CompletionOutput = LazyLoader("CompletionOutput", globals(), "vllm.outputs")
    EmbeddingOutput = LazyLoader("EmbeddingOutput", globals(), "vllm.outputs")
    EmbeddingRequestOutput = LazyLoader("EmbeddingRequestOutput", globals(), "vllm.outputs")
    PoolingOutput = LazyLoader("PoolingOutput", globals(), "vllm.outputs")
    PoolingRequestOutput = LazyLoader("PoolingRequestOutput", globals(), "vllm.outputs")
    RequestOutput = LazyLoader("RequestOutput", globals(), "vllm.outputs")
    ScoringOutput = LazyLoader("ScoringOutput", globals(), "vllm.outputs")
    ScoringRequestOutput = LazyLoader("ScoringRequestOutput", globals(), "vllm.outputs")
    SamplingParams = LazyLoader("SamplingParams", globals(), "vllm.sampling_params")
    PoolingParams = LazyLoader("PoolingParams", globals(), "vllm.pooling_params")