# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import os

import torch

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
