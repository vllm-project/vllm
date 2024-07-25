"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

import os

# Set OMP_NUM_THREADS to 1 if it is not already set
# Helps to avoid CPU contention, particularly when running in a container and
# when using multiprocessing.
#
# This must be set before importing torch to have the full effect. It seems that
# torch determines the number of threads it will use when it is imported.
if "OMP_NUM_THREADS" not in os.environ:
    print(
        "WARNING: Setting OMP_NUM_THREADS env var to 1 by default to avoid "
        "unnecessary CPU contention. Set in the external environment to tune "
        "this value for optimal performance.")
    os.environ["OMP_NUM_THREADS"] = "1"

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import PromptInputs, TextPrompt, TokensPrompt
from vllm.model_executor.models import ModelRegistry
from vllm.outputs import (CompletionOutput, EmbeddingOutput,
                          EmbeddingRequestOutput, RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from .version import __commit__, __version__

__all__ = [
    "__commit__",
    "__version__",
    "LLM",
    "ModelRegistry",
    "PromptInputs",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
