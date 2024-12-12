"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
from vllm.model_executor.models import ModelRegistry
from vllm.outputs import (CompletionOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from .version import __version__, __version_tuple__

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
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]


def __getattr__(name: str):
    import warnings

    if name == "EmbeddingOutput":
        msg = ("EmbeddingOutput has been renamed to PoolingOutput. "
               "The original name will be removed in an upcoming version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return PoolingOutput

    if name == "EmbeddingRequestOutput":
        msg = ("EmbeddingRequestOutput has been renamed to "
               "PoolingRequestOutput. "
               "The original name will be removed in an upcoming version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return PoolingRequestOutput

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
