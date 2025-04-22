# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # isort:skip  # noqa: F401


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


def __getattr__(name):
    if name == "LLM":
        from vllm.entrypoints.llm import LLM
        return LLM
    elif name == "ModelRegistry":
        from vllm.model_executor.models import ModelRegistry
        return ModelRegistry
    elif name == "PromptType":
        from vllm.inputs import PromptType
        return PromptType
    elif name == "TextPrompt":
        from vllm.inputs import TextPrompt
        return TextPrompt
    elif name == "TokensPrompt":
        from vllm.inputs import TokensPrompt
        return TokensPrompt
    elif name == "SamplingParams":
        from vllm.sampling_params import SamplingParams
        return SamplingParams
    elif name == "RequestOutput":
        from vllm.outputs import RequestOutput
        return RequestOutput
    elif name == "CompletionOutput":
        from vllm.outputs import CompletionOutput
        return CompletionOutput
    elif name == "EmbeddingOutput":
        from vllm.outputs import EmbeddingOutput
        return EmbeddingOutput
    elif name == "PoolingOutput":
        from vllm.outputs import PoolingOutput
        return PoolingOutput
    elif name == "PoolingRequestOutput":
        from vllm.outputs import PoolingRequestOutput
        return PoolingRequestOutput
    elif name == "EmbeddingRequestOutput":
        from vllm.outputs import EmbeddingRequestOutput
        return EmbeddingRequestOutput
    elif name == "ClassificationOutput":
        from vllm.outputs import ClassificationOutput
        return ClassificationOutput
    elif name == "ClassificationRequestOutput":
        from vllm.outputs import ClassificationRequestOutput
        return ClassificationRequestOutput
    elif name == "ScoringOutput":
        from vllm.outputs import ScoringOutput
        return ScoringOutput
    elif name == "ScoringRequestOutput":
        from vllm.outputs import ScoringRequestOutput
        return ScoringRequestOutput
    elif name == "LLMEngine":
        from vllm.engine.llm_engine import LLMEngine
        return LLMEngine
    elif name == "EngineArgs":
        from vllm.engine.arg_utils import EngineArgs
        return EngineArgs
    elif name == "AsyncLLMEngine":
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        return AsyncLLMEngine
    elif name == "AsyncEngineArgs":
        from vllm.engine.arg_utils import AsyncEngineArgs
        return AsyncEngineArgs
    elif name == "initialize_ray_cluster":
        from vllm.executor.ray_utils import initialize_ray_cluster
        return initialize_ray_cluster
    elif name == "PoolingParams":
        from vllm.pooling_params import PoolingParams
        return PoolingParams
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
