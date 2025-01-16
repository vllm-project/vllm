"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

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

from .version import __version__, __version_tuple__


def configure_as_vllm_process():
    """
    set some common config/environment variables that should be set
    for all processes created by vllm and all processes
    that interact with vllm workers.
    """
    import os

    import torch

    # see https://github.com/NVIDIA/nccl/issues/1234
    os.environ['NCCL_CUMEM_ENABLE'] = '0'

    # see https://github.com/vllm-project/vllm/issues/10480
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # see https://github.com/vllm-project/vllm/issues/10619
    torch._inductor.config.compile_threads = 1

    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        # see https://github.com/pytorch/pytorch/blob/43c5f59/torch/_dynamo/config.py#L158
        torch._dynamo.config.disable = True
    elif current_platform.is_hpu():
        # NOTE(kzawora): PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '1') == '1'
        if is_lazy:
            torch._dynamo.config.disable = True
            # NOTE(kzawora) multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html # noqa: E501
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'


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
    "configure_as_vllm_process",
]
