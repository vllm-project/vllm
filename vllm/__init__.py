"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

import dataclasses

import torch

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.model_executor.models import ModelRegistry
from vllm.outputs import (CompletionOutput, EmbeddingOutput,
                          EmbeddingRequestOutput, RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

torch_dtypes = [
    getattr(torch, attr) for attr in dir(torch)
    if isinstance(getattr(torch, attr), torch.dtype)
]
dtype_map = {dtype: i for i, dtype in enumerate(torch_dtypes)}


@dataclasses.dataclass
class TensorMeta:
    """
    This class is placed here to reduce the size of qualified name,
    which will be used in pickle serialization.
    """
    device: str
    dtype: torch.dtype
    size: torch.Size

    def __getstate__(self):
        return [self.device, dtype_map[self.dtype], tuple(self.size)]

    def __setstate__(self, state):
        self.device = state[0]
        self.dtype = torch_dtypes[state[1]]
        self.size = torch.Size(state[2])


__version__ = "0.4.2"

__all__ = [
    "LLM",
    "ModelRegistry",
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
    "TensorMeta",
]
