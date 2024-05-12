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


@dataclasses.dataclass
class TensorMeta:
    """
    This class is placed here to reduce the size of qualified name,
    which will be used in pickle serialization.
    """
    device: str
    dtype: torch.dtype
    size: torch.Size

    torch_dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32,
        torch.float64, torch.bfloat16
    ]
    dtype_map = {dtype: i for i, dtype in enumerate(torch_dtypes)}

    def __getstate__(self):
        return [self.device, self.dtype_map[self.dtype], tuple(self.size)]

    def __setstate__(self, state):
        self.device = state[0]
        self.dtype = self.torch_dtypes[state[1]]
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
