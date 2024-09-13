
import numpy as np
import pytest
import random
from typing import (Any, Callable, Dict, List, Optional, Tuple, TypedDict,
                    TypeVar, Union)

import gc
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_cpu
from transformers import BatchEncoding, BatchFeature
import torch
import torch.nn as nn
from vllm.wde.entrypoints.llm import LLM

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


def cleanup():
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()


class HfRunner:
    def wrap_device(self, input: _T) -> _T:
        if not is_cpu():
            # Check if the input is already on the GPU
            if hasattr(input, 'device') and input.device.type == "cuda":
                return input  # Already on GPU, no need to move
            return input.to("cuda")
        else:
            # Check if the input is already on the CPU
            if hasattr(input, 'device') and input.device.type == "cpu":
                return input  # Already on CPU, no need to move
            return input.to("cpu")

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
    ) -> None:
        # depend on FlagEmbedding peft
        from FlagEmbedding import BGEM3FlagModel
        self.model_name = model_name
        model = BGEM3FlagModel(self.model_name, use_fp16=dtype == "half")
        self.model = model

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        output = self.model.encode(prompts)
        return output['dense_vecs']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class VllmRunner:
    def __init__(
        self,
        model_name: str,
        max_num_seqs: int = 4,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            dtype=dtype)

    def encode(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.encode(prompts)
        outputs = []
        for req_output in req_outputs:
            embedding = req_output.outputs
            outputs.append(embedding)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


@pytest.fixture(scope="session")
def example_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 11
    random.shuffle(prompts)
    return prompts


MODELS = [
    'BAAI/bge-m3'
]


def compare_embeddings_np(embeddings1, embeddings2):
    similarities = [
        e1 @ e2.T
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_seqs", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync", "async", "double_buffer"])
@torch.inference_mode
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_seqs: int,
    scheduling: str
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(model, dtype=dtype, max_num_seqs=max_num_seqs) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)
        vllm_outputs = [t.cpu().numpy() for t in vllm_outputs]

    similarities = compare_embeddings_np(hf_outputs, vllm_outputs)
    all_similarities = np.stack(similarities)
    tolerance = 1e-2
    assert np.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
