import pytest
import random
from typing import TypeVar
import itertools as it
from transformers import BatchEncoding, BatchFeature
import torch
import torch.nn as nn
from tests.wde.utils import VllmRunner, compare_embeddings

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


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


@pytest.fixture(scope="session")
def attention_impls():
    return ["FLASH_ATTN", "TORCH_SDPA"]


MODELS = ['BAAI/bge-m3']


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_seqs", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync"])
@torch.inference_mode
def test_basic_correctness(hf_runner, vllm_runner, example_prompts, model: str,
                           dtype: str, max_num_seqs: int, scheduling: str,
                           attention_impls: list[str]) -> None:
    impl_outputs_list = []

    for attention_impl in attention_impls:
        with vllm_runner(model,
                         dtype=dtype,
                         max_num_seqs=max_num_seqs,
                         scheduling=scheduling,
                         attention_impl=attention_impl) as vllm_model:
            vllm_outputs = vllm_model.encode(example_prompts)
            impl_outputs_list.append((attention_impl, vllm_outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all((all_similarities <= 1.0 + tolerance)
                         & (all_similarities >= 1.0 - tolerance)), \
            f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
