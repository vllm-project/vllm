import itertools as it
import random
from typing import List, TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature

from tests.wde.utils import HfRunner, VllmRunner, compare_embeddings

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class Qwen2HfRunner(HfRunner):

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        outputs = self.model(**encoded_input, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        seq_len = encoded_input.attention_mask.sum(axis=1)

        last_hidden_states_list = []
        for e, s in zip(last_hidden_states, seq_len):
            last_hidden_states_list.append(e[s - 1])
        return last_hidden_states_list


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


MODELS = ["Qwen/Qwen2-0.5B-Instruct"]

AttentionImpls_fp32 = ["TORCH_SDPA", "XFORMERS", "TORCH_NAIVE"]
AttentionImpls_fp16 = [
    "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
]
AttentionImpls_bf16 = [
    "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
]

AttentionImpls = {
    "float": AttentionImpls_fp32,
    "half": AttentionImpls_fp16,
    "bfloat16": AttentionImpls_bf16,
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("scheduling", ["sync"])
@torch.inference_mode
def test_basic_correctness_fp16(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_seqs: int,
    scheduling: str,
) -> None:
    attention_impls = AttentionImpls[dtype]

    impl_outputs_list = []

    for attention_impl in attention_impls:
        with vllm_runner(model,
                         dtype=dtype,
                         max_num_seqs=max_num_seqs,
                         scheduling=scheduling,
                         attention_impl=attention_impl,
                         output_last_hidden_states=True) as vllm_model:
            vllm_outputs = vllm_model.encode(example_prompts)
            impl_outputs_list.append((attention_impl, vllm_outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all(
            (all_similarities <= 1.0 + tolerance)
            & (all_similarities >= 1.0 - tolerance)
        ), f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
