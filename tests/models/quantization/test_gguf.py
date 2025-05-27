# SPDX-License-Identifier: Apache-2.0
"""
Tests gguf models against unquantized models generations
Note: To pass the test, quantization higher than Q4 should be used
"""

import os
from typing import NamedTuple

import pytest
from huggingface_hub import hf_hub_download
from pytest import MarkDecorator
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported

from ...conftest import VllmRunner
from ...utils import multi_gpu_test
from ..utils import check_logprobs_close

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024


class GGUFTestConfig(NamedTuple):
    original_model: str
    gguf_repo: str
    gguf_filename: str
    marks: list[MarkDecorator] = []

    @property
    def gguf_model(self):
        return hf_hub_download(self.gguf_repo, filename=self.gguf_filename)


LLAMA_CONFIG = GGUFTestConfig(
    original_model="meta-llama/Llama-3.2-1B-Instruct",
    gguf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
    gguf_filename="Llama-3.2-1B-Instruct-IQ4_XS.gguf",
)

QWEN2_CONFIG = GGUFTestConfig(
    original_model="Qwen/Qwen2.5-1.5B-Instruct",
    gguf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    gguf_filename="qwen2.5-1.5b-instruct-q6_k.gguf",
)

PHI3_CONFIG = GGUFTestConfig(
    original_model="microsoft/Phi-3.5-mini-instruct",
    gguf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
    gguf_filename="Phi-3.5-mini-instruct-IQ4_XS.gguf",
)

GPT2_CONFIG = GGUFTestConfig(
    original_model="openai-community/gpt2-large",
    gguf_repo="QuantFactory/gpt2-large-GGUF",
    gguf_filename="gpt2-large.Q4_K_M.gguf",
)

STABLELM_CONFIG = GGUFTestConfig(
    original_model="stabilityai/stablelm-3b-4e1t",
    gguf_repo="afrideva/stablelm-3b-4e1t-GGUF",
    gguf_filename="stablelm-3b-4e1t.q4_k_m.gguf",
)

STARCODER_CONFIG = GGUFTestConfig(
    original_model="bigcode/starcoder2-3b",
    gguf_repo="QuantFactory/starcoder2-3b-GGUF",
    gguf_filename="starcoder2-3b.Q6_K.gguf",
)

DOLPHIN_CONFIG = GGUFTestConfig(
    # Test VocabParallelEmbedding sharding issue.
    original_model="cognitivecomputations/TinyDolphin-2.8-1.1b",
    gguf_repo="tsunemoto/TinyDolphin-2.8-1.1b-GGUF",
    gguf_filename="tinydolphin-2.8-1.1b.Q6_K.gguf",
)

MODELS = [
    LLAMA_CONFIG,
    QWEN2_CONFIG,
    PHI3_CONFIG,
    GPT2_CONFIG,
    # STABLELM_CONFIG,  # enable this when v1 support head_size=80
    DOLPHIN_CONFIG,
    # STARCODER_CONFIG, # broken
]


def check_model_outputs(
    vllm_runner: type[VllmRunner],
    prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model.original_model)
    if tokenizer.chat_template is not None:
        messages = [[{
            'role': 'user',
            'content': prompt
        }] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)

    # Run gguf model.
    with vllm_runner(model_name=model.gguf_model,
                     enforce_eager=True,
                     tokenizer_name=model.original_model,
                     dtype=dtype,
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=tp_size) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            prompts[:-1], max_tokens, num_logprobs)

    # Run unquantized model.
    # Should run with tp=1, otherwise the test will stuck at
    # nccl initialization.
    with vllm_runner(
            model_name=model.original_model,
            enforce_eager=True,  # faster tests
            dtype=dtype,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=1) as original_model:
        original_outputs = original_model.generate_greedy_logprobs(
            prompts[:-1], max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )


@pytest.mark.skipif(not is_quant_method_supported("gguf"),
                    reason="gguf is not supported on this GPU type.")
@pytest.mark.parametrize("model", [
    pytest.param(test_config, marks=test_config.marks)
    for test_config in MODELS
])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [1])
def test_models(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    check_model_outputs(vllm_runner, example_prompts, model, dtype, max_tokens,
                        num_logprobs, tp_size)


@pytest.mark.skipif(not is_quant_method_supported("gguf"),
                    reason="gguf is not supported on this GPU type.")
@pytest.mark.parametrize("model", [LLAMA_CONFIG])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [2])
@multi_gpu_test(num_gpus=2)
def test_distributed(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: GGUFTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    check_model_outputs(vllm_runner, example_prompts, model, dtype, max_tokens,
                        num_logprobs, tp_size)
