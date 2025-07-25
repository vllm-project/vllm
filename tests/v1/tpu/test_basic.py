# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A basic correctness check for TPUs

Run `pytest tests/v1/tpu/test_basic.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch_xla._internal import tpu

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from tests.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    # TODO: Enable this model when fixed.
    # "Qwen/Qwen1.5-MoE-A2.7B",
    # TODO: Enable this models with v6e
    # "Qwen/Qwen2-7B-Instruct",
    # "meta-llama/Llama-3.1-8B",
]

TENSOR_PARALLEL_SIZES = [1]
MAX_NUM_REQS = [16, 1024]

# TODO: Enable when CI/CD will have a multi-tpu instance
# TENSOR_PARALLEL_SIZES = [1, 4]


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", TENSOR_PARALLEL_SIZES)
@pytest.mark.parametrize("max_num_seqs", MAX_NUM_REQS)
def test_basic(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    max_tokens: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
) -> None:
    prompt = "The next numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with vllm_runner(
                model,
                # Note: max_num_batched_tokens == 1024 is needed here to
                # actually test chunked prompt
                max_num_batched_tokens=1024,
                max_model_len=8192,
                gpu_memory_utilization=0.7,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)
        output = vllm_outputs[0][1]

        assert "1024" in output or "0, 1" in output


@pytest.mark.skip(reason="Temporarily disabled due to timeout")
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("max_num_seqs", [16])
def test_phi3(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    max_tokens: int,
    max_num_seqs: int,
) -> None:
    prompts = [
        "A robot may not injure a human being",
        "It is only with the heart that one can see rightly;",
        "The greatest glory in living lies not in never falling,",
    ]
    answers = [
        " or, by violating privacy",
        " what is essential is love.",
        " but in rising every time we fall.",
    ]
    # test head dim = 96
    model = "microsoft/Phi-3-mini-128k-instruct"

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with vllm_runner(model,
                         max_num_batched_tokens=256,
                         max_num_seqs=max_num_seqs) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens)
        # vllm_outputs is a list of tuples whose first element is the token id
        # and the second element is the output (including the prompt).
        for output, answer in zip(vllm_outputs, answers):
            generated_text = output[1]
            assert answer in generated_text


TP_SIZE_8 = 8


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a test for TPU only")
@pytest.mark.skipif(tpu.num_available_chips() < TP_SIZE_8,
                    reason=f"This test requires {TP_SIZE_8} TPU chips.")
def test_gemma3_27b_with_text_input_and_tp(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "google/gemma-3-27b-it"
    max_tokens = 16
    tensor_parallel_size = TP_SIZE_8
    max_num_seqs = 4
    prompts = [
        "A robot may not injure a human being",
        "It is only with the heart that one can see rightly;",
        "The greatest glory in living lies not in never falling,",
    ]
    answers = [
        " or, through inaction, allow a human being to come to harm.",
        " what is essential is invisible to the eye.",
        " but in rising every time we fall.",
    ]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with vllm_runner(
                model,
                max_num_batched_tokens=256,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens)
        # vllm_outputs is a list of tuples whose first element is the token id
        # and the second element is the output (including the prompt).
        for output, answer in zip(vllm_outputs, answers):
            generated_text = output[1]
            assert answer in generated_text


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
def test_w8a8_quantization(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    max_tokens = 5
    tensor_parallel_size = 1
    max_num_seqs = 4

    prompt = "The next numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with vllm_runner(
                model,
                max_num_batched_tokens=64,
                max_model_len=4096,
                gpu_memory_utilization=0.7,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)
        output = vllm_outputs[0][1]

        assert "1024" in output or "0, 1" in output
