# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest

from .conftest import run_equality_correctness_test

MAIN_MODEL = "JackFram/llama-68m"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{

        # Verify equality when cuda graphs allowed.
        "enforce_eager": False,
        "model_name": "JackFram/llama-68m",
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "speculative_config": {
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 5,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cuda_graph(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    """Verify spec decode equality when cuda graphs are enabled.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  seed=seed,
                                  temperature=0.0)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        # Explicitly specify draft model quantization
        {
            "speculative_config": {
                "model": "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
                "num_speculative_tokens": 5,
                "quantization": "gptq",
            },
        },
        # Explicitly specify GPTQ-based draft model to use marlin quantization
        {
            "speculative_config": {
                "model": "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
                "num_speculative_tokens": 5,
                "quantization": "marlin",
            },
        },
        # Not explicitly specify draft model quantization
        {
            "speculative_config": {
                "model": "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
                "num_speculative_tokens": 5,
                "quantization": None,
            },
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_speculative_model_quantization_config(vllm_runner, common_llm_kwargs,
                                               per_test_common_llm_kwargs,
                                               baseline_llm_kwargs,
                                               test_llm_kwargs,
                                               batch_size: int, seed: int):
    """Verify spec decode works well with draft model quantization configs.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=32,
                                  seed=seed,
                                  temperature=0.0)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": MAIN_MODEL,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_config": {
        "model": "JackFram/llama-68m",
        "num_speculative_tokens": 3,
        "disable_mqa_scorer": True,
    },
}])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_mqa_scorer(vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
                    baseline_llm_kwargs, test_llm_kwargs, batch_size: int,
                    output_len: int, seed: int):
    """Verify that speculative decoding generates the same output
    with batch expansion scorer and mqa scorer.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  seed=seed,
                                  temperature=0.0)
