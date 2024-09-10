"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest

from .conftest import run_equality_correctness_test

MAIN_MODEL = "JackFram/llama-68m"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Required for spec decode.
        "--use-v2-block-manager",
    ]])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "--speculative-model",
            "JackFram/llama-68m",
            "--num-speculative-tokens",
            "5",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[]])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cuda_graph(common_llm_kwargs, per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    """Verify spec decode equality when cuda graphs are enabled.
    """
    run_equality_correctness_test(MAIN_MODEL,
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
        # Skip cuda graph recording for fast test.
        "--enforce-eager",

        # Required for spec decode.
        "--use-v2-block-manager",
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    [
        "--speculative-model",
        "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
        "--num-speculative-tokens",
        "5",
    ],
])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        # Explicitly specify draft model quantization
        [
            "--speculative-model-quantization",
            "gptq",
        ],
        # Explicitly specify GPTQ-based draft model to use marlin quantization
        [
            "--speculative-model-quantization",
            "marlin",
        ],
        # Not explicitly specify draft model quantization
        [],
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_speculative_model_quantization_config(common_llm_kwargs,
                                               per_test_common_llm_kwargs,
                                               baseline_llm_kwargs,
                                               test_llm_kwargs,
                                               batch_size: int, seed: int):
    """Verify spec decode works well with draft model quantization configs.
    """
    run_equality_correctness_test(MAIN_MODEL,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=32,
                                  seed=seed,
                                  temperature=0.0)
