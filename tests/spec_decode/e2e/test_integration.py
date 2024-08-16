"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest

from .conftest import run_greedy_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Required for spec decode.
        "use_v2_block_manager": True,

        # Verify equality when cuda graphs allowed.
        "enforce_eager": False,
        "model": "JackFram/llama-68m",
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cuda_graph(baseline_llm_generator, test_llm_generator,
                                batch_size, output_len):
    """Verify spec decode equality when cuda graphs are enabled.
    """
    run_greedy_equality_correctness_test(
        baseline_llm_generator,
        test_llm_generator,
        batch_size,
        max_output_len=output_len,
        force_output_len=True,
    )


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "speculative_model": "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        # Explicitly specify draft model quantization
        {
            "speculative_model_quantization": "gptq",
        },
        # Explicitly specify GPTQ-based draft model to use marlin quantization
        {
            "speculative_model_quantization": "marlin",
        },
        # Not explicitly specify draft model quantization
        {
            "speculative_model_quantization": None,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_speculative_model_quantization_config(baseline_llm_generator,
                                               test_llm_generator,
                                               batch_size: int):
    """Verify spec decode works well with draft model quantization configs.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=32,
                                         force_output_len=True)
