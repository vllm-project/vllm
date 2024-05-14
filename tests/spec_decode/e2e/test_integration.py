"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs or tensor parallelism.
"""

import pytest

from .conftest import run_greedy_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 2,

        # Use AsyncLLM engine, so that the engine runs in its own process.
        # Otherwise, since vLLM does not follow true SPMD, the test runner
        # process will have both the engine and the rank0 worker. NCCL is not
        # cleaned up properly, and its server host thread leaks, causing the
        # second run of the test to fail with internal NCCL error.
        "use_async": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 3,
    },
    {
        "speculative_model": "[ngram]",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 3,
    },
])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_tp_gt_1(baseline_llm_generator, test_llm_generator, batch_size: int,
                 output_len: int):
    """Verify greedy equality when tensor parallelism is used.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


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
