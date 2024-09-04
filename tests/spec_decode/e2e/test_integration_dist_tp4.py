"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

import pytest
import torch

from .conftest import run_greedy_equality_correctness_test


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 4,

        # Use AsyncLLM engine, so that the engine runs in its own process.
        # Otherwise, since vLLM does not follow true SPMD, the test runner
        # process will have both the engine and the rank0 worker. NCCL is not
        # cleaned up properly, and its server host thread leaks, causing the
        # second run of the test to fail with internal NCCL error.
        "use_async": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        #TODO(wooyeon): add spec_draft_dp=2 case
        {
            "speculative_draft_tensor_parallel_size": 1,
        },
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_draft_model_tp_lt_target_model_tp4(test_llm_generator,
                                            baseline_llm_generator,
                                            batch_size: int):
    """Verify spec decode works well with smaller tp for draft models.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=32,
                                         force_output_len=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 4,

        # Use AsyncLLM engine, so that the engine runs in its own process.
        # Otherwise, since vLLM does not follow true SPMD, the test runner
        # process will have both the engine and the rank0 worker. NCCL is not
        # cleaned up properly, and its server host thread leaks, causing the
        # second run of the test to fail with internal NCCL error.
        "use_async": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,

            # Artificially limit the draft model max model len; this forces vLLM
            # to skip speculation once the sequences grow beyond 32-k tokens.
            "speculative_max_model_len": 32,
        },
    ])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # This must be a good bit larger than speculative_max_model_len so that
        # we can test the case where all seqs are skipped, but still small to
        # ensure fast test.
        64,
    ])
@pytest.mark.parametrize("seed", [1])
def test_skip_speculation(baseline_llm_generator, test_llm_generator,
                          batch_size: int, output_len: int):
    """Verify job failure with RuntimeError when all sequences skip speculation.
    We do this by setting the max model len of the draft model to an
    artificially low value, such that when the sequences grow beyond it, they
    are skipped in speculative decoding.

    TODO: fix it to pass without raising Error. (#5814)
    """
    with pytest.raises(RuntimeError):
        run_greedy_equality_correctness_test(baseline_llm_generator,
                                             test_llm_generator,
                                             batch_size,
                                             max_output_len=output_len,
                                             force_output_len=True)
