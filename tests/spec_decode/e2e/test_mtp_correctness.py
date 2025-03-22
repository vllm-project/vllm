# SPDX-License-Identifier: Apache-2.0
"""This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality.

However, we still need to verify below scenario could be passed:
    * Batch size 1 greedy equality
    * Batch size >1 greedy equality
    * Test greedy equality under preemption
    * Test greedy equality under various number of speculative tokens.

With those tests, we can say at least, mtp would not break the
correctess for the target model outputs.
"""

import pytest

from ..utils import maybe_enable_chunked_prefill
from .conftest import run_equality_correctness_test

# main model
MAIN_MODEL = "luccafong/deepseek_mtp_main_random"

# max. number of speculative tokens
MAX_SPEC_TOKENS = 3

# precision
PRECISION = "bfloat16"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,

        # GPU memory utilization
        "gpu_memory_utilization": 0.85
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "num_speculative_tokens": MAX_SPEC_TOKENS,
    },
])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_e2e_greedy_correctness(vllm_runner, common_llm_kwargs,
                                    per_test_common_llm_kwargs,
                                    baseline_llm_kwargs, test_llm_kwargs,
                                    batch_size: int, output_len: int,
                                    seed: int, prefill_chunk_size: bool):
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,

        # GPU memory utilization
        "gpu_memory_utilization": 0.85
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "num_speculative_tokens": MAX_SPEC_TOKENS,
        "disable_logprobs_during_spec_decoding": False,
    },
    {
        "num_speculative_tokens": MAX_SPEC_TOKENS,
        "disable_logprobs_during_spec_decoding": True,
    },
])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("logprobs", [1, 6])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_e2e_greedy_logprobs(vllm_runner, common_llm_kwargs,
                                 per_test_common_llm_kwargs,
                                 baseline_llm_kwargs, test_llm_kwargs,
                                 batch_size: int, output_len: int, seed: int,
                                 logprobs: int, prefill_chunk_size: int):
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  output_len,
                                  seed,
                                  logprobs=logprobs,
                                  prompt_logprobs=logprobs,
                                  disable_logprobs=test_llm_kwargs[
                                      'disable_logprobs_during_spec_decoding'])


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "enforce_eager": False,

        # Print spec metrics.
        "disable_log_stats": False,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,
        "gpu_memory_utilization": 0.85
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "num_speculative_tokens": MAX_SPEC_TOKENS,
    },
])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_e2e_greedy_correctness_cuda_graph(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int, prefill_chunk_size: int):
    """Verify greedy equality with cuda graph enabled and different
    batch sizes."""
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "block_size": 8,
        # 2 for small prompt, 256//8 for generated.
        "num_gpu_blocks_override": 2 + 256 // 8,
        "max_model_len": (2 + 256 // 8) * 8,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,

        # GPU memory utilization
        "gpu_memory_utilization": 0.9
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "num_speculative_tokens": MAX_SPEC_TOKENS,
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        128,
    ])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_e2e_greedy_correctness_with_preemption(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int, prefill_chunk_size: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
    """
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,

        # GPU memory utilization
        "gpu_memory_utilization": 0.9
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "num_speculative_tokens": k,
        }
        # Try a range of num. speculative tokens
        for k in range(1, 3)
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_different_k(vllm_runner, common_llm_kwargs,
                         per_test_common_llm_kwargs, baseline_llm_kwargs,
                         test_llm_kwargs, batch_size: int, output_len: int,
                         seed: int, prefill_chunk_size: int):
    """Verify that mtp speculative decoding produces exact equality
    to without spec decode with different values of num_speculative_tokens.
    """
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,

        # GPU memory utilization
        "gpu_memory_utilization": 0.9
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "num_speculative_tokens": MAX_SPEC_TOKENS,
                             "speculative_disable_by_batch_size": 4
                         }])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 32])
def test_mtp_disable_queue(vllm_runner, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           test_llm_kwargs, batch_size: int, output_len: int,
                           seed: int, prefill_chunk_size: int):
    """Verify that mtp speculative decoding produces exact equality
    to without spec decode when speculation is disabled for large
    batch sizes.
    """
    maybe_enable_chunked_prefill(prefill_chunk_size, test_llm_kwargs)
    run_equality_correctness_test(vllm_runner, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
