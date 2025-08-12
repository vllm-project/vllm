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

With those tests, we can say at least, MLPSpeculator would not break the
correctess for the target model outputs.
"""

import pytest

from .conftest import run_greedy_equality_correctness_test

# main model
MAIN_MODEL = "ibm-granite/granite-3b-code-instruct"

# speculative model
SPEC_MODEL = "ibm-granite/granite-3b-code-instruct-accelerator"

# max. number of speculative tokens: this corresponds to
# n_predict in the config.json of the speculator model.
MAX_SPEC_TOKENS = 5

# precision
PRECISION = "float16"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model": MAIN_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": SPEC_MODEL,
    },
])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
def test_mlp_e2e_greedy_correctness(baseline_llm_generator, test_llm_generator,
                                    batch_size: int, output_len: int):
    """Verify greedy equality with different batch size."""
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "block_size": 8,
        # 2 for small prompt, 256//8 for generated.
        "num_gpu_blocks_override": 2 + 256 // 8,
        "max_model_len": (2 + 256 // 8) * 8,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model": MAIN_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": SPEC_MODEL,
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
def test_mlp_e2e_greedy_correctness_with_preemption(baseline_llm_generator,
                                                    test_llm_generator,
                                                    batch_size: int,
                                                    output_len: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model": MAIN_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": SPEC_MODEL,
            "num_speculative_tokens": k,
        }
        # Try a range of num. speculative tokens
        for k in range(1, 1 + MAX_SPEC_TOKENS)
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_mlp_different_k(baseline_llm_generator, test_llm_generator,
                         batch_size: int, output_len: int):
    """Verify that mlp speculative decoding produces exact equality
    to without spec decode with different values of num_speculative_tokens.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model": MAIN_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": SPEC_MODEL,
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
def test_mlp_disable_queue(baseline_llm_generator, test_llm_generator,
                           batch_size: int, output_len: int):
    """Verify that mlp speculative decoding produces exact equality
    to without spec decode when speculation is disabled for large
    batch sizes.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)
