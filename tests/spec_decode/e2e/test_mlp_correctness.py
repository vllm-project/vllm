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

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.vocab_parallel_embedding import pad_vocab_size

from .conftest import run_equality_correctness_test

# main model
MAIN_MODEL = "JackFram/llama-160m"

# speculative model
SPEC_MODEL = "ibm-fms/llama-160m-accelerator"

# max. number of speculative tokens: this corresponds to
# n_predict in the config.json of the speculator model.
MAX_SPEC_TOKENS = 3

# precision
PRECISION = "float32"


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
        "model_name": MAIN_MODEL,
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
def test_mlp_e2e_greedy_correctness(vllm_runner, common_llm_kwargs,
                                    per_test_common_llm_kwargs,
                                    baseline_llm_kwargs, test_llm_kwargs,
                                    batch_size: int, output_len: int,
                                    seed: int):
    """Verify greedy equality with different batch size."""
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
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": SPEC_MODEL,
    },
])
@pytest.mark.parametrize("output_len", [2048])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
def test_mlp_e2e_acceptance_rate(vllm_runner, common_llm_kwargs,
                                 per_test_common_llm_kwargs,
                                 baseline_llm_kwargs, test_llm_kwargs,
                                 batch_size: int, output_len: int, seed: int):
    """Verify acceptance rate with different batch size and large output 
    length."""
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  temperature=0.0,
                                  seed=seed,
                                  expected_acceptance_rate=0.48)


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
        "model_name": MAIN_MODEL,

        # Speculative model
        "speculative_model": SPEC_MODEL,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{"seed": 1}])
@pytest.mark.parametrize("test_llm_kwargs", [{"seed": 5}])
@pytest.mark.parametrize("output_len", [64])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("temperature", [0.1, 1.0])
@pytest.mark.parametrize("seed", [1])
def test_mlp_e2e_seeded_correctness(vllm_runner, common_llm_kwargs,
                                    per_test_common_llm_kwargs,
                                    baseline_llm_kwargs, test_llm_kwargs,
                                    batch_size: int, output_len: int,
                                    temperature: float, seed: int):
    """Verify seeded runs produce the same output."""
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  temperature=temperature,
                                  seed=seed)

    # Ensure this same test does fail if we _don't_ include per-request seeds
    with pytest.raises(AssertionError):
        run_equality_correctness_test(vllm_runner,
                                      common_llm_kwargs,
                                      per_test_common_llm_kwargs,
                                      baseline_llm_kwargs,
                                      test_llm_kwargs,
                                      batch_size,
                                      max_output_len=output_len,
                                      temperature=temperature,
                                      seed=seed,
                                      disable_seed=True)


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
        "model_name": MAIN_MODEL,
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
def test_mlp_e2e_greedy_correctness_with_preemption(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
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
        "model_name": MAIN_MODEL,
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
def test_mlp_e2e_greedy_correctness_with_padding(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality when the vocab dimension is padded
    """

    # Default pad_to is 64, test model has vocab_size of 32000
    def patched_pad_vocab_size(vocab_size, pad_to=None):
        return pad_vocab_size(vocab_size, pad_to=32064)

    with patch(
            "vllm.model_executor.layers.vocab_parallel_embedding.pad_vocab_size",
            patched_pad_vocab_size):
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
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,
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
def test_mlp_different_k(vllm_runner, common_llm_kwargs,
                         per_test_common_llm_kwargs, baseline_llm_kwargs,
                         test_llm_kwargs, batch_size: int, seed: int,
                         output_len: int):
    """Verify that mlp speculative decoding produces exact equality
    to without spec decode with different values of num_speculative_tokens.
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
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Precision
        "dtype": PRECISION,

        # Main model
        "model_name": MAIN_MODEL,
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
def test_mlp_disable_queue(vllm_runner, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           test_llm_kwargs, batch_size: int, seed: int,
                           output_len: int):
    """Verify that mlp speculative decoding produces exact equality
    to without spec decode when speculation is disabled for large
    batch sizes.
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
