"""This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality.

For ngram lookup, its idea comes from https://github.com/apoorvumang/prompt-lookup-decoding,
and is merged into transform code base: https://github.com/huggingface/transformers/pull/27775.
Since there is no model is needed for generate the proposal, we could make
the testcase much simpler than drafter multi-step one.

However, we still need to verify below scenario could be passed:
    * Batch size 1 greedy equality
    * Batch size >1 greedy equality
    * Test greedy equality under preemption
    * Test greedy equality under various ngram sizes / speculative sizes

With those tests, we can say at least, ngram spec would not break the correctess
for the target model outputs.
"""

import pytest

from .conftest import run_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce_eager",

        # Required for spec decode.
        "--use-v2-block-manager",

        # Print spec metrics.
        "--disable-log-stats",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative_model",
    "[ngram]",
    "--num_speculative-tokens",
    "5",
    "--ngram-prompt-lookup-max",
    "3",
]])
@pytest.mark.parametrize("output_len", [
    256,
])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
def test_ngram_e2e_greedy_correctness(common_llm_kwargs,
                                      per_test_common_llm_kwargs,
                                      baseline_llm_kwargs, test_llm_kwargs,
                                      batch_size: int, output_len: int,
                                      seed: int):
    """Verify greedy equality on a tiny model with different batch size."""
    run_equality_correctness_test("JackFram/llama-68m",
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
    [[
        "--block-size",
        "8",
        # 2 for small prompt, 256//8 for generated.
        "--num-gpu-blocks-override",
        f"{2 + 256 // 8}",
        "--max-model-len",
        f"{(2 + 256 // 8) * 8}",

        # Skip cuda graph recording for fast test.
        "--enforce-eager",

        # Required for spec decode.
        "--use-v2-block-manager"
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    [
        "--speculative-model",
        "[ngram]",
        "--num-speculative-tokens",
        "5",
        "--ngram-prompt-lookup_max",
        "3",
    ],
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        256,
    ])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_ngram_e2e_greedy_correctness_with_preemption(
        common_llm_kwargs, per_test_common_llm_kwargs, baseline_llm_kwargs,
        test_llm_kwargs, batch_size: int, output_len: int, seed: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
    """
    run_equality_correctness_test("JackFram/llama-160m",
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  temperature=0,
                                  seed=seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        "--enforce-eager",

        # Required for spec decode.
        "--use-v2-block-manager"
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[[
    "--speculative-model", "[ngram]", "--ngram-prompt-lookup-max", "3",
    "--num-speculative-tokens", f"{k}"
] for k in [1, 3, 5]] + [[
    "--speculative-model", "[ngram]", "--ngram-prompt-lookup-max", "1",
    "--num-speculative-tokens", f"{k}"
] for k in [1, 3, 5]]])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_ngram_different_k(common_llm_kwargs, per_test_common_llm_kwargs,
                           baseline_llm_kwargs, test_llm_kwargs,
                           batch_size: int, output_len: int, seed: int):
    """Verify that ngram speculative decoding produces exact equality
    to without spec decode with many different values of k and
    different ngram_prompt_lookup_max.
    """
    run_equality_correctness_test("JackFram/llama-68m",
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
    [[
        # Skip cuda graph recording for fast test.
        "--enforce-eager",

        # Required for spec decode.
        "--use-v2-block-manager"
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative-model", "[ngram]", "--num-speculative-tokens", "5",
    "--ngram-prompt-lookup-max", "3", "--speculative-disable-by-batch-size",
    "4"
]])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_ngram_disable_queue(common_llm_kwargs, per_test_common_llm_kwargs,
                             baseline_llm_kwargs, test_llm_kwargs,
                             batch_size: int, output_len: int, seed: int):
    """Verify that ngram speculative decoding produces exact equality
    to without spec decode with many different values of k and
    different ngram_prompt_lookup_max.
    """
    run_equality_correctness_test("JackFram/llama-68m",
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  seed=seed)
