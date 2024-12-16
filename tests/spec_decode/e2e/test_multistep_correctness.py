"""The tests in this file verify end-to-end speculative decoding correctness.

This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality. This gives us good coverage of temp=0.

At temp=0, the TypicalAcceptanceSampler ensures that only the tokens with the
highest probability in the target distribution are accepted. Therefore, we can 
expect greedy equality for the TypicalAcceptanceSampler at temp=0.

For temp>0, we rely on unit tests on the rejection sampler to verify that the
output distribution is the same with spec decode vs. no spec decode (this would
be prohibitively expensive to run with a real model). Similarly, for the
TypicalAcceptance sampler also, we rely on unit tests to validate temp>0
test cases.

NOTE: Speculative decoding's distribution equality requires that the measured
distributions of the target model and proposal model be deterministic given the
same input. vLLM largely guarantees this.

@cadedaniel has seen cases where the output probabilities of a draft/target
model change slightly with certain batch sizes or prompts, even with Torch
determinism flags set. It is unclear if this is a bug in vLLM, due to non-
determinism in on-device batched operations, a bug in vLLM's spec decode
implementation, or the "hardware numerics" limitations. Either way, rejection
sampling ensures the output distribution matches the target model, but it breaks
greedy-equality tests for those batch sizes/prompts.
"""

from itertools import cycle

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams

from ...utils import fork_new_process_for_each_test
from .conftest import (get_output_from_llm_generator,
                       run_equality_correctness_test)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
            "enable_chunked_prefill": False,
        },
        {
            # Chunked prefill enabled with small value
            # to make sure we get mixed batches.
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 4
        },
        {
            # Verify the detokenizer assertions in the test work when spec
            # decode is disabled.
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_with_detokenization(test_llm_generator,
                                             batch_size: int):
    """Run generation with speculative decoding on a batch. Verify the engine
    generates the correct number of tokens (via ignore_eos=True), and that the
    detokenization matches HF transformers.
    """
    output_len = 32
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    batch_tokens, batch_token_ids, _ = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    # Expect a generation for each prompt in the batch.
    assert len(batch_token_ids) == len(prompts)

    # Expect each generation to have expected number of tokens (note ignore_eos
    # is True).
    assert [len(token_ids)
            for token_ids in batch_token_ids] == ([output_len] * batch_size)

    # Expect detokenized string to match.
    tok = AutoTokenizer.from_pretrained("JackFram/llama-68m")
    for actual_tokens, actual_token_ids in zip(batch_tokens, batch_token_ids):
        expected_tokens = tok.decode(actual_token_ids)
        print(f"{actual_token_ids=}")
        assert actual_tokens.strip() == expected_tokens.strip()


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model_name": "JackFram/llama-68m",
        },
        {
            "model_name": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": "JackFram/llama-68m",
                             "num_speculative_tokens": 5,
                             "enable_chunked_prefill": False,
                             "disable_logprobs_during_spec_decoding": False
                         }, {
                             "speculative_model": "JackFram/llama-68m",
                             "num_speculative_tokens": 3,
                             "enable_chunked_prefill": True,
                             "max_num_batched_tokens": 4,
                             "max_num_seqs": 4,
                             "disable_logprobs_during_spec_decoding": False
                         }])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use long output len for the small model test.
        10,
    ])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_tiny_model_bs1(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality on a tiny model with batch size of one.

    Since this test is cheaper than other e2e correctness tests, we generate
    with a higher output_len.

    When the draft model is the same as the target model, we further check
    whether all speculative tokens are accepted.
    """
    ensure_all_accepted = per_test_common_llm_kwargs.get(
        "model_name") == test_llm_kwargs.get("speculative_model")
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  seed=seed,
                                  prompt_logprobs=2,
                                  logprobs=2,
                                  disable_logprobs=False,
                                  temperature=0.0,
                                  ensure_all_accepted=ensure_all_accepted)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model_name": "JackFram/llama-68m",
        },
        {
            "model_name": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        256,
    ])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality on a tiny model and large batch size.
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
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model_name": "JackFram/llama-68m",
        },
        {
            "model_name": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    },
])
@pytest.mark.parametrize("max_output_len", [
    256,
])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs_diff_output_len(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int,
        max_output_len: int, seed: int):
    """Verify greedy equality on a tiny model, with a large batch size, and when
    sampling respects the EOS token.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len,
                                  seed=seed,
                                  temperature=0.0,
                                  ignore_eos=False)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # A "real" model (not tiny).
        "model_name": "meta-llama/Llama-2-7b-chat-hf",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    },
])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use decently long output len for a high quality test.
        256,
    ])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_real_model_bs1(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality on a "real" model and batch size of 1. This is
    separate from large BS tests to make identifying the source of bugs easier.
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
        # A "real" model (not tiny).
        "model_name": "meta-llama/Llama-2-7b-chat-hf",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    },
])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        64,
    ])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_real_model_large_bs(
        vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, batch_size: int, output_len: int,
        seed: int):
    """Verify greedy equality with a "real" model on a nontrivial batch size.
    This is the closest test to a real production workload.
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
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "model_name": "JackFram/llama-160m",
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        256,
    ])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_spec_decode_e2e_greedy_correctness_with_preemption(
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
        "model_name": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # As of this writing, vLLM only compiles with these 3 block sizes by
        # default.
        {
            "block_size": 8,
        },
        {
            "block_size": 16,
        },
        {
            "block_size": 32,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
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
@fork_new_process_for_each_test
def test_spec_decode_different_block_size(vllm_runner, common_llm_kwargs,
                                          per_test_common_llm_kwargs,
                                          baseline_llm_kwargs, test_llm_kwargs,
                                          batch_size: int, output_len: int,
                                          seed: int):
    """Verify greedy equality over different block sizes.
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
            "enable_chunked_prefill": False,
        },
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 4,
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
@fork_new_process_for_each_test
def test_skip_speculation(vllm_runner, common_llm_kwargs,
                          per_test_common_llm_kwargs, baseline_llm_kwargs,
                          test_llm_kwargs, batch_size: int, output_len: int,
                          seed: int):
    """Verify greedy equality when some (or all) sequences skip speculation.
    We do this by setting the max model len of the draft model to an
    artificially low value, such that when the sequences grow beyond it, they
    are skipped in speculative decoding.
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
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "speculative_disable_by_batch_size": 2,
        "enable_chunked_prefill": False,
    },
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "speculative_disable_by_batch_size": 2,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4,
    },
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [10])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_disable_speculation(vllm_runner, common_llm_kwargs,
                             per_test_common_llm_kwargs, baseline_llm_kwargs,
                             test_llm_kwargs, batch_size: int, output_len: int,
                             seed: int):
    """Verify greedy equality when all sequences disable speculation.
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
        "model_name": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": k,
            "enable_chunked_prefill": False,
        }
        # Try a range of common k, as well as large speculation.
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63]
    ] + [{
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": k,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4,
    } for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63]])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_many_k(vllm_runner, common_llm_kwargs, per_test_common_llm_kwargs,
                baseline_llm_kwargs, test_llm_kwargs, batch_size: int,
                output_len: int, seed: int):
    """Verify that speculative decoding produces exact equality to without spec
    decode with many different values of k.
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
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": k,
            "spec_decoding_acceptance_method": "typical_acceptance_sampler",
            "enable_chunked_prefill": False
        }
        # Try a range of common k.
        for k in [1, 2, 3]
    ] + [{
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": k,
        "spec_decoding_acceptance_method": "typical_acceptance_sampler",
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4,
        "max_num_seqs": 4
    } for k in [1, 2, 3]])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@fork_new_process_for_each_test
def test_typical_acceptance_sampling(vllm_runner, common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs, test_llm_kwargs,
                                     batch_size: int, output_len: int,
                                     seed: int):
    """Verify that speculative decoding produces exact equality to without spec
    decode with TypicalAcceptanceSampler as the draft token acceptance
    sampling method.
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
