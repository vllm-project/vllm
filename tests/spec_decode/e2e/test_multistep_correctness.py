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

from .conftest import (get_output_from_llm_generator,
                       run_greedy_equality_correctness_test)


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
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
        },
        {
            # Verify the detokenizer assertions in the test work when spec
            # decode is disabled.
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
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

    batch_tokens, batch_token_ids = get_output_from_llm_generator(
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
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Use AsyncLLM engine
        "use_async": True,
    }])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_with_async_engine(test_llm_generator,
                                           baseline_llm_generator,
                                           batch_size: int):
    """Verify spec decode works well with async LLM engine.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=32,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model": "JackFram/llama-68m",
        },
        {
            "model": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use long output len for the small model test.
        1536,
    ])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_bs1(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality on a tiny model with batch size of one.

    Since this test is cheaper than other e2e correctness tests, we generate
    with a higher output_len.
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

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model": "JackFram/llama-68m",
        },
        {
            "model": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
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
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality on a tiny model and large batch size.
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
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model": "JackFram/llama-68m",
        },
        {
            "model": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize("max_output_len", [
    256,
])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs_diff_output_len(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        max_output_len: int):
    """Verify greedy equality on a tiny model, with a large batch size, and when
    sampling respects the EOS token.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len=False)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # A "real" model (not tiny).
        "model": "meta-llama/Llama-2-7b-chat-hf",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
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
def test_spec_decode_e2e_greedy_correctness_real_model_bs1(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality on a "real" model and batch size of 1. This is
    separate from large BS tests to make identifying the source of bugs easier.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # A "real" model (not tiny).
        "model": "meta-llama/Llama-2-7b-chat-hf",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Print spec metrics.
        "disable_log_stats": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
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
def test_spec_decode_e2e_greedy_correctness_real_model_large_bs(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        output_len: int):
    """Verify greedy equality with a "real" model on a nontrivial batch size.
    This is the closest test to a real production workload.
    """
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
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "model": "JackFram/llama-160m",
    },
])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
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
def test_spec_decode_e2e_greedy_correctness_with_preemption(
        baseline_llm_generator, test_llm_generator, batch_size: int,
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
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
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
def test_spec_decode_different_block_size(baseline_llm_generator,
                                          test_llm_generator, batch_size: int,
                                          output_len: int):
    """Verify greedy equality over different block sizes.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
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
    """Verify greedy equality when some (or all) sequences skip speculation.
    We do this by setting the max model len of the draft model to an
    artificially low value, such that when the sequences grow beyond it, they
    are skipped in speculative decoding.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "speculative_disable_by_batch_size": 2,
    },
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [10])
@pytest.mark.parametrize("seed", [1])
def test_disable_speculation(baseline_llm_generator, test_llm_generator,
                             batch_size: int, output_len: int):
    """Verify greedy equality when all sequences disable speculation.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": k,
        }
        # Try a range of common k, as well as large speculation.
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63]
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_many_k(baseline_llm_generator, test_llm_generator, batch_size: int,
                output_len: int):
    """Verify that speculative decoding produces exact equality to without spec
    decode with many different values of k.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": k,
            "spec_decoding_acceptance_method": "typical_acceptance_sampler"
        }
        # Try a range of common k.
        for k in [1, 2, 3]
    ])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_typical_acceptance_sampling(baseline_llm_generator,
                                     test_llm_generator, batch_size: int,
                                     output_len: int):
    """Verify that speculative decoding produces exact equality to without spec
    decode with TypicalAcceptanceSampler as the draft token acceptance
    sampling method.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)
