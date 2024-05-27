import math
from itertools import cycle

import pytest

from vllm import SamplingParams

from .conftest import get_logprobs_from_llm_generator


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "max_logprobs": 6,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 3,
}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        7,
    ])
@pytest.mark.parametrize("seed", [1])
def test_logprobs_equality(baseline_llm_generator, test_llm_generator,
                           batch_size: int, output_len: int):
    """Verify output logprobs are equal with and without speculative decoding.
    """
    run_greedy_logprobs_correctness_test(baseline_llm_generator,
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
        "use_v2_block_manager": True,
        "max_logprobs": 6,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 3,
}])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_logprobs", [6])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        7,
    ])
@pytest.mark.parametrize("seed", [1])
def test_diff_num_logprobs(baseline_llm_generator, test_llm_generator,
                           batch_size: int, output_len: int,
                           num_logprobs: int):
    """Verify output logprobs are equal with and without spec decode.
    This specifies a number of logprobs >1.
    """
    run_greedy_logprobs_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True,
                                         logprob_rank=num_logprobs)


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
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 3,
}, {
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 6,
}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_logprobs_different_k(baseline_llm_generator, test_llm_generator,
                              batch_size: int, output_len: int):
    """Veriy logprob greedy equality with different speculation lens.
    """
    run_greedy_logprobs_correctness_test(baseline_llm_generator,
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
    [{
        "speculative_model": "JackFram/llama-160m",
        "num_speculative_tokens": 3,

        # Artificially limit the draft model max model len; this forces vLLM
        # to skip speculation once the sequences grow beyond 32-k tokens.
        "speculative_max_model_len": 32,
    }])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_logprobs_when_skip_speculation(baseline_llm_generator,
                                        test_llm_generator, batch_size: int,
                                        output_len: int):
    """Verify logprobs greedy equality when some sequences skip speculation.
    """
    run_greedy_logprobs_correctness_test(baseline_llm_generator,
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
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 3,
}])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_logprobs_temp_1(baseline_llm_generator, test_llm_generator,
                         batch_size: int, output_len: int):
    """Verify at least one logprob result has num_logprobs+1, which tests the
    case where the sampled token is not in top-k logprobs.

    Ideally, this test should validate equality with non-spec by getting
    logprobs. This is left as future improvement.
    """
    batch_size = 8
    max_output_len = output_len
    force_output_len = True
    logprob_rank = 5

    temperature = 1.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generated max_output_len tokens, then set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
        logprobs=logprob_rank,
    )

    spec_batch_logprobs = get_logprobs_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    num_returned_logprobs = [
        len(logprob_dict) for seq_logprobs in spec_batch_logprobs
        for logprob_dict in seq_logprobs
    ]

    # Assert one of the returned logprobs has > num_logprobs (indicating the
    # sampled token is not in top-k).
    assert any([
        num_returned > logprob_rank for num_returned in num_returned_logprobs
    ])


def run_greedy_logprobs_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len: bool,
                                         logprob_rank: int = 1):
    """Helper method that compares the logprobs outputs of both the baseline LLM
    and the test LLM. It asserts greedy equality of the logprobs when the
    temperature is zero.
    """
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generated max_output_len tokens, then set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
        logprobs=logprob_rank,
    )

    spec_batch_logprobs = get_logprobs_from_llm_generator(
        test_llm_generator, prompts, sampling_params)
    baseline_batch_logprobs = get_logprobs_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    assert len(baseline_batch_logprobs) == len(prompts)
    assert len(spec_batch_logprobs) == len(prompts)

    # For each sequence in the batch.
    for i, (baseline_logprobs, spec_logprobs) in enumerate(
            zip(baseline_batch_logprobs, spec_batch_logprobs)):
        assert len(spec_logprobs) == len(baseline_logprobs)

        # For each generated position of the sequence.
        for pos, (spec_pos_logprobs, baseline_pos_logprobs) in enumerate(
                zip(spec_logprobs, baseline_logprobs)):

            # Map rank to token/logprob in spec output.
            spec_rank_to_token_id = {
                value.rank: key
                for key, value in spec_pos_logprobs.items()
            }
            spec_rank_to_logprob = {
                value.rank: value.logprob
                for key, value in spec_pos_logprobs.items()
            }

            # Map rank to token/logprob in baseline output.
            baseline_rank_to_token_id = {
                value.rank: key
                for key, value in baseline_pos_logprobs.items()
            }
            baseline_rank_to_logprob = {
                value.rank: value.logprob
                for key, value in baseline_pos_logprobs.items()
            }

            # Assert set of ranks returned is equal.
            assert set(spec_rank_to_token_id.keys()) == set(
                baseline_rank_to_token_id.keys())

            # Assert each logprob/token id is correct, keyed by rank.
            for rank in sorted(set(spec_rank_to_token_id.keys())):
                assert spec_rank_to_token_id[
                    rank] == baseline_rank_to_token_id[rank], f"{rank}"
                assert math.isclose(
                    a=spec_rank_to_logprob[rank],
                    b=baseline_rank_to_logprob[rank],
                    abs_tol=1e-1,
                )
