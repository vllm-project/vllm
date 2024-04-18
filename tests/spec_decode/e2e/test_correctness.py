from itertools import cycle
from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
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
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        # Try two different num spec tokens.
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
        "use_v2_block_manager": True
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
        "use_v2_block_manager": True
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
        "speculative_max_model_len": 32,
    },
])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        64,
    ])
@pytest.mark.parametrize("seed", [1])
def test_skip_speculation(baseline_llm_generator, test_llm_generator,
                          batch_size: int, output_len: int):
    """Verify correct output when we skip speculation.
    Test skip 1, skip >1, skip all.
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
        #{
        #    "model": "JackFram/llama-160m",
        #},
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
        #1536,
        128,
    ])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seed", [1])
#@pytest.mark.skip("used for local testing (cade to remove)")
def test_wip_validate_acceptance_rate(baseline_llm_generator,
                                      test_llm_generator, batch_size: int,
                                      output_len: int):
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


def run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len: bool,
                                         print_tokens: bool = False):
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
    )

    spec_batch_tokens, spec_batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    (baseline_batch_tokens,
     baseline_batch_token_ids) = get_output_from_llm_generator(
         baseline_llm_generator, prompts, sampling_params)

    assert len(baseline_batch_token_ids) == len(prompts)
    assert len(spec_batch_token_ids) == len(prompts)

    for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
            spec_tokens) in enumerate(
                zip(baseline_batch_token_ids, baseline_batch_tokens,
                    spec_batch_token_ids, spec_batch_tokens)):
        if print_tokens:
            print(f'{i=} {baseline_tokens=}')
            print(f'{i=}     {spec_tokens=}')
        print(f'{i=} {baseline_token_ids=}')
        print(f'{i=}     {spec_token_ids=}')
        assert baseline_token_ids == spec_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Skip real loading for fast test.
        "load_format": "dummy",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Expect failure as spec decode not supported by
            # Ray backend.
            "worker_use_ray": True,
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail(test_llm_generator):
    """Verify that speculative decoding with Ray fails.
    """
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    with pytest.raises(AssertionError,
                       match="Speculative decoding not yet supported for "):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]]]:
    tokens = []
    token_ids = []
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]
        del llm

    return tokens, token_ids
