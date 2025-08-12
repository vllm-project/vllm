from itertools import cycle

import pytest

from vllm import SamplingParams


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # Allow only 5 sequences of ~1024 tokens in worst case.
        "block_size": 16,
        "num_gpu_blocks_override": 5 * (64 + 1),
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "use_v2_block_manager": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{"use_v2_block_manager": True}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_v1_v2_greedy_equality_with_preemption(baseline_llm_generator,
                                               test_llm_generator, batch_size):
    """Verify block manager v2 produces same outputs as block manager v1, even
    when there is preemption.

    This constructs two LLM, each with limited number of GPU blocks. The limit
    is decided such that as the sequences in the batch grow, sequences must be
    preempted and removed from cache.

    If the output token ids are equivalent, then we have confidence that the KV
    cache is not corrupted in the v2 block manager.

    NOTE: We want a significant number of generated tokens so that any incorrect
    KV mapping has time to build up error.
    """
    output_len = 1024
    temperature = 0.0

    # We want to ensure equality even with preemption.
    # We force the total block size to be 1 + cdiv(output_len, block_size)
    # so that only one sequence can fit at a time (once the sequences grow).

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

    print('Getting token ids from block manager v1')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids from block manager v2')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # Use a large block size to trigger more copy-on-writes.
        "block_size": 32,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "use_v2_block_manager": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{"use_v2_block_manager": True}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_v1_v2_greedy_equality_with_cow(baseline_llm_generator,
                                        test_llm_generator, batch_size):
    """Verify beam search equality with block manager v1 and v2.

    This requires copy-on-writes; if the v1 and v2 output is the same, then
    we have some confidence cow is working.
    """
    output_len = 128
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
        use_beam_search=True,
        best_of=2,
    )

    print('Getting token ids from block manager v1')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids from block manager v2')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # Our prompts will generate 128 tokens; since the prompts themselves are
        # small, we don't need much KV space beyond 128.
        "max_model_len": 160,

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # Lookahead scheduling only supported in v2 block manager.
        "use_v2_block_manager": True,
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "block_size": 16,

            # Allow only 2 sequences of ~128 tokens in worst case.
            # Note 8 = 128/block_size
            "num_gpu_blocks_override": 2 * (8 + 1),
        },
        {
            "block_size": 8,

            # Allow only 2 sequences of ~128 tokens in worst case.
            # Note 16 = 128/block_size
            "num_gpu_blocks_override": 2 * (16 + 1),
        }
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "num_lookahead_slots": 0,
}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [{
        # We run one test with block_size < lookahead_slots, one test with
        # block_size > lookahead_slots
        "num_lookahead_slots": 10,
    }])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_lookahead_greedy_equality_with_preemption(baseline_llm_generator,
                                                   test_llm_generator,
                                                   batch_size):
    """Verify vLLM produces the same output with greedy sampling, when lookahead
    scheduling is used vs. not.

    Lookahead scheduling is not expected to modify the output, as it simply
    allocates empty slots ahead of the known token ids in a sliding fashion.

    This test constrains the total number of blocks to force preemption. It also
    varies the block size so that the lookahead size is less than and greater
    than the block size.
    """
    output_len = 128
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

    print('Getting token ids without lookahead scheduling')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids with lookahead scheduling')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [
        {
            # Use a small model for a fast test.
            "model": "facebook/opt-125m",

            # skip cuda graph creation for fast test.
            "enforce_eager": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 2,
            "max_num_seqs": 2,
        },
    ])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [
    {
        "use_v2_block_manager": False,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "use_v2_block_manager": True,
        "num_lookahead_slots": 0,
    },
    {
        "use_v2_block_manager": True,
        "num_lookahead_slots": 5,
    },
])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_chunked_prefill_block_manager_v2(baseline_llm_generator,
                                          test_llm_generator, batch_size):
    """Verify that chunked prefill works with BlockManagerV2, with and without
    lookahead scheduling.
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

    print('Getting token ids with BlockManagerV1')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids with BlockManagerV2')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # Allow only 5 sequences of ~1024 tokens in worst case.
        "block_size": 16,
        "num_gpu_blocks_override": 5 * (64 + 1),

        # Enable prefill cache
        "enable_prefix_caching": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "use_v2_block_manager": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{"use_v2_block_manager": True}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_v1_v2_greedy_equality_prefix_caching_enabled_with_preemption(
        baseline_llm_generator, test_llm_generator, batch_size):
    """Verify block manager v2 produces same outputs as block manager v1, even
    when there is preemption.

    This constructs two LLM, each with limited number of GPU blocks. The limit
    is decided such that as the sequences in the batch grow, sequences must be
    preempted and removed from cache.

    If the output token ids are equivalent, then we have confidence that the KV
    cache is not corrupted in the v2 block manager.

    NOTE: We want a significant number of generated tokens so that any incorrect
    KV mapping has time to build up error.
    """
    output_len = 1024
    temperature = 0.0

    # We want to ensure equality even with preemption.
    # We force the total block size to be 1 + cdiv(output_len, block_size)
    # so that only one sequence can fit at a time (once the sequences grow).

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

    print('Getting token ids from block manager v1')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids from block manager v2')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # Allow only 5 sequences of ~1024 tokens in worst case.
        "block_size": 16,
        "num_gpu_blocks_override": 5 * (64 + 1),

        # Test APC in v2 block
        "use_v2_block_manager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "enable_prefix_caching": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{"enable_prefix_caching": True}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_auto_prefix_caching_with_preemption(baseline_llm_generator,
                                             test_llm_generator, batch_size):
    """Verify block manager v2 with auto prefix caching enabled produces same
    outputs as auto prefix caching disabled, even when there is preemption.

    This constructs two LLM, each with limited number of GPU blocks. The limit
    is decided such that as the sequences in the batch grow, sequences must be
    preempted and removed from cache.

    If the output token ids are equivalent, then we have confidence that auto
    prefix caching itself at least don't cause result error.
    """
    output_len = 1024
    temperature = 0.0

    # We want to ensure equality even with preemption.
    # We force the total block size to be 1 + cdiv(output_len, block_size)
    # so that only one sequence can fit at a time (once the sequences grow).
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

    print('Getting token ids with APC disabled')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids with APC enabled')
    test_token_ids = get_token_ids_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    for expected_token_ids, actual_token_ids in zip(baseline_token_ids,
                                                    test_token_ids):
        assert expected_token_ids == actual_token_ids

    assert baseline_token_ids == test_token_ids


def get_token_ids_from_llm_generator(llm_generator, prompts, sampling_params):
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        del llm

    return token_ids
