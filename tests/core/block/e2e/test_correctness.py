from itertools import cycle

import pytest

from vllm import SamplingParams

from .conftest import get_token_ids_from_llm_generator


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
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "preemption_mode": "swap"
}, {
    "preemption_mode": "recompute"
}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_block_manager_with_preemption(baseline_llm_generator,
                                       test_llm_generator, batch_size):
    """Verify block manager produces same outputs even when there is preemption.

    This constructs two LLM, each with limited number of GPU blocks. The limit
    is decided such that as the sequences in the batch grow, sequences must be
    preempted and removed from cache.

    If the output token ids are equivalent, then we have confidence that the KV
    cache is not corrupted.

    NOTE: We want a significant number of generated tokens so that any incorrect
    KV mapping has time to build up error.

    NOTE(Kuntai): Though we have removed block manager v1, this test is still
    useful as it asserts the behavior of block manager v2 (now it is called 
    SelfAttnBlockSpaceManager) is the same when swapping / preemption, so we  
    keep this test.
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

    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

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
            "num_gpu_blocks_override": 2 * (16 + 2),
        }
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "num_lookahead_slots": 0,
}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            # We run one test with block_size < lookahead_slots, one test with
            # block_size > lookahead_slots
            "num_lookahead_slots": 10,
            "preemption_mode": "swap",
        },
        {
            "num_lookahead_slots": 10,
            "preemption_mode": "recompute",
        }
    ])
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
        },
    ])
@pytest.mark.parametrize("per_test_common_llm_kwargs",
                         [{
                             "block_size": 8,
                             "max_num_batched_tokens": 2,
                             "max_num_seqs": 2,
                         }, {
                             "block_size": 8,
                             "max_num_batched_tokens": 3,
                             "max_num_seqs": 2,
                         }, {
                             "block_size": 8,
                             "max_num_batched_tokens": 256,
                             "max_num_seqs": 10,
                         }])
@pytest.mark.parametrize("baseline_llm_kwargs", [
    {},
])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "num_lookahead_slots": 0,
    },
    {
        "num_lookahead_slots": 5,
    },
])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_chunked_prefill_block_manager(baseline_llm_generator,
                                       test_llm_generator, batch_size):
    """Verify that chunked prefill works with SelfAttnBlockSpaceManager, 
    with and without lookahead scheduling.
    """
    output_len = 32
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        ("1 + " * 50) + " 1 = ",  # Longer prompt.
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    print('Getting token ids with BlockManager')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids with BlockManager, with lookahead slots.')
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
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "preemption_mode": "swap"
}, {
    "preemption_mode": "recompute"
}])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("seed", [1])
def test_block_manager_prefix_caching_enabled_with_preemption(
        baseline_llm_generator, test_llm_generator, batch_size):
    """Verify block manager produces same outputs even when there is preemption.

    This constructs two LLM, each with limited number of GPU blocks. The limit
    is decided such that as the sequences in the batch grow, sequences must be
    preempted and removed from cache.

    If the output token ids are equivalent, then we have confidence that the KV
    cache is not corrupted.

    NOTE: We want a significant number of generated tokens so that any incorrect
    KV mapping has time to build up error.

    NOTE(Kuntai): Though we have removed block manager v1, this test is still
    useful as it asserts the behavior of block manager v2 (now it is called 
    SelfAttnBlockSpaceManager) is the same when swapping / preemption, so we  
    keep this test.
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

    print('Getting token ids from block manager')
    baseline_token_ids = get_token_ids_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

    print('Getting token ids from block manager, with preemption')
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
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "enable_prefix_caching": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "enable_prefix_caching": True,
    "preemption_mode": "swap"
}, {
    "enable_prefix_caching": True,
    "preemption_mode": "recompute"
}])
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


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "facebook/opt-125m",

        # skip cuda graph creation for fast test.
        "enforce_eager": True,

        # we keep the blocks small, so that hit eviction quickly
        "max_model_len": 48,
        "block_size": 16,
        "num_gpu_blocks_override": 3,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "enable_prefix_caching": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "enable_prefix_caching": True,
}])
@pytest.mark.parametrize("seed", [1])
def test_auto_prefix_caching_after_evition_start(baseline_llm_generator,
                                                 test_llm_generator):
    """Verify block manager v2 with auto prefix caching could works normal
    even when eviction started.
    With APC enabled, all blocks are held by native block at the beginning.
    Then blocks are managed by evictor instead. If cache hit at the evitor's
    block, then it could be reused, or we need to recompute its kv cache.
    """
    output_len = 10
    temperature = 0.0

    prompts = [
        "You are a helpful assistant. Please answer truthfully and write "
        "out your thinking step by step to be sure you get the right answer. "
        "If you make a mistake, attempt to correct it. who are you?",
        "You are a helpful assistant. Please answer truthfully and write out "
        "your thinking step by step to be sure you get the right answer. You "
        "are helpful and harmless and you follow ethical guidelines. "
        "who are you?"
    ]

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
