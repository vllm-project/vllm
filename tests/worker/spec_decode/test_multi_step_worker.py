import torch
import random
import pytest
from unittest.mock import MagicMock

from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed

from .utils import (create_execute_model_data, create_worker,
                    create_seq_group_metadata_from_prompts, zero_kv_cache,
                    patch_execute_model_with_seeds,
                    assert_logprobs_dict_allclose)


@pytest.mark.parametrize('num_steps', list(range(1, 17)))
def test_assert_enough_kv_space(num_steps: int):
    """Test that the multi step worker checks for sufficient space in the KV
    cache. It should throw if it cannot run all the steps.
    """
    block_size = 16
    num_gpu_blocks = 2048 // block_size

    prompts = [
        list(range(block_size * 3)),
        list(range(block_size * 2)),
    ]

    prev_output_tokens = [
        list(range(block_size * 1)),
        list(range(block_size * 2)),
    ]

    final_seq_lens = [
        len(prompt + output) + num_steps
        for prompt, output in zip(prompts, prev_output_tokens)
    ]

    inputs = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_seq_lens,
        continuations=prev_output_tokens)

    assert_enough_kv_space = MultiStepWorker._assert_enough_kv_space  # pylint: disable=protected-access
    worker = MagicMock()
    worker.model_runner.block_size = block_size

    for seq_group_metadata in inputs:
        original_block_tables = seq_group_metadata.block_tables

        # No exception.
        assert_enough_kv_space(worker, inputs, num_steps)

        seq_group_metadata.block_tables = {
            seq_id: []
            for seq_id, physical_blocks in original_block_tables.items()
        }

        # Expect exception.
        with pytest.raises(ValueError,
                           match='times but found insufficient KV space for'):
            assert_enough_kv_space(worker, inputs, num_steps)

        seq_group_metadata.block_tables = original_block_tables


@torch.inference_mode()
def test_same_output_for_single_step():
    """Verify the multi step worker produces the same output as the normal
    worker for num_steps=1.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'

    block_size = 32
    num_gpu_blocks = 2048 // block_size
    multi_step_worker = create_worker(
        MultiStepWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    worker = create_worker(
        Worker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    multi_step_worker.model_runner = worker.model_runner
    multi_step_worker.cache_engine = worker.cache_engine

    num_steps = 1

    prompts = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]

    final_seq_lens = [len(prompt) + num_steps for prompt in prompts]

    multi_step_execute_model_data = create_execute_model_data(
        seq_group_metadata_list=create_seq_group_metadata_from_prompts(
            prompts, num_gpu_blocks, block_size,
            final_seq_lens=final_seq_lens))

    single_step_execute_model_data = create_execute_model_data(
        seq_group_metadata_list=create_seq_group_metadata_from_prompts(
            prompts, num_gpu_blocks, block_size,
            final_seq_lens=final_seq_lens))

    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    actual_output = multi_step_worker.execute_model_multi_step(
        **multi_step_execute_model_data.to_dict(), num_steps=num_steps)
    assert len(actual_output) == num_steps
    actual_output = actual_output[0]

    zero_kv_cache(worker.cache_engine)
    set_random_seed(seed)
    expected_output = worker.execute_model(
        **single_step_execute_model_data.to_dict(), )

    actual_token_ids = [
        output.samples[0].output_token for output in actual_output
    ]
    actual_logprobs = [output.samples[0].logprobs for output in actual_output]

    expected_token_ids = [
        output.samples[0].output_token for output in expected_output
    ]
    expected_logprobs = [
        output.samples[0].logprobs for output in expected_output
    ]

    assert actual_token_ids == expected_token_ids

    print(f'{actual_logprobs=}')
    print(f'{expected_logprobs=}')
    assert_logprobs_dict_allclose(actual_logprobs, expected_logprobs)


@torch.inference_mode()
def test_same_output_for_multi_step():
    """Verify the multi-step worker produces the same output as the normal
    worker when num_steps > 1. This test runs the multi-step worker once, and
    then runs the worker num_steps times, and compares the output.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'

    block_size = 16
    num_gpu_blocks = 2048 // block_size
    multi_step_worker = create_worker(
        MultiStepWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    worker = create_worker(
        Worker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    # Make sure we go over the block boundary.
    num_steps = block_size + 1

    random.seed(seed)
    prompts = [[
        random.randint(0, 1000) for _ in range(random.randint(10, 20))
    ] for _ in range(10)]

    final_seq_lens = [len(prompt) + num_steps for prompt in prompts]

    rand_seeds = list(random.randint(0, 100) for _ in range(num_steps))
    multi_step_worker.execute_model = patch_execute_model_with_seeds(
        multi_step_worker, rand_seeds)
    worker.execute_model = patch_execute_model_with_seeds(worker, rand_seeds)

    continuations = [[1] for _ in prompts]
    execute_model_data = create_execute_model_data(
        create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=continuations,
            final_seq_lens=final_seq_lens), )

    # Run multi-step.
    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    multi_step_output = multi_step_worker.execute_model_multi_step(
        **execute_model_data.to_dict(), num_steps=num_steps)

    # Run single-step repeatedly.
    zero_kv_cache(worker.cache_engine)
    single_step_output = []
    continuations = [[1] for _ in prompts]
    set_random_seed(seed)

    for _ in multi_step_output:

        execute_model_data = create_execute_model_data(
            create_seq_group_metadata_from_prompts(
                prompts,
                num_gpu_blocks,
                block_size,
                continuations=continuations,
                final_seq_lens=final_seq_lens))

        single_step_output.append(
            worker.execute_model(**execute_model_data.to_dict(), ))

        # Append output tokens to new sequence data.
        for i, seq_group_output in enumerate(single_step_output[-1]):
            continuations[i].append(seq_group_output.samples[0].output_token)

    # Get token ids and logprobs for comparison.
    multi_step_output_logprobs = [[] for _ in prompts]
    single_step_output_logprobs = [[] for _ in prompts]

    multi_step_output_token_ids = [[] for _ in prompts]
    single_step_output_token_ids = [[] for _ in prompts]
    for i, _ in enumerate(prompts):
        for multi_step, single_step in zip(multi_step_output,
                                           single_step_output):
            multi_step_output_token_ids[i].append(
                multi_step[i].samples[0].output_token)
            single_step_output_token_ids[i].append(
                single_step[i].samples[0].output_token)

            multi_step_output_logprobs[i].append(
                multi_step[i].samples[0].logprobs)
            single_step_output_logprobs[i].append(
                single_step[i].samples[0].logprobs)

    # Print per-sequence token ids
    for i, (multi_step_tokens, single_step_tokens) in enumerate(
            zip(multi_step_output_token_ids, single_step_output_token_ids)):
        print(f'{i=} {multi_step_tokens=}')
        print(f'{i=} {single_step_tokens=}')
        print(f'{i=} equal {multi_step_tokens == single_step_tokens}')

    # Assert token ids are equal.
    for multi_step_tokens, single_step_tokens in zip(
            multi_step_output_token_ids, single_step_output_token_ids):
        assert multi_step_tokens == single_step_tokens

    # Assert logprobs are equal.
    for multi_step_logprobs, single_step_logprobs in zip(
            multi_step_output_logprobs, single_step_output_logprobs):
        assert_logprobs_dict_allclose(multi_step_logprobs,
                                      single_step_logprobs)
