import random
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.utils import set_random_seed
from vllm.sequence import ExecuteModelRequest, Logprob, SamplerOutput
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker

from .utils import (assert_logprobs_dict_allclose, create_batch, create_sampler_output_list,
                    create_seq_group_metadata_from_prompts, create_worker,
                    patch_execute_model_with_seeds, zero_kv_cache)


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

    final_prompt_lens = [
        len(prompt + output) + num_steps
        for prompt, output in zip(prompts, prev_output_tokens)
    ]

    inputs = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens,
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
        model_runner_cls=TP1DraftModelRunner,
    )
    worker = create_worker(
        Worker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    # multi_step_worker.model_runner = worker.model_runner
    # multi_step_worker.cache_engine = worker.cache_engine

    num_steps = 1

    prompts = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]

    final_prompt_lens = [len(prompt) + num_steps for prompt in prompts]

    multi_step_seq_group = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    actual_output, _ = multi_step_worker.sampler_output(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=multi_step_seq_group),
        sample_len=num_steps, seq_ids_with_bonus_token_in_last_step=set())
    assert len(actual_output) == num_steps
    actual_output = actual_output[0]

    single_step_seq_group = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    zero_kv_cache(worker.cache_engine)
    set_random_seed(seed)
    expected_output = worker.execute_model(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=single_step_seq_group))[0]

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

@pytest.mark.parametrize("disable_bonus_tokens", [True, False])
@torch.inference_mode()
def test_same_output_for_multi_step(disable_bonus_tokens:bool):
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
        model_runner_cls=TP1DraftModelRunner,
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

    # For the case of disable_bonus_tokens = False we execute (num_steps + 1)
    # iterations in order to accomodate the draft token. 
    final_prompt_lens = [len(prompt) + num_steps + 1 for prompt in prompts]

    rand_seeds = list(random.randint(0, 100) for _ in range(num_steps + 1))
    multi_step_worker.execute_model = patch_execute_model_with_seeds(
        multi_step_worker, rand_seeds)
    worker.execute_model = patch_execute_model_with_seeds(worker, rand_seeds)

    # Continuations to use for the single step worker.
    continuations = [[1] for _ in prompts]
    # Continuations to use for the multi step step worker.
    multi_step_worker_continuations = [[1] for _ in prompts]
    indices_of_seq_with_bonus_tokens = []
    if not disable_bonus_tokens:
        # Bonus tokens are enabled. For half of the sequences, add bonus
        # tokens. Make one forward pass of the model and add the generated
        # tokens to the continuations of the sequences selected for bonus tokens.
        # Do nothing for other sequences.
        num_sequences_with_bonus_tokens = len(prompts) // 2 
        indices_of_seq_with_bonus_tokens= random.sample(
            range(len(prompts)), num_sequences_with_bonus_tokens)
        seq_group_metadata_list = create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=continuations,
            final_prompt_lens=final_prompt_lens)
        step_output = (
            worker.execute_model(execute_model_req=ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list)))
        # Add generated tokens to the continuations for both single-step
        # and multi-step workers for the sequences selected for bonus tokens.
        for i, seq_group_output in enumerate(step_output[0].outputs):
            if i in indices_of_seq_with_bonus_tokens:
                multi_step_worker_continuations[i].append(
                    seq_group_output.samples[0].output_token)
                continuations[i].append(
                    seq_group_output.samples[0].output_token)
    
    # Run single-step repeatedly.
    zero_kv_cache(worker.cache_engine)
    single_step_output: List[SamplerOutput] = []
    set_random_seed(seed)
    for step in range(num_steps):
        seq_group_metadata_list = create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=continuations,
            final_prompt_lens=final_prompt_lens)
        single_step_output.extend(
            worker.execute_model(execute_model_req=ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list)))
        # Append output tokens to new sequence data.
        for i, seq_group_output in enumerate(single_step_output[-1]):
            continuations[i].append(seq_group_output.samples[0].output_token)
    random.seed(seed)
    # Run multi-step.
    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        continuations=multi_step_worker_continuations,
        final_prompt_lens=final_prompt_lens)
    multi_step_output, _ = multi_step_worker.sampler_output(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list),
        sample_len=num_steps,
        seq_ids_with_bonus_token_in_last_step=indices_of_seq_with_bonus_tokens)


    # Get token ids and logprobs for comparison.
    multi_step_output_logprobs: List[List[Dict[int,
                                               Logprob]]] = [[]
                                                             for _ in prompts]
    single_step_output_logprobs: List[List[Dict[int,
                                                Logprob]]] = [[]
                                                              for _ in prompts]

    multi_step_output_token_ids: List[List[int]] = [[] for _ in prompts]
    single_step_output_token_ids: List[List[int]] = [[] for _ in prompts]
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


@torch.inference_mode()
def test_draft_proposals_full_speculation_len():
    """Verify Top1Proposer correctly handles case where all sequences
    can speculate.
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device = 'cuda:0'

    draft_worker = MagicMock()
    proposer = Top1Proposer(
        worker=draft_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=2048,
    )
    draft_worker.sampler_output.return_value = [
        SamplerOutput(
            outputs=[],
            sampled_token_probs=torch.rand(batch_size,
                                           vocab_size,
                                           device=device,
                                           dtype=torch.float32),
            logprobs=torch.rand(batch_size,
                                vocab_size,
                                device=device,
                                dtype=torch.float32),
            sampled_token_ids=torch.randint(low=0,
                                            high=vocab_size,
                                            size=(batch_size, ),
                                            device=device,
                                            dtype=torch.long),
        ) for _ in range(k)
    ], True

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=k),
        seq_ids_with_bonus_token_in_last_step=set())

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([batch_size, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([batch_size, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [k for _ in range(batch_size)]
    

@torch.inference_mode()
def test_draft_proposals_no_speculations():
    """Verify Top1Proposer correctly handles case where no sequences
    can speculate.
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device = 'cuda:0'
    prompt_len = 10

    draft_worker = MagicMock()
    proposer = Top1Proposer(
        worker=draft_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=prompt_len + k - 1,
    )

    seq_group_metadata_list, _, _ = create_batch(batch_size,
                                                 k,
                                                 prompt_len=prompt_len)

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=k),
        seq_ids_with_bonus_token_in_last_step=set())

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([batch_size, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([batch_size, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [0 for _ in range(batch_size)]


@torch.inference_mode()
def test_draft_proposals_mixed_k():
    """Verify Top1Proposer correctly handles case some sequences can
    speculate and some can't.
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device = 'cuda:0'

    small_prompt_len = 5
    long_prompt_len = 10
    prev_output_token_len = 20

    expected_num_proposal_seqs = 6
    expected_num_no_proposal_seqs = batch_size - expected_num_proposal_seqs

    prompt_len = [
        small_prompt_len for _ in range(expected_num_proposal_seqs - 1)
    ] + [long_prompt_len
         for _ in range(expected_num_no_proposal_seqs)] + [small_prompt_len]

    draft_worker = MagicMock()
    proposer = Top1Proposer(
        worker=draft_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=long_prompt_len + prev_output_token_len + k - 1,
    )

    draft_worker.sampler_output.return_value = [
        SamplerOutput(
            outputs=[],
            sampled_token_probs=torch.rand(expected_num_proposal_seqs,
                                           vocab_size,
                                           device=device,
                                           dtype=torch.float32),
            logprobs=torch.rand(expected_num_proposal_seqs,
                                vocab_size,
                                device=device,
                                dtype=torch.float32),
            sampled_token_ids=torch.randint(
                low=0,
                high=vocab_size,
                size=(expected_num_proposal_seqs, ),
                device=device,
                dtype=torch.long),
        ) for _ in range(k)
    ], True

    seq_group_metadata_list, _, _ = create_batch(
        batch_size,
        k,
        prompt_len=prompt_len,
        prev_output_token_len=prev_output_token_len,
    )

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=k),
        seq_ids_with_bonus_token_in_last_step=set())

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([batch_size, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([batch_size, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [
        k for _ in range(expected_num_proposal_seqs - 1)
    ] + [0 for _ in range(expected_num_no_proposal_seqs)] + [k]


@torch.inference_mode()
def test_expand_execute_model_request_for_bonus_tokens():
    """
    Test the expansion of the execute model request to handle bonus tokens.
    
    This test ensures that the execute model request is correctly expanded. For
    sequences with a bonus token, the expanded batch should contain two sequences:
    one with the bonus token and one without. Sequences without a bonus token
    are added unchanged to the expanded batch.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'
    block_size = 16
    num_gpu_blocks = 2048 // block_size
    # Create prompts and continuations to be used in the 
    # execute_model_request.
    prompts = [[
        random.randint(0, 1000) for _ in range(random.randint(10, 20))
    ] for _ in range(10)]
    continuations = [[random.randint(0, 1000) for _ in range(random.randint(1, 10))] for _ in prompts]
    # Create an ExecuteModelRequest using the prompts and continuations.
    execute_model_request = ExecuteModelRequest(
        seq_group_metadata_list=create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=continuations,
            final_prompt_lens=[len(prompt) + 100 for prompt in prompts]))
    # Validate that the number of sequence groups matches the number of prompts.
    assert len(
        execute_model_request.seq_group_metadata_list) == len(prompts)
    seq_id_prompt_map = {}
    seq_id_output_token_map = {}
    for seq_group in execute_model_request.seq_group_metadata_list:
        seq_id = next(iter(seq_group.seq_data.keys()))
        seq_id_prompt_map[seq_id] = \
            seq_group.seq_data[seq_id].prompt_token_ids
        seq_id_output_token_map[seq_id] = \
            seq_group.seq_data[seq_id].output_token_ids
    
    # Construct a list of seq_ids with bonus tokens.
    # The seq_ids are in the range 0 to num_prompts.
    num_sequences_with_bonus_tokens = len(prompts) // 2 
    seq_ids_with_bonus_tokens= random.sample(
        range(len(prompts)), num_sequences_with_bonus_tokens)
    # Expand the execute_model_request.
    expanded_request, indices_of_original_sequence_groups =\
        MultiStepWorker._expand_execute_model_request(
            execute_model_request,
            set(seq_ids_with_bonus_tokens))
    # Validate that the number of sequence groups is now the original number
    # plus the number of sequences with bonus tokens.
    assert len(expanded_request.seq_group_metadata_list) == \
        len(prompts) + len(seq_ids_with_bonus_tokens)
    # Iterate through the updated request and validate the following:
    # 1. If the sequence group is part of the original request, it contains all
    #    token ids including the bonus tokens.
    # 2. If the sequence group is newly added, it does not contain the bonus token.
    for index, seq_group_metadata in enumerate(
        expanded_request.seq_group_metadata_list):
        seq_id = next(iter(seq_group_metadata.seq_data.keys()))
        assert seq_group_metadata.seq_data[seq_id].prompt_token_ids ==\
            seq_id_prompt_map[seq_id] 
        if index in indices_of_original_sequence_groups:
            assert seq_group_metadata.seq_data[seq_id].output_token_ids ==\
                seq_id_output_token_map[seq_id]
        else:
            assert seq_group_metadata.seq_data[seq_id].output_token_ids ==\
                seq_id_output_token_map[seq_id][:-1]


@torch.inference_mode()
@pytest.mark.parametrize('num_steps', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 32, 64])
def test_filter_model_output(num_steps: int, batch_size: int):
    """
    Test the _filter_model_output function of the MultiStepWorker class.

    This test ensures that the _filter_model_output method correctly filters the
    model's output, retaining only the specified sequences.
    """
    vocab_size = 32_000

    target_token_ids = torch.randint(low=0,
                                     high=vocab_size,
                                     size=(batch_size , (num_steps)),
                                     dtype=torch.int64,
                                     device='cuda')
    target_token_probs = torch.rand(batch_size,
                                    num_steps,
                                    vocab_size,
                                    dtype=torch.float32,
                                    device='cuda')
    target_token_logprobs = torch.rand(batch_size,
                                       num_steps,
                                       vocab_size,
                                       dtype=torch.float32,
                                       device='cuda')
    sampler_output_list = create_sampler_output_list(target_token_ids,
                                               target_token_probs,
                                               target_token_logprobs)
    output_indices_to_retain = random.sample(
        range(num_steps), max(1, num_steps // 2))
    filtered_sampler_output_list = MultiStepWorker._filter_model_output(
        sampler_output_list, output_indices_to_retain)
    for outer_index, sampler_output in enumerate(sampler_output_list):
        filtered_sampler_output = filtered_sampler_output_list[outer_index]
        for inner_index, index_to_retain in enumerate(output_indices_to_retain):
            assert sampler_output.outputs[index_to_retain] == \
                filtered_sampler_output.outputs[inner_index]




