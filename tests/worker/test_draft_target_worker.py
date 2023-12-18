
import torch
import random
import pytest

from vllm.worker.draft_target_worker import DraftTargetWorker
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SequenceGroupMetadata

from .utils import (mock_worker, create_seq_group_metadata_from_prompts,
                    create_batch, create_sampler_output_list)

from unittest.mock import MagicMock


def test_get_all_seq_ids():
    """Verify get_all_seq_ids extracts all seq ids.
    """
    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock())

    expected_seq_ids = list(range(10)) + list(range(100, 110))

    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=str(seq_id),
            is_prompt=True,
            is_chunked_prefill=False,
            seq_data={
                seq_id: MagicMock(),
            },
            sampling_params=MagicMock(),
            block_tables={
                seq_id: MagicMock(),
            },
            lora_request=None,
        ) for seq_id in expected_seq_ids
    ]

    actual_seq_ids = worker._get_all_seq_ids(seq_group_metadata_list)  # pylint: disable=protected-access
    assert actual_seq_ids == expected_seq_ids


@pytest.mark.parametrize('num_target_seq_ids', [100])
def test_create_target_seq_id_iterator(num_target_seq_ids: int):
    """Assert all target seq ids are greater than input seq ids.
    """
    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock())

    all_seq_ids = [
        [1, 3, 5, 7],
        list(range(100)) + [0],
        [100],
    ]

    for seq_ids in all_seq_ids:
        max_seq_id = max(seq_ids)
        iterator = worker._create_target_seq_id_iterator(seq_ids)  # pylint: disable=protected-access
        for _ in range(num_target_seq_ids):
            assert next(iterator) > max_seq_id


@pytest.mark.parametrize('k', [1, 2, 6])
def test_get_token_ids_to_score(k: int):
    """Verify DraftTargetWorker correctly determines which token ids need
    to be scored.
    """
    proposal_token_ids = torch.tensor(
        list(range(k)),
        dtype=torch.int64,
        device='cuda',
    )

    expected_output = [
        [],
    ]
    for i in range(proposal_token_ids.shape[0]):
        expected_output.append(proposal_token_ids[:i + 1].tolist())

    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock())
    actual_output = worker._get_token_ids_to_score(proposal_token_ids)  # pylint: disable=protected-access

    actual_output = [
        x.tolist() if isinstance(x, torch.Tensor) else x for x in actual_output
    ]

    assert actual_output == expected_output


@pytest.mark.parametrize('k', [1, 2, 6])
def test_create_single_target_seq_group_metadata(k: int):
    """Verify correct creation of a target seq group metadata.
    """

    prompt_tokens = [1, 2, 3]
    prev_output_tokens = [4, 5, 6]

    token_ids = list(range(k))

    num_tokens_processed = len(prompt_tokens) + len(prev_output_tokens) - 1

    final_seq_len = len(prompt_tokens) + len(prev_output_tokens) + len(
        token_ids)

    block_size = 32
    input_seq_group_metadata = create_seq_group_metadata_from_prompts(
        [prompt_tokens], 2048 // block_size, block_size, [final_seq_len],
        [prev_output_tokens], [num_tokens_processed])[0]

    input_seq_id = list(input_seq_group_metadata.seq_data.keys())[0]
    target_seq_id = 100

    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock())
    output = worker._create_single_target_seq_group_metadata(  # pylint: disable=protected-access
        input_seq_group_metadata,
        input_seq_id,
        target_seq_id,
        token_ids,
    )

    assert output.request_id == input_seq_group_metadata.request_id
    assert len(output.seq_data) == 1
    assert output.seq_data[target_seq_id].get_prompt_token_ids(
    ) == prompt_tokens
    assert output.seq_data[target_seq_id].get_output_token_ids(
    ) == prev_output_tokens + token_ids

    assert output.seq_data[target_seq_id].get_num_processed_token_ids(
    ) == num_tokens_processed + k

    assert len(output.block_tables) == 1
    assert output.block_tables[
        target_seq_id] == input_seq_group_metadata.block_tables[input_seq_id]


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_calls_draft_model(k: int, batch_size: int):
    """Verify that the DraftTargetWorker calls the draft model with correct
    inputs. Everything else is mocked out.
    """

    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler)

    exception_secret = 'artifical stop'
    draft_worker.execute_model.side_effect = ValueError(exception_secret)

    execute_model_data, _, _ = create_batch(batch_size, k)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_data)

    call_args_list = draft_worker.execute_model.call_args_list
    assert len(call_args_list) == 1

    for args, _ in call_args_list:
        (actual_execute_model_data, ) = args
        assert actual_execute_model_data == execute_model_data


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_calls_target_model(k: int, batch_size: int):
    """Verify that the DraftTargetWorker calls the target model with correct
    inputs. Everything else is mocked out.
    """
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler)

    vocab_size = 32_000

    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(k, batch_size),
                                    dtype=torch.int64,
                                    device='cuda')
    draft_token_probs = torch.rand(k,
                                   batch_size,
                                   vocab_size,
                                   dtype=torch.float32,
                                   device='cuda')

    draft_output = create_sampler_output_list(draft_token_ids,
                                              draft_token_probs)
    draft_worker.execute_model.return_value = draft_output
    execute_model_data, prompts, prev_output_tokens = create_batch(
        batch_size, k)

    exception_secret = 'artifical stop'
    target_worker.execute_model.side_effect = ValueError(exception_secret)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_data)

    seen_contexts = []

    call_args_list = target_worker.execute_model.call_args_list
    assert len(call_args_list) == 1
    for args, _ in call_args_list:
        (target_execute_model_data, ) = args

        assert len(target_execute_model_data.seq_group_metadata_list) == (
            k + 1) * batch_size
        for seq_group_metadata in (
                target_execute_model_data.seq_group_metadata_list):
            for seq_data in seq_group_metadata.seq_data.values():
                seen_contexts.append(seq_data.get_token_ids())

    expected_seen_contexts = []

    for prompt, prev_generated, draft_tokens in zip(
            prompts, prev_output_tokens,
            draft_token_ids.transpose(0, 1).tolist()):

        for i in range(len(draft_tokens) + 1):
            expected_seen_contexts.append(prompt + prev_generated +
                                          draft_tokens[:i])

    seen_contexts.sort()
    expected_seen_contexts.sort()
    assert expected_seen_contexts == seen_contexts


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_calls_rejection_sampler(k: int, batch_size: int):
    """Verify that the DraftTargetWorker calls the rejection sampler with
    correct inputs. Everything else is mocked out.
    """
    vocab_size = 32_000

    draft_worker = mock_worker(vocab_size)
    target_worker = mock_worker(vocab_size)
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler)

    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(k, batch_size),
                                    dtype=torch.int64,
                                    device='cuda')
    draft_token_probs = torch.rand(k,
                                   batch_size,
                                   vocab_size,
                                   dtype=torch.float32,
                                   device='cuda')

    draft_output = create_sampler_output_list(draft_token_ids,
                                              draft_token_probs)
    draft_worker.execute_model.return_value = draft_output
    execute_model_data, _, _ = create_batch(batch_size, k)

    target_token_ids = torch.randint(low=0,
                                     high=vocab_size,
                                     size=(1, batch_size * (k + 1)),
                                     dtype=torch.int64,
                                     device='cuda')
    target_token_probs = torch.rand(1,
                                    batch_size * (k + 1),
                                    vocab_size,
                                    dtype=torch.float32,
                                    device='cuda')
    target_output = create_sampler_output_list(target_token_ids,
                                               target_token_probs)

    target_worker.execute_model.return_value = target_output[0]

    exception_secret = 'artifical stop'
    rejection_sampler.side_effect = ValueError(exception_secret)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_data)

    assert len(rejection_sampler.call_args_list) == 1
    args, _ = rejection_sampler.call_args_list[0]
    (actual_proposal_scores, actual_bonus_token_ids, actual_proposal_probs,
     actual_proposal_token_ids) = args

    assert torch.equal(actual_bonus_token_ids,
                       target_token_ids.reshape(batch_size, k + 1)[:, -1:])
    assert torch.equal(
        actual_proposal_scores,
        target_token_probs.reshape(batch_size, k + 1, -1)[:, :-1])
    assert torch.equal(actual_proposal_token_ids,
                       draft_token_ids.transpose(0, 1))
    assert torch.equal(actual_proposal_probs,
                       draft_token_probs.transpose(0, 1))


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_formats_output(k: int, batch_size: int):
    """Verify that the DraftTargetWorker formats rejection sampler output
    correctly. Everything else is mocked out.
    """
    vocab_size = 32_000

    draft_worker = mock_worker(vocab_size)
    target_worker = mock_worker(vocab_size)
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler)

    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(k, batch_size),
                                    dtype=torch.int64,
                                    device='cuda')
    draft_token_probs = torch.rand(k,
                                   batch_size,
                                   vocab_size,
                                   dtype=torch.float32,
                                   device='cuda')

    draft_output = create_sampler_output_list(draft_token_ids,
                                              draft_token_probs)
    draft_worker.execute_model.return_value = draft_output
    execute_model_data, _, _ = create_batch(batch_size, k)

    target_token_ids = torch.randint(low=0,
                                     high=vocab_size,
                                     size=(1, batch_size * (k + 1)),
                                     dtype=torch.int64,
                                     device='cuda')
    target_token_probs = torch.rand(1,
                                    batch_size * (k + 1),
                                    vocab_size,
                                    dtype=torch.float32,
                                    device='cuda')
    target_output = create_sampler_output_list(target_token_ids,
                                               target_token_probs)

    target_worker.execute_model.return_value = target_output[0]

    rejection_sampler_output = torch.randint(low=0,
                                             high=vocab_size,
                                             size=(batch_size, k + 1),
                                             dtype=torch.int64,
                                             device='cuda')
    for i in range(batch_size):
        rejection_sampler_output[i][-random.randint(0, k + 1):] = -1

    rejection_sampler.return_value = rejection_sampler_output

    output = worker.execute_model(execute_model_data)

    expected_output = create_sampler_output_list(
        rejection_sampler_output.transpose(0, 1), [None for _ in range(k + 1)])

    seq_ids = [
        next(iter(seq_group_metadata.seq_data.keys()))
        for seq_group_metadata in execute_model_data.seq_group_metadata_list
    ]
    actual_output_by_seq = {seq_id: [] for seq_id in seq_ids}
    expected_output_by_seq = {seq_id: [] for seq_id in seq_ids}

    for step in output:
        for seq_group in step:
            for sample in seq_group.samples:
                seq_id = sample.parent_seq_id
                actual_output_by_seq[seq_id].append(sample)

    for step in expected_output:
        for seq_group in step:
            for sample in seq_group.samples:
                seq_id = sample.parent_seq_id
                expected_output_by_seq[seq_id].append(sample)

    all_seen_seq_ids = set(
        list(actual_output_by_seq.keys()) +
        list(expected_output_by_seq.keys()))
    for seq_id in all_seen_seq_ids:
        actual_by_step = actual_output_by_seq[seq_id]
        expected_by_step = expected_output_by_seq[seq_id]

        for i in range(k + 1):
            if i >= len(actual_by_step):
                assert expected_by_step[i].output_token == -1
                continue
            assert actual_by_step[i].output_token == expected_by_step[
                i].output_token
            assert actual_by_step[i].logprobs == expected_by_step[i].logprobs
