import torch
import random
import pytest

from vllm.worker.spec_decode.draft_target_worker import DraftTargetWorker, calculate_gpu_blocks
from vllm.worker.spec_decode.util import SpeculativeProposals
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SequenceGroupMetadata

from .utils import mock_worker, create_batch, ExecuteModelData, create_seq_group_metadata_from_prompts, create_sampler_output_list
#from .utils import (mock_worker,
#                    create_seq_group_metadata_from_prompts, create_batch,
#                    create_sampler_output_list)

from unittest.mock import MagicMock


def test_get_all_seq_ids():
    """Verify get_all_seq_ids extracts all seq ids.
    """
    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock(),
                               MagicMock())

    expected_seq_ids = list(range(10)) + list(range(100, 110))

    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=str(seq_id),
            is_prompt=True,
            #is_chunked_prefill=False,
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
    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock(),
                               MagicMock())

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

    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock(),
                               MagicMock())
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

    worker = DraftTargetWorker(mock_worker(), mock_worker(), MagicMock(),
                               MagicMock())
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

    #assert output.seq_data[target_seq_id].get_num_processed_token_ids(
    #) == num_tokens_processed + k

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
    max_model_len = 2048
    draft_worker = mock_worker(max_model_len=max_model_len)
    target_worker = mock_worker(max_model_len=max_model_len)
    rejection_sampler = MagicMock()
    metrics_collector = MagicMock()
    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    exception_secret = 'artifical stop'
    draft_worker.get_spec_proposals.side_effect = ValueError(exception_secret)

    execute_model_data, _, _ = create_batch(batch_size, k)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(**execute_model_data.to_dict(), num_spec_tokens=k)

    call_args_list = draft_worker.get_spec_proposals.call_args_list
    assert len(call_args_list) == 1

    for args, _ in call_args_list:
        #(actual_execute_model_data, actual_k, actual_max_model_len) = args
        (seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out,
         blocks_to_copy, actual_k, actual_max_model_len) = args
        actual_execute_model_data = ExecuteModelData(seq_group_metadata_list,
                                                     blocks_to_swap_in,
                                                     blocks_to_swap_out,
                                                     blocks_to_copy)
        assert actual_execute_model_data == execute_model_data
        assert actual_k == k
        assert actual_max_model_len == max_model_len


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
    metrics_collector = MagicMock()

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    vocab_size = 32_000

    proposal_token_ids = torch.randint(low=0,
                                       high=vocab_size,
                                       size=(batch_size, k),
                                       dtype=torch.int64,
                                       device='cuda')
    proposal_probs = torch.rand(batch_size,
                                k,
                                vocab_size,
                                dtype=torch.float32,
                                device='cuda')
    proposal_lens = torch.ones(batch_size, dtype=torch.int64, device='cuda') * k

    execute_model_data, prompts, prev_output_tokens = create_batch(
        batch_size, k)

    draft_worker.get_spec_proposals.return_value = SpeculativeProposals(
        #spec_seqs=execute_model_data.seq_group_metadata_list,
        #non_spec_seqs=[],
        #all_seqs=execute_model_data.seq_group_metadata_list,
        #original_indices=torch.arange(batch_size),
        proposal_token_ids=proposal_token_ids,
        proposal_probs=proposal_probs,
        proposal_lens=proposal_lens)

    exception_secret = 'artifical stop'
    target_worker.execute_model.side_effect = ValueError(exception_secret)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(**execute_model_data.to_dict(), num_spec_tokens=k)

    seen_contexts = []

    call_args_list = target_worker.execute_model.call_args_list
    assert len(call_args_list) == 1
    for args, kwargs in call_args_list:
        #(target_execute_model_data, ) = args
        #(seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy, _) = args
        #target_execute_model_data = ExecuteModelData(seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        target_execute_model_data = ExecuteModelData.from_dict(kwargs)

        assert len(target_execute_model_data.seq_group_metadata_list) == (
            k + 1) * batch_size
        for seq_group_metadata in (
                target_execute_model_data.seq_group_metadata_list):
            for seq_data in seq_group_metadata.seq_data.values():
                seen_contexts.append(seq_data.get_token_ids())

    expected_seen_contexts = []

    for prompt, prev_generated, draft_tokens in zip(
            prompts, prev_output_tokens, proposal_token_ids.tolist()):

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
    metrics_collector = MagicMock()
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    proposal_token_ids = torch.randint(low=0,
                                       high=vocab_size,
                                       size=(batch_size, k),
                                       dtype=torch.int64,
                                       device='cuda')
    proposal_probs = torch.rand(batch_size,
                                k,
                                vocab_size,
                                dtype=torch.float32,
                                device='cuda')

    proposal_lens = torch.ones(batch_size, dtype=torch.int64, device='cuda') * k

    execute_model_data, _, _ = create_batch(batch_size, k)

    draft_worker.get_spec_proposals.return_value = SpeculativeProposals(
        #spec_seqs=execute_model_data.seq_group_metadata_list,
        #non_spec_seqs=[],
        #all_seqs=execute_model_data.seq_group_metadata_list,
        #original_indices=torch.arange(batch_size),
        proposal_token_ids=proposal_token_ids,
        proposal_probs=proposal_probs,
        proposal_lens=proposal_lens)

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
        worker.execute_model(**execute_model_data.to_dict(), num_spec_tokens=k)

    assert len(rejection_sampler.call_args_list) == 1
    args, _ = rejection_sampler.call_args_list[0]
    (actual_proposal_scores, actual_bonus_token_ids, actual_proposal_probs,
     actual_proposal_token_ids) = args

    assert torch.equal(actual_bonus_token_ids,
                       target_token_ids.reshape(batch_size, k + 1)[:, -1:])
    assert torch.equal(
        actual_proposal_scores,
        target_token_probs.reshape(batch_size, k + 1, -1)[:, :-1])
    assert torch.equal(actual_proposal_token_ids, proposal_token_ids)
    assert torch.equal(actual_proposal_probs, proposal_probs)


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
    metrics_collector = MagicMock()
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    proposal_token_ids = torch.randint(low=0,
                                       high=vocab_size,
                                       size=(batch_size, k),
                                       dtype=torch.int64,
                                       device='cuda')
    proposal_probs = torch.rand(batch_size,
                                k,
                                vocab_size,
                                dtype=torch.float32,
                                device='cuda')

    proposal_lens = torch.ones(batch_size, dtype=torch.int64, device='cuda') * k

    execute_model_data, _, _ = create_batch(batch_size, k)

    draft_worker.get_spec_proposals.return_value = SpeculativeProposals(
        proposal_token_ids=proposal_token_ids,
        proposal_probs=proposal_probs,
        proposal_lens=proposal_lens)

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
        minimum_accepted_tokens = 1
        rejection_sampler_output[i][
            -random.randint(minimum_accepted_tokens, k + 1):] = -1

    rejection_sampler.return_value = rejection_sampler_output

    output = worker.execute_model(**execute_model_data.to_dict(),
                                  num_spec_tokens=k)

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


@pytest.mark.parametrize('k', [1, 2])
@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('returns_metrics', [True, False])
@torch.inference_mode()
def test_collects_metrics(k: int, batch_size: int, returns_metrics: bool):
    """TODO
    """
    vocab_size = 32_000

    draft_worker = mock_worker(vocab_size)
    target_worker = mock_worker(vocab_size)
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)
    worker.init_model()

    proposal_token_ids = torch.randint(low=0,
                                       high=vocab_size,
                                       size=(batch_size, k),
                                       dtype=torch.int64,
                                       device='cuda')
    proposal_probs = torch.rand(batch_size,
                                k,
                                vocab_size,
                                dtype=torch.float32,
                                device='cuda')

    proposal_lens = torch.ones(batch_size, dtype=torch.int64, device='cuda') * k

    execute_model_data, _, _ = create_batch(batch_size, k)

    draft_worker.get_spec_proposals.return_value = SpeculativeProposals(
        proposal_token_ids=proposal_token_ids,
        proposal_probs=proposal_probs,
        proposal_lens=proposal_lens)

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
        minimum_accepted_tokens = 1
        rejection_sampler_output[i][
            -random.randint(minimum_accepted_tokens, k + 1):] = -1

    rejection_sampler.return_value = rejection_sampler_output

    mock_rejsample_metrics = "cade make this a dtw metrics" if returns_metrics else None
    metrics_collector.maybe_collect_rejsample_metrics.return_value = mock_rejsample_metrics

    output = worker.execute_model(**execute_model_data.to_dict(),
                                  num_spec_tokens=k)
    assert output[0].draft_target_worker_metrics == mock_rejsample_metrics

    call_args_list = metrics_collector.maybe_collect_rejsample_metrics.call_args_list
    assert len(call_args_list) == 1
    args, kwargs = call_args_list[0]
    assert args[0] == k or kwargs.get('k', -1) == k


@pytest.mark.parametrize('k', [0])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_k_equals_zero(k: int, batch_size: int):
    """Verify that the DraftTargetWorker calls the draft and target workers
    when k is zero. This happens during prefill.
    """
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    execute_model_data, prompts, prev_output_tokens = create_batch(
        batch_size, k, prev_output_token_len=0)

    out = worker.execute_model(**execute_model_data.to_dict(),
                               num_spec_tokens=k)

    assert len(out) == 1, f"expected only one token output when {k=}"
    assert out[0].probs == None, "expect gpu tensor references to be None"
    assert out[
        0].sampled_tokens == None, "expect gpu tensor references to be None"

    assert draft_worker.execute_model.called_once_with(
        **execute_model_data.to_dict())
    assert target_worker.execute_model.called_once_with(
        **execute_model_data.to_dict())


@pytest.mark.parametrize('k', [0, 5])
@pytest.mark.parametrize('batch_size', [0])
@torch.inference_mode()
def test_empty_input_batch(k: int, batch_size: int):
    """Verify that the DraftTargetWorker calls the draft and target workers
    when the input batch is empty. This can happen if the engine communicates
    to the workers information without scheduling a batch.
    """
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    execute_model_data, prompts, prev_output_tokens = create_batch(
        batch_size, k, prev_output_token_len=0)

    out = worker.execute_model(**execute_model_data.to_dict(),
                               num_spec_tokens=k)

    assert len(out) == 1, f"expected only one token output when {k=}"
    assert out[0].probs == None, "expect gpu tensor references to be None"
    assert out[
        0].sampled_tokens == None, "expect gpu tensor references to be None"

    assert draft_worker.execute_model.called_once_with(
        **execute_model_data.to_dict())
    assert target_worker.execute_model.called_once_with(
        **execute_model_data.to_dict())


@torch.inference_mode()
def test_init_model():
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    worker.init_model()

    assert draft_worker.init_model.called_once()
    assert draft_worker.include_gpu_probs_tensors.called_once()

    assert target_worker.init_model.called_once()
    assert target_worker.include_gpu_probs_tensors.called_once()

    assert metrics_collector.init_gpu_tensors.called_once()
    assert rejection_sampler.init_gpu_tensors.called_once()


@torch.inference_mode()
def test_init_cache_engine():
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    cache_config = MagicMock()

    worker.init_cache_engine(cache_config)

    assert draft_worker.init_cache_engine.called_once_with(cache_config)
    assert target_worker.init_cache_engine.called_once_with(cache_config)


@pytest.mark.parametrize('available_gpu_blocks', [1, 1024])
@pytest.mark.parametrize('available_cpu_blocks', [500])
@pytest.mark.parametrize('target_kv_size_bytes', [2 * 2 * 4096])
@pytest.mark.parametrize('draft_kv_size_bytes', [0, 2 * 2 * 768, 2 * 2 * 4096])
@torch.inference_mode()
def test_profile_num_available_blocks(available_gpu_blocks: int,
                                      available_cpu_blocks: int,
                                      target_kv_size_bytes: int,
                                      draft_kv_size_bytes: int):
    draft_worker = mock_worker()
    target_worker = mock_worker()
    rejection_sampler = MagicMock()
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock()

    target_worker.profile_num_available_blocks.return_value = (
        available_gpu_blocks, available_cpu_blocks)
    target_worker.get_kv_size_bytes.return_value = target_kv_size_bytes
    draft_worker.get_kv_size_bytes.return_value = draft_kv_size_bytes

    worker = DraftTargetWorker(draft_worker, target_worker, rejection_sampler,
                               metrics_collector)

    # These values do not directly impact the adjusted block size calculation,
    # so they can be fixed.
    gpu_memory_utilization = 0.9
    cpu_swap_space = 100
    block_size = 16

    num_gpu_blocks, num_cpu_blocks = worker.profile_num_available_blocks(
        block_size, gpu_memory_utilization, cpu_swap_space)

    assert target_worker.profile_num_available_blocks.called_once_with(
        block_size, gpu_memory_utilization, cpu_swap_space)
    assert num_cpu_blocks == available_cpu_blocks

    assert num_gpu_blocks == calculate_gpu_blocks(target_kv_size_bytes,
                                                  draft_kv_size_bytes,
                                                  available_gpu_blocks)


@pytest.mark.parametrize('available_gpu_blocks',
                         list(range(20)) + [1024, 1024**2])
@pytest.mark.parametrize('target_kv_size_bytes', [2 * 2 * 4096, 2 * 2 * 8192])
@pytest.mark.parametrize('draft_kv_size_bytes', [0, 2 * 2 * 768, 2 * 2 * 4096])
@torch.inference_mode()
def test_calculate_gpu_blocks(available_gpu_blocks: int,
                              target_kv_size_bytes: int,
                              draft_kv_size_bytes: int):
    num_blocks = calculate_gpu_blocks(target_kv_size_bytes,
                                      draft_kv_size_bytes,
                                      available_gpu_blocks)
    assert (num_blocks * target_kv_size_bytes) + (
        num_blocks * draft_kv_size_bytes) <= (available_gpu_blocks *
                                              target_kv_size_bytes)
