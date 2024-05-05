import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.metrics import (AsyncMetricsCollector,
                                      SpecDecodeWorkerMetrics)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.spec_decode_worker import (SpecDecodeWorker,
                                                 split_num_cache_blocks_evenly)

from .utils import create_batch, create_sampler_output_list, mock_worker


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_calls_draft_model(k: int, batch_size: int):
    """Verify SpecDecodeWorker calls the draft worker with correct
    inputs. Everything else is mocked out.
    """
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    exception_secret = 'artificial stop'
    draft_worker.get_spec_proposals.side_effect = ValueError(exception_secret)

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list, num_lookahead_slots=k)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_req=execute_model_req)

    call_args_list = draft_worker.get_spec_proposals.call_args_list
    assert len(call_args_list) == 1

    for args, _ in call_args_list:
        actual_execute_model_data = args[0]
        assert actual_execute_model_data == execute_model_req


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_calls_target_model(k: int, batch_size: int):
    """Verify SpecDecodeWorker calls the target model with correct
    inputs. Everything else is mocked out.
    """
    draft_worker = mock_worker(cls=MultiStepWorker, use_spec=False)
    target_worker = mock_worker(use_spec=False)
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)
    worker.init_device()

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
    proposal_lens = torch.ones(batch_size, dtype=torch.int64,
                               device='cuda') * k

    seq_group_metadata_list, prompts, prev_output_tokens = create_batch(
        batch_size, k)

    draft_worker.get_spec_proposals.return_value = SpeculativeProposals(
        proposal_token_ids=proposal_token_ids,
        proposal_probs=proposal_probs,
        proposal_lens=proposal_lens)

    exception_secret = 'artificial stop'
    target_worker.execute_model.side_effect = ValueError(exception_secret)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=k))

    seen_contexts = []

    call_args_list = target_worker.execute_model.call_args_list
    assert len(call_args_list) == 1
    for _, kwargs in call_args_list:
        seq_group_metadata_list = kwargs[
            "execute_model_req"].seq_group_metadata_list

        assert len(seq_group_metadata_list) == (k + 1) * batch_size
        for seq_group_metadata in seq_group_metadata_list:
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
    """Verify SpecDecodeWorker calls the rejection sampler with
    correct inputs. Everything else is mocked out.
    """
    vocab_size = 32_000

    draft_worker = mock_worker(cls=MultiStepWorker,
                               vocab_size=vocab_size,
                               use_spec=False)
    target_worker = mock_worker(vocab_size=vocab_size, use_spec=False)
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)
    worker.init_device()

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

    proposal_lens = torch.ones(batch_size, dtype=torch.int64,
                               device='cuda') * k

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)

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
    target_token_logprobs = torch.rand(1,
                                       batch_size * (k + 1),
                                       vocab_size,
                                       dtype=torch.float32,
                                       device='cuda')
    target_output = create_sampler_output_list(target_token_ids,
                                               target_token_probs,
                                               target_token_logprobs)

    target_worker.execute_model.return_value = [target_output[0]]

    exception_secret = 'artificial stop'
    rejection_sampler.side_effect = ValueError(exception_secret)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=k))

    assert len(rejection_sampler.call_args_list) == 1
    _, kwargs = rejection_sampler.call_args_list[0]
    actual = SimpleNamespace(**kwargs)

    assert torch.equal(actual.bonus_token_ids,
                       target_token_ids.reshape(batch_size, k + 1)[:, -1:])
    assert torch.equal(
        actual.target_probs,
        target_token_probs.reshape(batch_size, k + 1, -1)[:, :-1])
    assert torch.equal(actual.draft_token_ids, proposal_token_ids)
    assert torch.equal(actual.draft_probs, proposal_probs)


@pytest.mark.parametrize('k', [1, 2, 6])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_correctly_formats_output(k: int, batch_size: int):
    """Verify SpecDecodeWorker formats sampler output correctly.
    Everything else is mocked out.
    """
    vocab_size = 32_000

    draft_worker = mock_worker(cls=MultiStepWorker,
                               vocab_size=vocab_size,
                               use_spec=False)
    target_worker = mock_worker(vocab_size=vocab_size, use_spec=False)
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)
    worker.init_device()

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

    proposal_lens = torch.ones(batch_size, dtype=torch.int64,
                               device='cuda') * k

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)

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
    target_token_logprobs = torch.rand(1,
                                       batch_size * (k + 1),
                                       vocab_size,
                                       dtype=torch.float32,
                                       device='cuda')
    target_output = create_sampler_output_list(target_token_ids,
                                               target_token_probs,
                                               target_token_logprobs)

    target_worker.execute_model.return_value = [target_output[0]]

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

    output = worker.execute_model(execute_model_req=ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k))

    expected_output = create_sampler_output_list(
        token_ids=rejection_sampler_output.transpose(0, 1),
        probs=[None for _ in range(k + 1)],
        logprobs=[None for _ in range(k + 1)])

    seq_ids = [
        next(iter(seq_group_metadata.seq_data.keys()))
        for seq_group_metadata in seq_group_metadata_list
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


@pytest.mark.parametrize('k', [1, 2])
@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('returns_metrics', [True, False])
@torch.inference_mode()
def test_collects_metrics(k: int, batch_size: int, returns_metrics: bool):
    """Verify SpecDecodeWorker collects metrics.
    """
    vocab_size = 32_000

    draft_worker = mock_worker(cls=MultiStepWorker,
                               vocab_size=vocab_size,
                               use_spec=False)
    target_worker = mock_worker(vocab_size=vocab_size, use_spec=False)
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)
    worker.init_device()

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

    proposal_lens = torch.ones(batch_size, dtype=torch.int64,
                               device='cuda') * k

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)

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
    target_token_logprobs = torch.rand(1,
                                       batch_size * (k + 1),
                                       vocab_size,
                                       dtype=torch.float32,
                                       device='cuda')
    target_output = create_sampler_output_list(target_token_ids,
                                               target_token_probs,
                                               target_token_logprobs)

    target_worker.execute_model.return_value = [target_output[0]]

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

    mock_rejsample_metrics = MagicMock(
        spec=SpecDecodeWorkerMetrics) if returns_metrics else None
    metrics_collector.maybe_collect_rejsample_metrics.return_value = (
        mock_rejsample_metrics)

    output = worker.execute_model(execute_model_req=ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k))
    assert output[0].spec_decode_worker_metrics == mock_rejsample_metrics

    call_args_list = (
        metrics_collector.maybe_collect_rejsample_metrics.call_args_list)
    assert len(call_args_list) == 1
    args, kwargs = call_args_list[0]
    assert args[0] == k or kwargs.get('k', -1) == k


@pytest.mark.parametrize('k', [0])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
@torch.inference_mode()
def test_k_equals_zero(k: int, batch_size: int):
    """Verify that the SpecDecodeWorker calls the draft and target workers
    when k is zero. This happens during prefill.
    """
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    target_worker.execute_model.return_value = [MagicMock(spec=SamplerOutput)]

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    seq_group_metadata_list, _, _ = create_batch(batch_size,
                                                 k,
                                                 prev_output_token_len=0)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list, num_lookahead_slots=k)

    out = worker.execute_model(execute_model_req=execute_model_req)

    assert len(out) == 1, f"expected only one token output when {k=}"
    assert out[0].probs is None, "expect gpu tensor references to be None"
    assert out[
        0].sampled_tokens is None, "expect gpu tensor references to be None"

    draft_worker.execute_model.assert_called_once_with(execute_model_req)
    target_worker.execute_model.assert_called_once_with(execute_model_req)


@pytest.mark.parametrize('k', [0, 5])
@pytest.mark.parametrize('batch_size', [0])
@torch.inference_mode()
def test_empty_input_batch(k: int, batch_size: int):
    """Verify that the SpecDecodeWorker calls the draft and target workers
    when the input batch is empty. This can happen if the engine communicates
    to the workers information without scheduling a batch.
    """
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    target_worker.execute_model.return_value = [MagicMock(spec=SamplerOutput)]

    draft_worker.device = 'cuda'
    target_worker.device = 'cuda'

    set_random_seed(1)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    seq_group_metadata_list, _, _ = create_batch(batch_size,
                                                 k,
                                                 prev_output_token_len=0)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list, num_lookahead_slots=k)

    out = worker.execute_model(execute_model_req=execute_model_req)

    assert len(out) == 1, f"expected only one token output when {k=}"
    assert out[0].probs is None, "expect gpu tensor references to be None"
    assert out[
        0].sampled_tokens is None, "expect gpu tensor references to be None"

    draft_worker.execute_model.assert_called_once_with(execute_model_req)
    target_worker.execute_model.assert_called_once_with(execute_model_req)


@pytest.mark.skip_global_cleanup
def test_init_device():
    """Verify SpecDecodeWorker invokes proposer/scorer worker init_device, as
    well as other GPU initialization.
    """
    draft_worker = mock_worker(cls=MultiStepWorker, use_spec=False)
    target_worker = mock_worker(use_spec=False)
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    worker.init_device()

    draft_worker.init_device.assert_called_once()

    target_worker.init_device.assert_called_once()

    metrics_collector.init_gpu_tensors.assert_called_once()
    rejection_sampler.init_gpu_tensors.assert_called_once()


@torch.inference_mode()
def test_initialize_cache():
    """Verify SpecDecodeWorker invokes initialize_cache on proposer/scorer
    workers.
    """
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    kwargs = {"num_gpu_blocks": 1024, "num_cpu_blocks": 1023}
    worker.initialize_cache(**kwargs)

    draft_worker.initialize_cache.assert_called_once_with(**kwargs)
    target_worker.initialize_cache.assert_called_once_with(**kwargs)


@pytest.mark.parametrize('available_gpu_blocks', [1, 1024])
@pytest.mark.parametrize('available_cpu_blocks', [500])
@pytest.mark.parametrize('target_cache_block_size_bytes', [2 * 2 * 4096])
@pytest.mark.parametrize('draft_kv_size_bytes', [0, 2 * 2 * 768, 2 * 2 * 4096])
@pytest.mark.skip_global_cleanup
def test_determine_num_available_blocks(available_gpu_blocks: int,
                                        available_cpu_blocks: int,
                                        target_cache_block_size_bytes: int,
                                        draft_kv_size_bytes: int):
    """Verify SpecDecodeWorker correctly profiles num available GPU blocks.
    Specifically, it should run profiling in the scorer worker, and then evenly
    split the blocks between proposer and scorer worker.
    """
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    rejection_sampler.token_id_dtype = torch.int64
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)

    target_worker.determine_num_available_blocks.return_value = (
        available_gpu_blocks, available_cpu_blocks)
    target_worker.get_cache_block_size_bytes.return_value = (
        target_cache_block_size_bytes)
    draft_worker.get_cache_block_size_bytes.return_value = draft_kv_size_bytes

    worker = SpecDecodeWorker(draft_worker, target_worker, rejection_sampler,
                              metrics_collector)

    num_gpu_blocks, num_cpu_blocks = worker.determine_num_available_blocks()

    target_worker.determine_num_available_blocks.assert_called_once()
    assert num_cpu_blocks == available_cpu_blocks

    assert num_gpu_blocks == split_num_cache_blocks_evenly(
        target_cache_block_size_bytes, draft_kv_size_bytes,
        available_gpu_blocks)


@pytest.mark.parametrize('available_gpu_blocks',
                         list(range(20)) + [1024, 1024**2])
@pytest.mark.parametrize('target_cache_block_size_bytes',
                         [2 * 2 * 4096, 2 * 2 * 8192])
@pytest.mark.parametrize('draft_kv_size_bytes', [0, 2 * 2 * 768, 2 * 2 * 4096])
@pytest.mark.skip_global_cleanup
def test_split_num_cache_blocks_evenly(available_gpu_blocks: int,
                                       target_cache_block_size_bytes: int,
                                       draft_kv_size_bytes: int):
    """Verify split_num_cache_blocks_evenly does not exceed original memory
    allocation in bytes.
    """
    num_blocks = split_num_cache_blocks_evenly(target_cache_block_size_bytes,
                                               draft_kv_size_bytes,
                                               available_gpu_blocks)
    assert (num_blocks * target_cache_block_size_bytes) + (
        num_blocks * draft_kv_size_bytes) <= (available_gpu_blocks *
                                              target_cache_block_size_bytes)
