from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker
from vllm.spec_decode.top1_proposer import Top1Proposer

from .utils import create_batch, mock_worker


@pytest.mark.parametrize('queue_size', [2, 4])
@torch.inference_mode()
def test_disable_spec_tokens(queue_size: int):
    """Verify that speculative tokens are disabled when the queue size
    exceeds the threshold.
    """
    batch_size = 1
    k = 5
    disable_at_queue_size = 3

    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    rejection_sampler = MagicMock(spec=RejectionSampler)
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    worker = SpecDecodeWorker(proposer_worker=draft_worker,
                              scorer_worker=target_worker,
                              rejection_sampler=rejection_sampler,
                              metrics_collector=metrics_collector,
                              disable_at_queue_size=disable_at_queue_size)

    exception_secret = 'artificial stop'
    draft_worker.get_spec_proposals.side_effect = ValueError(exception_secret)

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k,
        running_queue_size=queue_size)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_req=execute_model_req)

    # When the queue size is larger than the threshold,
    # we expect no speculative tokens (0).
    expected_num_spec_tokens = None if queue_size < disable_at_queue_size else 0
    assert seq_group_metadata_list[
        0]._num_speculative_tokens == expected_num_spec_tokens

    proposer = Top1Proposer(
        worker=draft_worker,
        device='cpu',  # not used
        vocab_size=100,  # not used
        max_proposal_len=2048,
    )

    # The request with spec token 0 should have proposal_len=0.
    proposal_lens, _, _ = proposer._split_by_proposal_len(
        seq_group_metadata_list, k)
    assert proposal_lens[0] == (k if queue_size < disable_at_queue_size else 0)
