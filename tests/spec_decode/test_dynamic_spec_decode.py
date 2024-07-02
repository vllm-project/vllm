from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker
from vllm.spec_decode.top1_proposer import Top1Proposer

from .test_utils import mock_spec_decode_sampler
from .utils import create_batch, mock_worker


@pytest.mark.parametrize('queue_size', [4])
@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('k', [1])
@pytest.mark.parametrize("acceptance_sampler_method",
                         ["rejection_sampler", "typical_acceptance_sampler"])
@torch.inference_mode()
def test_disable_spec_tokens(queue_size: int, batch_size: int, k: int,
                             acceptance_sampler_method: str):
    """Verify that speculative tokens are disabled when the batch size
    exceeds the threshold.
    """
    disable_by_batch_size = 3
    draft_worker = mock_worker(cls=MultiStepWorker)
    target_worker = mock_worker()
    metrics_collector = MagicMock(spec=AsyncMetricsCollector)
    worker = SpecDecodeWorker(proposer_worker=draft_worker,
                              scorer_worker=target_worker,
                              spec_decode_sampler=mock_spec_decode_sampler(
                                  acceptance_sampler_method),
                              metrics_collector=metrics_collector,
                              disable_by_batch_size=disable_by_batch_size)

    exception_secret = 'artificial stop'
    draft_worker.get_spec_proposals.side_effect = ValueError(exception_secret)

    seq_group_metadata_list, _, _ = create_batch(batch_size, k)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k,
        running_queue_size=queue_size)

    if queue_size > disable_by_batch_size:
        with patch.object(worker,
                          '_run_no_spec',
                          side_effect=ValueError(exception_secret)), \
            pytest.raises(ValueError, match=exception_secret):
            worker.execute_model(execute_model_req=execute_model_req)

    # When the batch size is larger than the threshold,
    # we expect no speculative tokens (0).
    expected_num_spec_tokens = None if queue_size < disable_by_batch_size else 0
    assert seq_group_metadata_list[
        0].num_speculative_tokens == expected_num_spec_tokens

    draft_worker.sampler_output.side_effect = ValueError(exception_secret)

    proposer = Top1Proposer(
        worker=draft_worker,
        device='cpu',  # not used
        vocab_size=100,  # not used
        # Must be long enough to avoid being skipped due to length.
        max_proposal_len=1024,
    )

    if queue_size < disable_by_batch_size:
        # Should raise exception when executing the mocked draft model.
        with pytest.raises(ValueError, match=exception_secret):
            proposer.get_spec_proposals(execute_model_req=ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                num_lookahead_slots=k), )
    else:
        # Should not execute the draft model because spec decode is disabled
        # for all requests. Accordingly, the proposal length should be 0.
        proposals = proposer.get_spec_proposals(
            execute_model_req=ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                num_lookahead_slots=k), )
        assert proposals.proposal_lens.tolist() == [0] * batch_size
