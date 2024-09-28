import pytest
import torch

from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.worker.worker import Worker

from .utils import create_batch, create_worker


def create_proposal(batch_size: int, propose_len: int, vocab_size: int,
                    device: str) -> SpeculativeProposals:
    proposal_probs = torch.rand((batch_size, propose_len, vocab_size),
                                device=device)
    proposal_token_ids = torch.argmax(proposal_probs, dim=-1)
    proposal_lens = torch.tensor([propose_len] * batch_size, device=device)
    return SpeculativeProposals(proposal_token_ids, proposal_probs,
                                proposal_lens)


def assert_score_equal(score1: SpeculativeScores,
                       score2: SpeculativeScores) -> None:
    assert torch.allclose(score1.probs, score2.probs)
    assert torch.allclose(score1.logprobs, score2.logprobs)
    assert torch.equal(score1.token_ids, score2.token_ids)


@pytest.mark.parametrize('model_name', ['facebook/opt-125m'])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('propose_len', [1, 3, 5])
@pytest.mark.parametrize('device', ['cuda'])
def test_scoroer(model_name: str, batch_size: int, propose_len: int,
                 device: str) -> None:
    """
    Compare the batch expansion scorer and mqa scorer return the same score
    """
    seed = 0
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    scorer_worker = create_worker(Worker, model_name, block_size,
                                  num_gpu_blocks, seed)
    scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor = True
    scorer_worker.model_runner.model.sampler.\
        should_modify_greedy_probs_inplace = True

    vocab_size = scorer_worker.vocab_size
    proposals = create_proposal(batch_size, propose_len, vocab_size, device)
    seq_group_metadatalist, _, _ = create_batch(batch_size,
                                                propose_len,
                                                block_size=block_size,
                                                num_gpu_blocks=num_gpu_blocks)
    requests = ExecuteModelRequest(seq_group_metadatalist,
                                   num_lookahead_slots=propose_len)

    batch_expansion_scorer = BatchExpansionTop1Scorer(scorer_worker, device,
                                                      vocab_size)
    batch_expansion_score = batch_expansion_scorer.score_proposals(
        requests, proposals)

    mqa_scorer = MQAScorer(scorer_worker, device, vocab_size)
    mqa_score = mqa_scorer.score_proposals(requests, proposals)

    assert_score_equal(batch_expansion_score, mqa_score)
