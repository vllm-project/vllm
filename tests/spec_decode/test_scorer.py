# SPDX-License-Identifier: Apache-2.0

import random
from typing import List

import pytest
import torch

from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.worker.worker import Worker

from .utils import create_batch, create_worker


def create_proposal(propose_lens: List[int], vocab_size: int,
                    device: str) -> SpeculativeProposals:
    batch_size = len(propose_lens)
    max_propose_len = max(propose_lens)
    proposal_probs = torch.rand((batch_size, max_propose_len, vocab_size),
                                device=device)

    proposal_token_ids = torch.full((batch_size, max_propose_len),
                                    fill_value=-1,
                                    device=device)
    for i in range(batch_size):
        proposal_token_ids[i][:propose_lens[i]] = torch.argmax(
            proposal_probs[i][:propose_lens[i]], dim=-1)

    propose_lens = torch.tensor(propose_lens, device=device)
    return SpeculativeProposals(proposal_token_ids, proposal_probs,
                                propose_lens)


def assert_score_equal(score1: SpeculativeScores,
                       score2: SpeculativeScores) -> None:
    assert torch.allclose(score1.probs, score2.probs)
    assert torch.allclose(score1.logprobs, score2.logprobs)
    assert torch.equal(
        score1.token_ids,
        score2.token_ids), f"{score1.token_ids}, {score2.token_ids}"


@pytest.mark.parametrize('model_name', ['facebook/opt-125m'])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('max_propose_len', [1, 3, 5])
@pytest.mark.parametrize('mixed_propose_len', [True])
@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('prefill_chunking', [False, True])
def test_scorer(model_name: str, batch_size: int, max_propose_len: int,
                mixed_propose_len: bool, device: str,
                prefill_chunking: bool) -> None:
    """
    Compare the batch expansion scorer and mqa scorer return the same score.
    We test for both queries with the same propose length and different 
    propose length, as well as mixed prefill-decode batches.
    """
    seed = 0
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    scorer_worker = create_worker(Worker, model_name, block_size,
                                  num_gpu_blocks, seed)
    scorer_worker.model_runner.disable_logprobs = True  # accessed by mqa_scorer
    scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor = True
    scorer_worker.model_runner.model.sampler.\
        should_modify_greedy_probs_inplace = True

    vocab_size = scorer_worker.vocab_size

    if not mixed_propose_len:
        propose_lens = [max_propose_len] * batch_size
    else:
        # There must be at least 1 decode request, otherwise
        # we have nothing to score (`_run_no_spec`).
        non_zero_cnt = random.randint(1, batch_size)
        propose_lens = [max_propose_len
                        ] * non_zero_cnt + [0] * (batch_size - non_zero_cnt)
        random.shuffle(propose_lens)

    seq_group_metadatalist, _, _ = create_batch(batch_size,
                                                max_propose_len,
                                                block_size=block_size,
                                                num_gpu_blocks=num_gpu_blocks)

    if mixed_propose_len and prefill_chunking and (n_prefills :=
                                                   batch_size - non_zero_cnt):
        prefill, _, _ = create_batch(n_prefills,
                                     None,
                                     prefill_chunk_size=4,
                                     block_size=block_size,
                                     num_gpu_blocks=num_gpu_blocks,
                                     seq_ids=list(
                                         range(batch_size,
                                               batch_size + n_prefills)))
        # re-order to guarantee prefill|decode order
        target_group_metadatalist = [
            seq_group_metadatalist[i] for i, p in enumerate(propose_lens)
            if p > 0
        ]
        seq_group_metadatalist = prefill + target_group_metadatalist
        propose_lens = [0] * n_prefills + [p for p in propose_lens if p > 0]

    proposals = create_proposal(propose_lens, vocab_size, device)
    requests = ExecuteModelRequest(seq_group_metadatalist,
                                   num_lookahead_slots=max_propose_len)

    batch_expansion_scorer = BatchExpansionTop1Scorer(scorer_worker, device,
                                                      vocab_size)
    batch_expansion_score = batch_expansion_scorer.score_proposals(
        requests, proposals)

    mqa_scorer = MQAScorer(scorer_worker, device, vocab_size)
    mqa_score = mqa_scorer.score_proposals(requests, proposals)

    assert_score_equal(batch_expansion_score, mqa_score)
