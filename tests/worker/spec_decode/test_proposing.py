import torch
import random
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.spec_decode.proposing import DraftModelTop1Proposer
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplerOutput

from .utils import (create_execute_model_data, create_worker,
                    create_seq_group_metadata_from_prompts, zero_kv_cache,
                    patch_execute_model_with_seeds,
                    assert_logprobs_dict_allclose, create_batch)


@torch.inference_mode()
def test_draft_proposals_full_speculation_len():
    """
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device='cuda:0'

    draft_worker = MagicMock()
    proposer = DraftModelTop1Proposer(
        draft_worker=draft_worker,
        device=device,
        max_model_len=2048,
        vocab_size=vocab_size,
    )
    draft_worker.execute_model_multi_step.return_value = [SamplerOutput(
        outputs=[],
        sampled_token_probs=torch.rand(batch_size, vocab_size, device=device, dtype=torch.float32),
        sampled_token_ids=torch.randint(low=0, high=vocab_size, size=(batch_size,), device=device, dtype=torch.long),
    ) for _ in range(k)]

    execute_model_data, _, _ = create_batch(batch_size, k)

    proposals = proposer.get_proposals(
        **execute_model_data.to_dict(),
        max_proposal_len=k,
    )
    
    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([batch_size, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([batch_size, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [k for _ in range(batch_size)]

@torch.inference_mode()
def test_draft_proposals_no_speculations():
    """
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device='cuda:0'
    prompt_len = 10

    draft_worker = MagicMock()
    proposer = DraftModelTop1Proposer(
        draft_worker=draft_worker,
        device=device,
        max_model_len=prompt_len + k - 1,
        vocab_size=vocab_size,
    )

    execute_model_data, _, _ = create_batch(batch_size, k, prompt_len=prompt_len)

    proposals = proposer.get_proposals(
        **execute_model_data.to_dict(),
        max_proposal_len=k,
    )

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([0, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([0, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [0 for _ in range(batch_size)]


@torch.inference_mode()
def test_draft_proposals_mixed_k():
    """
    """
    k = 10
    batch_size = 32
    vocab_size = 32_000
    device='cuda:0'

    small_prompt_len = 5
    long_prompt_len = 10
    prev_output_token_len = 20

    expected_num_proposal_seqs = 6
    expected_num_no_proposal_seqs = batch_size - expected_num_proposal_seqs

    prompt_len = [small_prompt_len for _ in range(expected_num_proposal_seqs - 1)] + [long_prompt_len for _ in range(expected_num_no_proposal_seqs)] + [small_prompt_len]

    draft_worker = MagicMock()
    proposer = DraftModelTop1Proposer(
        draft_worker=draft_worker,
        device=device,
        max_model_len=long_prompt_len + prev_output_token_len + k - 1,
        vocab_size=vocab_size,
    )

    draft_worker.execute_model_multi_step.return_value = [SamplerOutput(
        outputs=[],
        sampled_token_probs=torch.rand(expected_num_proposal_seqs, vocab_size, device=device, dtype=torch.float32),
        sampled_token_ids=torch.randint(low=0, high=vocab_size, size=(expected_num_proposal_seqs,), device=device, dtype=torch.long),
    ) for _ in range(k)]

    execute_model_data, _, _ = create_batch(batch_size, k, prompt_len=prompt_len,
        prev_output_token_len=prev_output_token_len,)

    proposals = proposer.get_proposals(
        **execute_model_data.to_dict(),
        max_proposal_len=k,
    )

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([batch_size, k])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([batch_size, k])

    assert proposals.proposal_lens.shape == torch.Size([batch_size])
    assert proposals.proposal_lens.tolist() == [k for _ in range(expected_num_proposal_seqs - 1)] + [0 for _ in range(expected_num_no_proposal_seqs)] + [k]
