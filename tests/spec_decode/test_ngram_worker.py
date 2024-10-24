import torch

from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.top1_proposer import Top1Proposer

from .utils import create_seq_group_metadata_from_prompts, create_worker


def test_ngram_algo_correctness_for_single_no_match():
    """Verify our ngram algo find the right candidate in the prompt

    For the scenario cannot find any candidate in one single batch
    """
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    seed = 100
    model_name = 'JackFram/llama-68m'
    vocab_size = 32_000
    device = 'cuda:0'

    ngram_worker = create_worker(
        NGramWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    proposer = Top1Proposer(
        worker=ngram_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=20,
    )

    # set ngram window [1, 3], which is window=1/2/3
    ngram_worker.set_ngram_window_size(1, 3)

    prompts = [
        # shall find no candidate
        [1, 2, 3, 4, 5, 6, 7],
    ]

    proposal_len = 5
    final_prompt_lens = [len(prompt) + proposal_len for prompt in prompts]
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=proposal_len),
        seq_ids_with_bonus_token_in_last_step=None)

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([1, proposal_len])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([1, proposal_len])
    assert proposals.proposal_lens.shape == torch.Size([1])
    assert proposals.proposal_lens.tolist() == [0]


def test_ngram_algo_correctness_for_batches_not_match_all():
    """Verify our ngram algo find the right candidate in the prompt

    For the scenario find some candidate not full in batchs
    """
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    seed = 100
    model_name = 'JackFram/llama-68m'
    vocab_size = 32_000
    device = 'cuda:0'

    ngram_worker = create_worker(
        NGramWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    proposer = Top1Proposer(
        worker=ngram_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=20,
    )

    # set ngram window [1, 3], which is window=1/2/3
    ngram_worker.set_ngram_window_size(1, 3)

    prompts = [
        # shall find no candidate
        [1, 2, 3, 4, 5, 6, 7],
        # shall find candidate 12,13,14,15,16
        [11, 12, 13, 14, 15, 16, 11],
        # shall find candidate 23,24,25,26,21
        [21, 21, 22, 23, 24, 25, 26, 21, 22],
        # shall find candidate 34,35,36,37,38
        [31, 32, 31, 32, 33, 34, 35, 36, 37, 38, 31, 32, 33],
        # shall find no candidate as exceed max_proposal_len
        [
            31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 33, 34, 35, 36, 37,
            38, 31, 32, 33
        ],
    ]

    proposal_len = 5
    final_prompt_lens = [len(prompt) + proposal_len for prompt in prompts]
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=proposal_len),
        seq_ids_with_bonus_token_in_last_step=None)

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([5, proposal_len])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([5, proposal_len])
    assert proposals.proposal_lens.shape == torch.Size([5])

    # the first sequence has no match so proposal_len should be overwritten to 0
    assert proposals.proposal_lens.tolist(
    ) == [0] + [proposal_len for _ in range(3)] + [0]

    for i in range(proposal_len):
        assert proposals.proposal_token_ids[0][i] == -1
        assert proposals.proposal_token_ids[1][i] == prompts[1][i + 1]
        assert proposals.proposal_token_ids[2][i] == prompts[2][i + 3]
        assert proposals.proposal_token_ids[3][i] == prompts[3][i + 5]
        assert proposals.proposal_token_ids[4][i] == -1


def test_ngram_algo_correctness_for_batches_match_all():
    """Verify our ngram algo find the right candidate in the prompt

    For the scenario find candidate in all batchs
    """

    block_size = 32
    num_gpu_blocks = 2048 // block_size
    seed = 100
    model_name = 'JackFram/llama-68m'
    vocab_size = 32_000
    device = 'cuda:0'

    ngram_worker = create_worker(
        NGramWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    proposer = Top1Proposer(
        worker=ngram_worker,
        device=device,
        vocab_size=vocab_size,
        max_proposal_len=20,
    )

    # set ngram window [0, 3], which is window=1/2/3
    ngram_worker.set_ngram_window_size(1, 3)

    prompts = [
        # shall find candidate 12,13,14,15,16
        [11, 12, 13, 14, 15, 16, 11],
        # shall find candidate 23,24,25,26,21
        [21, 21, 22, 23, 24, 25, 26, 21, 22],
        # shall find candidate 34,35,36,37,38
        [31, 32, 31, 32, 33, 34, 35, 36, 37, 38, 31, 32, 33],
    ]

    proposal_len = 5
    final_prompt_lens = [len(prompt) + proposal_len for prompt in prompts]
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    proposals = proposer.get_spec_proposals(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=proposal_len),
        seq_ids_with_bonus_token_in_last_step=None)

    assert torch.is_tensor(proposals.proposal_token_ids)
    assert torch.is_tensor(proposals.proposal_probs)

    assert proposals.proposal_token_ids.shape == torch.Size([3, proposal_len])
    assert proposals.proposal_probs.shape[:-1] == torch.Size([3, proposal_len])
    assert proposals.proposal_lens.shape == torch.Size([3])

    assert proposals.proposal_lens.tolist() == [proposal_len for _ in range(3)]

    for i in range(proposal_len):
        assert proposals.proposal_token_ids[0][i] == prompts[0][i + 1]
        assert proposals.proposal_token_ids[1][i] == prompts[1][i + 3]
        assert proposals.proposal_token_ids[2][i] == prompts[2][i + 5]
