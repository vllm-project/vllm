# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that Triton spec-decode kernels compile and produce correct results
on CPU. These kernels require explicit int64 typing to satisfy Triton-CPU's
stricter type-checking (vs CUDA Triton which implicitly widens int32->int64).

Requires triton with a CPU backend. Skipped otherwise.
"""

import pytest
import torch

from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not HAS_TRITON, reason="Triton (with CPU backend) not available"
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestExpandKernel:
    def test_basic(self, device):
        from vllm.v1.sample.rejection_sampler import expand_kernel

        x = torch.tensor([5, 0, 8], dtype=torch.int64, device=device)
        cu_num_tokens = torch.tensor([3, 7, 10], dtype=torch.int64, device=device)
        out = torch.empty(10, dtype=torch.int64, device=device)

        expand_kernel[(3,)](out, x, cu_num_tokens, 0, -1, MAX_NUM_TOKENS=16)

        expected = torch.tensor([5, 5, 5, -1, -1, -1, -1, 8, 8, 8], dtype=torch.int64)
        assert torch.equal(out, expected)

    def test_no_replace(self, device):
        from vllm.v1.sample.rejection_sampler import expand_kernel

        x = torch.tensor([10, 20], dtype=torch.int64, device=device)
        cu = torch.tensor([2, 5], dtype=torch.int64, device=device)
        out = torch.empty(5, dtype=torch.int64, device=device)

        expand_kernel[(2,)](out, x, cu, 0, 0, MAX_NUM_TOKENS=16)

        expected = torch.tensor([10, 10, 20, 20, 20], dtype=torch.int64)
        assert torch.equal(out, expected)


class TestRejectionGreedySampleKernel:
    def test_all_accepted(self, device):
        from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel

        output = torch.full((2, 3), -1, dtype=torch.int64, device=device)
        cu_draft = torch.tensor([2, 4], dtype=torch.int64, device=device)
        draft_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=device)
        target_argmax = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=device)
        bonus = torch.tensor([99, 50], dtype=torch.int64, device=device)
        is_greedy = torch.tensor([True, True], dtype=torch.bool, device=device)

        rejection_greedy_sample_kernel[(2,)](
            output,
            cu_draft,
            draft_ids,
            target_argmax,
            bonus,
            is_greedy,
            2,
            None,
            None,
            False,
        )

        assert output[0].tolist() == [10, 20, 99]
        assert output[1].tolist() == [30, 40, 50]

    def test_rejection(self, device):
        from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel

        output = torch.full((1, 3), -1, dtype=torch.int64, device=device)
        cu_draft = torch.tensor([2], dtype=torch.int64, device=device)
        draft_ids = torch.tensor([10, 20], dtype=torch.int64, device=device)
        target_argmax = torch.tensor([10, 99], dtype=torch.int64, device=device)
        bonus = torch.tensor([50], dtype=torch.int64, device=device)
        is_greedy = torch.tensor([True], dtype=torch.bool, device=device)

        rejection_greedy_sample_kernel[(1,)](
            output,
            cu_draft,
            draft_ids,
            target_argmax,
            bonus,
            is_greedy,
            2,
            None,
            None,
            False,
        )

        # First token accepted (matches), second rejected (draft=20, target=99)
        assert output[0, 0].item() == 10
        assert output[0, 1].item() == 99
        # Bonus not written (rejection happened)
        assert output[0, 2].item() == -1


class TestRejectionRandomSampleKernel:
    def test_accepted(self, device):
        from vllm.v1.sample.rejection_sampler import rejection_random_sample_kernel

        output = torch.full((1, 2), -1, dtype=torch.int64, device=device)
        cu_draft = torch.tensor([1], dtype=torch.int64, device=device)
        draft_ids = torch.tensor([5], dtype=torch.int64, device=device)
        vocab_size = 8
        draft_probs = torch.zeros(1, vocab_size, device=device)
        draft_probs[0, 5] = 0.8
        target_probs = torch.zeros(1, vocab_size, device=device)
        target_probs[0, 5] = 0.9
        bonus = torch.tensor([7], dtype=torch.int64, device=device)
        recovered = torch.tensor([3], dtype=torch.int64, device=device)
        uniform = torch.tensor([0.5], dtype=torch.float64, device=device)
        is_greedy = torch.tensor([False], dtype=torch.bool, device=device)

        rejection_random_sample_kernel[(1,)](
            output,
            cu_draft,
            draft_ids,
            draft_probs,
            target_probs,
            bonus,
            recovered,
            uniform,
            is_greedy,
            1,
            vocab_size,
            None,
            False,
            False,
        )

        # target/draft = 0.9/0.8 = 1.125 > uniform=0.5, so accepted
        assert output[0].tolist() == [5, 7]


class TestSampleRecoveredTokensKernel:
    def test_basic(self, device):
        from vllm.v1.sample.rejection_sampler import sample_recovered_tokens_kernel

        output_ids = torch.full((2,), -1, dtype=torch.int64, device=device)
        cu_draft = torch.tensor([1, 2], dtype=torch.int64, device=device)
        draft_ids = torch.tensor([0, 1], dtype=torch.int64, device=device)
        vocab_size = 4
        target_probs = torch.tensor(
            [[0.1, 0.2, 0.5, 0.2], [0.1, 0.1, 0.1, 0.7]],
            dtype=torch.float32,
            device=device,
        )
        draft_probs = torch.tensor(
            [[0.4, 0.1, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]],
            dtype=torch.float32,
            device=device,
        )
        inv_q = torch.ones(2, vocab_size, dtype=torch.float32, device=device)

        sample_recovered_tokens_kernel[(2, 1)](
            output_ids,
            cu_draft,
            draft_ids,
            draft_probs,
            target_probs,
            inv_q,
            vocab_size,
            BLOCK_SIZE=16,
            NO_DRAFT_PROBS=False,
        )

        # max(target - draft): pos0=[-.3,.1,.4,-.2]->idx2, pos1=[0,-.4,-.1,.5]->idx3
        assert output_ids[0].item() == 2
        assert output_ids[1].item() == 3


class TestEaglePrepareInputsPaddedKernel:
    def test_basic(self, device):
        from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel

        cu_num_draft = torch.tensor([2, 5], dtype=torch.int64, device=device)
        valid_sampled = torch.tensor([2, 3], dtype=torch.int64, device=device)
        query_start_loc = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
        token_indices = torch.zeros(2, dtype=torch.int64, device=device)
        num_rejected = torch.zeros(2, dtype=torch.int64, device=device)

        eagle_prepare_inputs_padded_kernel[(2,)](
            cu_num_draft,
            valid_sampled,
            query_start_loc,
            token_indices,
            num_rejected,
            2,
        )

        # Req 0: draft=2, valid=2, rejected=2+1-2=1, q_last=5-1=4, idx=4-1=3
        # Req 1: draft=5-2=3, valid=3, rejected=3+1-3=1, q_last=12-1=11, idx=11-1=10
        assert token_indices[0].item() == 3
        assert token_indices[1].item() == 10
        assert num_rejected[0].item() == 1
        assert num_rejected[1].item() == 1
