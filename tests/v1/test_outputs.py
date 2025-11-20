# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest import TestCase

from vllm.v1.outputs import LogprobsLists


class TestLogprobsLists(TestCase):
    def setUp(self):
        self.logprobsLists = LogprobsLists(
            logprob_token_ids=[
                [1, 2],  # Request 0 token 0
                [3, 4],  # Request 0 token 1
                [5, 6],  # Request 1 token 0
                [7, 8],  # Request 1 token 1
                [9, 10],  # Request 1 token 2
                [11, 12],  # Request 2 token 0
                [13, 14],  # Request 2 token 1
                [15, 16],  # Request 2 token 2
                [17, 18],  # Request 2 token 3
            ],
            logprobs=[
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
                [1.3, 1.4],
                [1.5, 1.6],
                [1.7, 1.8],
            ],
            sampled_token_ranks=[1, 3, 5, 7, 9, 11, 13, 15, 17],
            cu_num_generated_tokens=[0, 2, 5, 9],
        )

    def test_slice_without_cu_num_generated_tokens(self):
        """Test slicing without cu_num_generated_tokens"""
        logprobsLists = LogprobsLists(
            logprob_token_ids=[[1], [2], [3]],
            logprobs=[[0.1], [0.2], [0.3]],
            sampled_token_ranks=[1, 2, 3],
            cu_num_generated_tokens=None,
        )

        sliced = logprobsLists.slice(1, 3)
        assert sliced.logprob_token_ids == [[2], [3]]
        assert sliced.logprobs == [[0.2], [0.3]]
        assert sliced.sampled_token_ranks == [2, 3]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_start(self):
        """Test slicing from the start position"""
        sliced = self.logprobsLists.slice(0, 2)
        assert len(sliced.logprob_token_ids) == 5
        assert sliced.logprob_token_ids == [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        assert sliced.cu_num_generated_tokens == [0, 2, 5]

    def test_slice_from_middle(self):
        """Test slicing from the middle position"""
        sliced = self.logprobsLists.slice(1, 3)
        assert len(sliced.logprob_token_ids) == 7
        assert sliced.logprob_token_ids == [
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
        ]
        assert sliced.cu_num_generated_tokens == [0, 3, 7]

    def test_slice_single_request(self):
        """Test slicing a single request"""
        sliced = self.logprobsLists.slice(1, 2)
        assert len(sliced.logprob_token_ids) == 3
        assert sliced.logprob_token_ids == [[5, 6], [7, 8], [9, 10]]
        assert sliced.cu_num_generated_tokens == [0, 3]

    def test_slice_last_request(self):
        """Test slicing the last request"""
        sliced = self.logprobsLists.slice(2, 3)
        assert len(sliced.logprob_token_ids) == 4
        assert sliced.logprob_token_ids == [[11, 12], [13, 14], [15, 16], [17, 18]]
        assert sliced.cu_num_generated_tokens == [0, 4]

    def test_slice_all_requests(self):
        """Test slicing all requests (full slice)"""
        sliced = self.logprobsLists.slice(0, 3)
        assert len(sliced.logprob_token_ids) == 9  # All tokens
        assert sliced.logprob_token_ids == self.logprobsLists.logprob_token_ids
        assert (
            sliced.cu_num_generated_tokens == self.logprobsLists.cu_num_generated_tokens
        )
