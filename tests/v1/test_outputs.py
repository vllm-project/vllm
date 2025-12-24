# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest import TestCase

import numpy as np

from vllm.v1.outputs import LogprobsLists, TrackedLogprobsLists


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

        sliced = logprobsLists.slice_request(1, num_positions=2)
        assert sliced.logprob_token_ids == [[2], [3]]
        assert sliced.logprobs == [[0.2], [0.3]]
        assert sliced.sampled_token_ranks == [2, 3]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_start(self):
        """Test slicing from the start position"""
        sliced = self.logprobsLists.slice_request(0, num_positions=5)
        assert len(sliced.logprob_token_ids) == 5
        assert sliced.logprob_token_ids == [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_middle(self):
        """Test slicing from the middle position"""
        sliced = self.logprobsLists.slice_request(1, num_positions=7)
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
        assert sliced.cu_num_generated_tokens is None

    def test_slice_single_request(self):
        """Test slicing a single request"""
        sliced = self.logprobsLists.slice_request(1, num_positions=3)
        assert len(sliced.logprob_token_ids) == 3
        assert sliced.logprob_token_ids == [[5, 6], [7, 8], [9, 10]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_last_request(self):
        """Test slicing the last request"""
        sliced = self.logprobsLists.slice_request(2, num_positions=4)
        assert len(sliced.logprob_token_ids) == 4
        assert sliced.logprob_token_ids == [[11, 12], [13, 14], [15, 16], [17, 18]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_all_requests(self):
        """Test slicing all requests (full slice)"""
        sliced = self.logprobsLists.slice_request(0, num_positions=9)
        assert len(sliced.logprob_token_ids) == 9  # All tokens
        assert sliced.logprob_token_ids == self.logprobsLists.logprob_token_ids
        assert sliced.cu_num_generated_tokens is None


class TestTrackedLogprobsLists(TestCase):
    """Tests for TrackedLogprobsLists slicing operations."""

    def setUp(self):
        """Create sample TrackedLogprobsLists with known data.

        Structure:
        - 3 requests with different numbers of generated tokens
        - Request 0: 2 tokens (rows 0-1)
        - Request 1: 1 token (row 2)
        - Request 2: 3 tokens (rows 3-5)
        - 3 tracked token IDs: [100, 200, 300]
        """
        self.tracked = TrackedLogprobsLists(
            logprobs=np.array(
                [
                    [-1.0, -2.0, -3.0],  # Request 0, token 0
                    [-1.1, -2.1, -3.1],  # Request 0, token 1
                    [-1.2, -2.2, -3.2],  # Request 1, token 0
                    [-1.3, -2.3, -3.3],  # Request 2, token 0
                    [-1.4, -2.4, -3.4],  # Request 2, token 1
                    [-1.5, -2.5, -3.5],  # Request 2, token 2
                ]
            ),
            token_ids=[100, 200, 300],
            cu_num_generated_tokens=[0, 2, 3, 6],
        )

    def test_slice_without_cu_num_generated_tokens(self):
        """Test slicing without cu_num_generated_tokens uses req_idx directly."""
        tracked = TrackedLogprobsLists(
            logprobs=np.array(
                [
                    [-1.0, -2.0],
                    [-1.1, -2.1],
                    [-1.2, -2.2],
                ]
            ),
            token_ids=[100, 200],
            cu_num_generated_tokens=None,
        )

        # Without cu_num_generated_tokens, req_idx is used as start index
        sliced = tracked.slice_request(1, num_positions=2)

        # Should slice rows 1 and 2
        np.testing.assert_array_almost_equal(
            sliced.logprobs, np.array([[-1.1, -2.1], [-1.2, -2.2]])
        )
        assert sliced.token_ids == [100, 200]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_first_request(self):
        """Test slicing the first request."""
        sliced = self.tracked.slice_request(0, num_positions=2)

        # Request 0 has 2 tokens at rows 0-1
        np.testing.assert_array_almost_equal(
            sliced.logprobs, np.array([[-1.0, -2.0, -3.0], [-1.1, -2.1, -3.1]])
        )
        assert sliced.token_ids == [100, 200, 300]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_middle_request(self):
        """Test slicing a middle request."""
        sliced = self.tracked.slice_request(1, num_positions=1)

        # Request 1 has 1 token at row 2
        np.testing.assert_array_almost_equal(
            sliced.logprobs, np.array([[-1.2, -2.2, -3.2]])
        )
        assert sliced.token_ids == [100, 200, 300]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_last_request(self):
        """Test slicing the last request."""
        sliced = self.tracked.slice_request(2, num_positions=3)

        # Request 2 has 3 tokens at rows 3-5
        np.testing.assert_array_almost_equal(
            sliced.logprobs,
            np.array(
                [
                    [-1.3, -2.3, -3.3],
                    [-1.4, -2.4, -3.4],
                    [-1.5, -2.5, -3.5],
                ]
            ),
        )
        assert sliced.token_ids == [100, 200, 300]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_single_token(self):
        """Test slicing a request with a single token."""
        # Create a tracked list where each request has 1 token
        tracked = TrackedLogprobsLists(
            logprobs=np.array(
                [
                    [-1.0, -2.0],
                    [-3.0, -4.0],
                    [-5.0, -6.0],
                ]
            ),
            token_ids=[100, 200],
            cu_num_generated_tokens=[0, 1, 2, 3],
        )

        sliced = tracked.slice_request(1, num_positions=1)

        np.testing.assert_array_almost_equal(sliced.logprobs, np.array([[-3.0, -4.0]]))
        assert sliced.token_ids == [100, 200]

    def test_slice_preserves_token_ids(self):
        """Test that token_ids are always preserved after slicing."""
        # Slice different requests and verify token_ids are unchanged
        for req_idx in range(3):
            num_tokens = [2, 1, 3][req_idx]  # Tokens per request
            sliced = self.tracked.slice_request(req_idx, num_positions=num_tokens)
            assert sliced.token_ids == [100, 200, 300]

    def test_slice_with_single_tracked_token(self):
        """Test slicing when only tracking a single token."""
        tracked = TrackedLogprobsLists(
            logprobs=np.array(
                [
                    [-1.0],
                    [-2.0],
                    [-3.0],
                ]
            ),
            token_ids=[42],
            cu_num_generated_tokens=[0, 1, 2, 3],
        )

        sliced = tracked.slice_request(1, num_positions=1)

        np.testing.assert_array_almost_equal(sliced.logprobs, np.array([[-2.0]]))
        assert sliced.token_ids == [42]

    def test_slice_with_many_tracked_tokens(self):
        """Test slicing when tracking many tokens (e.g., 100-class classification)."""
        num_tracked = 100
        num_tokens = 5
        token_ids = list(range(num_tracked))

        tracked = TrackedLogprobsLists(
            logprobs=np.random.randn(num_tokens, num_tracked),
            token_ids=token_ids,
            cu_num_generated_tokens=[0, 2, 5],  # 2 requests: 2 tokens, 3 tokens
        )

        sliced = tracked.slice_request(1, num_positions=3)

        assert sliced.logprobs.shape == (3, num_tracked)
        assert sliced.token_ids == token_ids
        assert len(sliced.token_ids) == num_tracked

    def test_slice_empty_result(self):
        """Test slicing with num_positions=0 returns empty array."""
        sliced = self.tracked.slice_request(0, num_positions=0)

        assert sliced.logprobs.shape == (0, 3)
        assert sliced.token_ids == [100, 200, 300]
