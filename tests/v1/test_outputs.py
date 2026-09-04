# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest import TestCase

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplingMaskLists,
)
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

DEVICE_TYPE = current_platform.device_type


def test_logprobs_tensors_cat():
    first = LogprobsTensors(
        torch.tensor([[1, 2]]),
        torch.tensor([[0.1, 0.2]]),
        torch.tensor([1]),
    )
    second = LogprobsTensors(
        torch.tensor([[3, 4]]),
        torch.tensor([[0.3, 0.4]]),
        torch.tensor([2]),
    )

    result = LogprobsTensors.cat([first, second], [0, 1, 2])

    assert result.logprob_token_ids.tolist() == [[1, 2], [3, 4]]
    assert result.logprobs.tolist() == (
        first.logprobs.tolist() + second.logprobs.tolist()
    )
    assert result.selected_token_ranks.tolist() == [1, 2]
    assert result.cu_num_generated_tokens == [0, 1, 2]
    assert LogprobsTensors.cat([first]) is first


def test_logprobs_tensors_tolists_with_tensor_boundaries():
    """Adaptive verification hands over the request boundaries as a tensor
    (they only exist on device); tolists() must materialize it as a plain
    list so slice_request splits requests correctly."""
    tensors = LogprobsTensors(
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        torch.tensor([1, 2, 3]),
        cu_num_generated_tokens_tensor=torch.tensor([0, 2, 3], dtype=torch.int32),
    )

    lists = tensors.to_cpu_nonblocking().tolists()

    assert lists.cu_num_generated_tokens == [0, 2, 3]
    sliced = lists.slice_request(1, 1)
    assert sliced.logprob_token_ids.tolist() == [[5, 6]]
    assert sliced.sampled_token_ranks.tolist() == [3]


def test_sampling_mask_lists_to_nested_list():
    mask = SamplingMaskLists(
        token_ids=np.array([10, 11, 12, 20, 21]),
        offsets=np.array([0, 3, 5]),
    )

    nested = mask.to_nested_list()

    assert nested == [[10, 11, 12], [20, 21]]
    assert SamplingMaskLists(np.array([7, 9])).to_nested_list() == [[7, 9]]


def test_sampling_mask_lists_slices_multiple_positions_by_request():
    masks = SamplingMaskLists(
        token_ids=np.array([10, 11, 20, 30, 31, 32, 40]),
        offsets=np.array([0, 2, 3, 6, 7]),
        cu_num_generated_tokens=[0, 1, 3, 4],
    )

    single = masks.slice_request(0, 1)
    multi = masks.slice_request(1, 2)

    assert single.to_nested_list() == [[10, 11]]
    assert single.offsets is None
    assert multi.to_nested_list() == [[20], [30, 31, 32]]
    assert multi.cu_num_generated_tokens is None


@pytest.mark.parametrize("max_num_kept", [512, 20_001])
def test_sampling_mask_tensors_match_finite_support(max_num_kept):
    """Whatever its size (empty, one, around the compact width, beyond the
    cap, the whole vocab), a sampled row's mask is exactly the ascending set
    of its finite logits; rows that sampled nothing are empty."""
    from vllm.v1.worker.gpu.sample.output import MAX_COMPACT_SUPPORT

    vocab_size = 20_001
    sizes = [0, 1, 7, 511, 512, 513, MAX_COMPACT_SUPPORT + 5, vocab_size, 40, 3]
    gen = torch.Generator().manual_seed(0)
    logits = torch.full((len(sizes), vocab_size), float("-inf"))
    expected = []
    for row, size in enumerate(sizes):
        kept = torch.randperm(vocab_size, generator=gen)[:size].sort().values
        logits[row, kept] = torch.randn(size, generator=gen)
        expected.append(kept.tolist())
    num_sampled_tokens = torch.ones(len(sizes), dtype=torch.int32)
    num_sampled_tokens[[1, 7]] = 0
    expected[1] = expected[7] = []

    tensors = SamplingMaskTensors.from_logits(
        logits.to(DEVICE_TYPE),
        torch.arange(len(sizes) + 1, device=DEVICE_TYPE, dtype=torch.int32),
        num_sampled_tokens.to(DEVICE_TYPE),
        max_num_kept,
    )

    assert tensors.token_ids.shape[1] == min(max_num_kept, MAX_COMPACT_SUPPORT)
    assert tensors.to_cpu_nonblocking().tolists().to_nested_list() == expected


def test_sampling_mask_tensors_uses_request_boundaries():
    logits = torch.full((6, 8), -float("inf"), device=DEVICE_TYPE)
    logits[0, [1, 3]] = 0
    logits[2, [2, 4, 6]] = 0
    logits[5, [0, 7]] = 0
    counts = torch.tensor([1, 0, 1], device=DEVICE_TYPE, dtype=torch.int32)

    tensors = SamplingMaskTensors.from_logits(
        logits,
        torch.tensor([0, 2, 5, 6], device=DEVICE_TYPE, dtype=torch.int32),
        counts,
        max_num_kept=3,
    )

    assert tensors.tolists().to_nested_list() == [[1, 3], [], [0, 7]]


def test_sampling_mask_tensors_multirow_request_layout():
    rows_per_request = 4
    counts = np.array([1, 2, 4, 0], dtype=np.int32)
    logits = torch.full((16, 16), -float("inf"), device=DEVICE_TYPE)
    expected = []
    for row in [0, 4, 5, 8, 9, 10, 11]:
        kept = torch.tensor(
            [row % 16, (row + 3) % 16, (row + 7) % 16], device=DEVICE_TYPE
        ).unique(sorted=True)
        logits[row, kept] = 0
        expected.append(kept.tolist())

    tensors = SamplingMaskTensors.from_logits(
        logits,
        torch.arange(0, 17, 4, device=DEVICE_TYPE, dtype=torch.int32),
        torch.from_numpy(counts).to(DEVICE_TYPE),
        max_num_kept=2,
        rows_per_request=rows_per_request,
    )
    result = tensors.to_cpu_nonblocking().tolists(counts)

    assert tensors.token_ids.shape == (16, 2)
    assert tensors.rows_per_request == rows_per_request
    assert result.to_nested_list() == expected
    assert result.cu_num_generated_tokens == [0, 1, 3, 7, 7]


def test_sampling_mask_preserves_top_k_boundary_ties():
    processed_logits = torch.tensor(
        [[6.0, 5.0, 4.0, 4.0, 4.0, -float("inf"), -float("inf")]],
        device=DEVICE_TYPE,
    )
    expected_token_ids = (
        torch.isfinite(processed_logits[0]).nonzero().flatten().tolist()
    )
    assert len(expected_token_ids) > 3

    tensors = SamplingMaskTensors.from_logits(
        processed_logits,
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_TYPE),
        num_sampled_tokens=torch.tensor([1], device=DEVICE_TYPE),
        max_num_kept=3,
    )
    result = tensors.tolists()

    assert result.to_nested_list() == [expected_token_ids]


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


def test_with_ec_conn_output_copies_shared_empty_output():
    """The shared empty output is copied, never written to."""
    ec_output = ECConnectorOutput(finished_sending={"mm_hash"})

    result = ModelRunnerOutput.with_ec_conn_output(EMPTY_MODEL_RUNNER_OUTPUT, ec_output)

    assert result is not EMPTY_MODEL_RUNNER_OUTPUT
    assert result.ec_connector_output is ec_output
    assert EMPTY_MODEL_RUNNER_OUTPUT.ec_connector_output is None
