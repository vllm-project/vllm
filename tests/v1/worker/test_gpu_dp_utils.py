# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
allow_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _run_sync(local_allow_skip: bool, other_allow_skip: bool) -> bool:
    """Run sync_cudagraph_and_dp_padding as dp rank 0 of 2, with rank 1's
    all_reduce contribution simulated."""

    def fake_all_reduce(tensor):
        tensor[0][1] = 8  # rank 1 num_tokens
        tensor[1][1] = CUDAGraphMode.NONE.value
        tensor[2][1] = 0  # uniform_token_count (0 means None)
        tensor[3][1] = -1  # max_query_len (-1 means None)
        tensor[4][1] = 0  # allow_ubatching
        tensor[5][1] = 2  # rank 1 num_reqs
        tensor[6][1] = int(other_allow_skip)

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=2
    )
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        _, sync = dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=None,
            desired_batch_desc=desc,
            num_tokens=4,
            num_reqs=2,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            allow_skip_drafts=local_allow_skip,
        )
    assert sync is not None
    return sync.skip_drafts


def test_skip_drafts_requires_all_ranks():
    assert _run_sync(True, True)
    assert not _run_sync(True, False)
    assert not _run_sync(False, True)
    assert not _run_sync(False, False)


def test_dispatch_dp1_returns_no_sync():
    # With a single rank there is no agreement to make; callers fall back to
    # their local allow_skip_drafts value.
    _, sync = dp_utils.dispatch_cg_and_sync_dp(
        cudagraph_manager=None,
        num_reqs=2,
        num_tokens=4,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        allow_skip_drafts=True,
    )
    assert sync is None


@pytest.fixture
def bounded_dbo_parallel_config():
    return ParallelConfig(
        enable_dbo=True,
        dbo_ubatch_capacity=3,
        all2all_backend="deepep_high_throughput",
        enable_expert_parallel=True,
        data_parallel_size=2,
        data_parallel_size_local=2,
        distributed_executor_backend="mp",
        dbo_decode_token_threshold=8,
        dbo_prefill_token_threshold=16,
    )


@pytest.mark.parametrize(
    ("tokens", "uniform", "votes", "expected"),
    [
        ([16, 16], True, [1, 1], 2),
        ([24, 32], True, [1, 1], 3),
        ([23, 96], True, [1, 1], 2),
        ([48, 48], False, [1, 1], 3),
        ([47, 48], False, [1, 1], 2),
        ([0, 32], True, [1, 1], 1),
        ([32, 32], True, [1, 0], 1),
    ],
)
def test_bounded_dbo_dp_agrees_on_active_count(
    bounded_dbo_parallel_config, tokens, uniform, votes, expected
):
    """Every EP rank chooses from the same minimum, including idle/veto fallback."""
    for rank in (0, 1):

        def all_reduce(tensor):
            tensor[0] = torch.tensor(tokens)
            tensor[1].fill_(CUDAGraphMode.NONE.value)
            tensor[2].fill_(int(uniform))
            tensor[3].fill_(-1)
            tensor[4] = torch.tensor(votes)
            tensor[5] = torch.tensor(tokens)
            tensor[6].zero_()

        with mock.patch.object(dp_utils, "get_dp_group") as group:
            group.return_value.cpu_all_reduce_ = all_reduce
            desc, sync = dp_utils.sync_cudagraph_and_dp_padding(
                None,
                BatchExecutionDescriptor(
                    CUDAGraphMode.NONE, tokens[rank], tokens[rank]
                ),
                num_tokens=tokens[rank],
                num_reqs=tokens[rank],
                uniform_token_count=int(uniform) or None,
                dp_size=2,
                dp_rank=rank,
                parallel_config=bounded_dbo_parallel_config,
                allow_ubatching=bool(votes[rank]),
                uniform_decode=uniform,
            )
        assert desc.num_ubatches == expected
        assert sync is not None and sync.eager
        if expected > 1:
            assert sync.num_tokens_across_dp.tolist() == [max(tokens)] * 2
