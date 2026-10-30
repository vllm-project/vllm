# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
allow_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

import numpy as np
import pytest
import torch

from vllm.config import ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import DPMetadata
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.pcp_manager import PCPManager


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
def pcp_dbo_parallel_config():
    return ParallelConfig(
        prefill_context_parallel_size=2,
        data_parallel_size=2,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        enable_dbo=True,
        dbo_prefill_token_threshold=2,
    )


@pytest.mark.parametrize(
    ("queries", "expected_ubatches"),
    [
        ([[7], [8]], 2),  # Different real counts, the same PCP-padded midpoint.
        ([[7], [9]], 1),  # DP padding would change the checked midpoint.
        ([[7, 20], [28]], 1),  # One DP replica vetoes a non-causal first slice.
    ],
)
def test_pcp_dbo_sync_agrees_across_all_ep_participants(
    queries, expected_ubatches, pcp_dbo_parallel_config
):
    """Transport simulation covers both DP subgroups in a PCP2 x DP2 EP group."""
    parallel = pcp_dbo_parallel_config
    decisions = []
    for pcp_rank in range(2):
        counts, votes = [], []
        for query_lens in queries:
            manager = PCPManager(2, pcp_rank, torch.device("cpu"), True)
            query_lens_np = np.array(query_lens, dtype=np.int32)
            prefilling = np.ones(len(query_lens), dtype=np.bool_)
            counts.append(
                manager.get_num_tokens_for_dispatch(query_lens_np, prefilling)
            )
            votes.append(manager.can_ubatch(query_lens_np, prefilling, counts[-1]))
        for dp_rank in range(2):
            peer = 1 - dp_rank
            peer_values = torch.tensor(
                [
                    counts[peer],
                    CUDAGraphMode.NONE.value,
                    0,
                    -1,
                    int(votes[peer]),
                    len(queries[peer]),
                    0,
                ]
            )

            def all_reduce(tensor, peer=peer, values=peer_values):
                tensor[:, peer] = values

            with mock.patch.object(dp_utils, "get_dp_group") as transport:
                transport.return_value.cpu_all_reduce_.side_effect = all_reduce
                desc, sync = dp_utils.dispatch_cg_and_sync_dp(
                    None,
                    num_reqs=len(queries[dp_rank]),
                    num_tokens=counts[dp_rank],
                    uniform_token_count=None,
                    dp_size=2,
                    dp_rank=dp_rank,
                    need_eager=True,
                    parallel_config=parallel,
                    allow_ubatching=votes[dp_rank],
                )
            assert sync is not None
            decisions.append(desc.num_ubatches)
            assert sync.num_tokens_across_dp.tolist() == counts
            if desc.num_ubatches == 2:
                split = desc.num_tokens // 2
                for width in (split, desc.num_tokens - split):
                    metadata = DPMetadata.make(
                        parallel, width, torch.full((2,), width, dtype=torch.int32)
                    )
                    with metadata.sp_local_sizes(1, pcp_size=2, use_ep=True) as sizes:
                        assert sizes == [width] * 4
    assert decisions == [expected_ubatches] * 4
