# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
want_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _run_sync(local_want_skip: bool, other_want_skip: bool) -> bool:
    """Run sync_cudagraph_and_dp_padding as dp rank 0 of 2, with rank 1's
    all_reduce contribution simulated."""

    def fake_all_reduce(tensor, group=None):
        tensor[0][1] = 8  # rank 1 num_tokens
        tensor[1][1] = CUDAGraphMode.NONE.value
        tensor[2][1] = 0
        tensor[3][1] = int(other_want_skip)

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=2
    )
    with (
        mock.patch.object(dp_utils, "get_dp_group") as mock_group,
        mock.patch.object(dp_utils.dist, "all_reduce", fake_all_reduce),
    ):
        mock_group.return_value.cpu_group = None
        _, _, skip_drafts = dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=None,
            desired_batch_desc=desc,
            num_tokens=4,
            num_reqs=2,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            want_skip_drafts=local_want_skip,
        )
    return skip_drafts


def test_skip_drafts_requires_all_ranks():
    assert _run_sync(True, True)
    assert not _run_sync(True, False)
    assert not _run_sync(False, True)
    assert not _run_sync(False, False)


def test_dispatch_dp1_passes_skip_through():
    for want in (False, True):
        _, _, skip_drafts = dp_utils.dispatch_cg_and_sync_dp(
            cudagraph_manager=None,
            num_reqs=2,
            num_tokens=4,
            uniform_token_count=None,
            dp_size=1,
            dp_rank=0,
            need_eager=True,
            want_skip_drafts=want,
        )
        assert skip_drafts is want
