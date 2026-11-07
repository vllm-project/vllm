# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
allow_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

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


def _sync_ubatching(
    local_num_tokens: int,
    other_num_tokens: int,
    num_ubatches: int = 2,
    graph_num_tokens: int | None = None,
) -> tuple[int, CUDAGraphMode]:
    """Run as dp rank 0 of 2 with DBO allowed on both ranks; returns the
    synced descriptor's num_ubatches and CUDA graph mode. With graph_num_tokens,
    a FULL microbatched graph of that size matches the step."""

    def fake_all_reduce(tensor):
        tensor[0][1] = other_num_tokens
        tensor[1][1] = CUDAGraphMode.NONE.value
        tensor[2][1] = 0
        tensor[3][1] = -1
        tensor[4][1] = 1  # allow_ubatching
        tensor[5][1] = 1
        tensor[6][1] = 0

    parallel_config = mock.Mock(
        enable_fault_tolerance=False,
        use_ubatching=True,
        num_ubatches=num_ubatches,
        dbo_prefill_token_threshold=1,
        dbo_decode_token_threshold=1,
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=local_num_tokens, num_reqs=1
    )
    cudagraph_manager = None
    if graph_num_tokens is not None:
        cudagraph_manager = mock.Mock()
        cudagraph_manager.dispatch.return_value = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=graph_num_tokens,
            num_reqs=1,
            num_ubatches=num_ubatches,
        )
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        synced_desc, _ = dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=cudagraph_manager,
            desired_batch_desc=desc,
            num_tokens=local_num_tokens,
            num_reqs=1,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            parallel_config=parallel_config,
            allow_ubatching=True,
        )
    return synced_desc.num_ubatches or 1, synced_desc.cg_mode


def test_ubatching_survives_a_rank_with_an_empty_last_ubatch():
    # Padded to 600 tokens, rank 1's 100 tokens all land in microbatch 0. Every
    # rank still splits so the expert all-to-all stays collective.
    assert _sync_ubatching(600, 100)[0] == 2
    assert _sync_ubatching(600, 500)[0] == 2


def test_eager_when_a_rank_exactly_fills_the_earlier_ubatches():
    # Rank 1's 64 tokens exactly fill microbatch 0 of a 128-token graph, so its
    # microbatch 1 is pure padding; a captured graph cannot replay that.
    assert _sync_ubatching(128, 64, 2, 128) == (2, CUDAGraphMode.NONE)
    assert _sync_ubatching(128, 65, 2, 128) == (2, CUDAGraphMode.FULL)


def test_eager_when_graph_padding_empties_a_last_ubatch():
    # With 3 microbatches, padding 600 tokens to a 640-token graph moves the last
    # microbatch's start to 426, past rank 1's 420 tokens; unpadded it starts at 400.
    assert _sync_ubatching(600, 420, 3, 640) == (3, CUDAGraphMode.NONE)
    assert _sync_ubatching(600, 500, 3, 640) == (3, CUDAGraphMode.FULL)
