# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


class DPProfilerSync:
    """Starts the torch profiler on the same step across all DP ranks.

    ``start_profile`` reaches each DP rank asynchronously, at a different point
    in its own step loop, and every step all DP ranks must jointly execute the
    EP all-to-all / DP coordination collective. A separate barrier on the
    profiler control path therefore deadlocks: the first rank to reach it stops
    stepping, so the others wedge on the next collective before they ever reach
    their own barrier (see VLLM_ENABLE_MULTINODE_PROFILING).

    Instead this rides the per-step DP coordination all-reduce that every rank
    already executes in lockstep. ``request_start`` sets a pending flag; the
    flag is OR-reduced across DP ranks inside ``sync_cudagraph_and_dp_padding``;
    once any rank has requested it, ``start_now`` latches on every rank on the
    same step, and the worker starts capture next step. No extra collective, no
    deadlock, and it needs only one rank to receive start_profile (the OR
    propagates it).
    """

    def __init__(self) -> None:
        # start_profile received, capture not yet begun.
        self._pending = False
        # Latched consensus, cleared when the worker consumes it.
        self.start_now = False

    def request_start(self) -> None:
        self._pending = True

    def cancel(self) -> None:
        """Drop a pending request (e.g. stop_profile before capture began)."""
        self._pending = False
        self.start_now = False

    def observe(self, consensus: bool) -> None:
        """Record the OR-reduced request flag from a DP coordination reduce."""
        if consensus:
            self.start_now = True

    def consume_start(self) -> bool:
        """Return True once, on the step every rank agreed to start capture."""
        if self.start_now:
            self.start_now = False
            self._pending = False
            return True
        return False


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    profiler_sync: DPProfilerSync | None = None,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    # Row 4 (profiler start request) only added under VLLM_ENABLE_MULTINODE_PROFILING.
    tensor = torch.zeros(
        5 if profiler_sync is not None else 4, dp_size, dtype=torch.int32, device="cpu"
    )
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    if profiler_sync is not None:
        tensor[4][dp_rank] = 1 if profiler_sync._pending else 0
    dist.all_reduce(tensor, group=group)

    # Latch the OR-reduced profiler start request across ranks.
    if profiler_sync is not None:
        profiler_sync.observe(bool(tensor[4].any().item()))

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=desired_batch_desc.num_active_loras,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1).item()):
        synced_max_query_len = int(max_query_lens_across_dp.max().item())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    need_eager: bool = False,
    num_active_loras: int = 0,
    profiler_sync: DPProfilerSync | None = None,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
            max_query_len=max_query_len,
        )

    if dp_size == 1:
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
        profiler_sync=profiler_sync,
    )
