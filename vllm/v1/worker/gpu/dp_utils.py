# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.ubatch_utils import is_last_ubatch_empty

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

logger = init_logger(__name__)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    num_active_loras: int = 0,
    wants_ubatch: bool = False,
    num_ubatches: int = 1,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = 1 if wants_ubatch else 0
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    wants_ubatch_across_dp = tensor[3]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    ubatch_desc = _maybe_ubatch_descriptor(
        num_tokens_across_dp, wants_ubatch_across_dp, num_reqs, num_ubatches
    )
    if ubatch_desc is not None:
        # Microbatching needs every rank to run the same number of tokens, so
        # that each rank can assume the others' microbatches are the same size.
        num_tokens_across_dp = torch.full_like(
            num_tokens_across_dp, ubatch_desc.num_tokens
        )
        return ubatch_desc, num_tokens_across_dp

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

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def _maybe_ubatch_descriptor(
    num_tokens_across_dp: torch.Tensor,
    wants_ubatch_across_dp: torch.Tensor,
    num_reqs: int,
    num_ubatches: int,
) -> BatchExecutionDescriptor | None:
    """Decide whether the group microbatches this step, and at what size.

    Microbatching is all-or-nothing: every rank has to split, because the
    expert all-to-all is collective. Returns the descriptor all ranks will run,
    or None to fall through to the regular (single batch) path.
    """
    if num_ubatches <= 1 or not torch.all(wants_ubatch_across_dp == 1).item():
        return None

    # Every rank runs the largest rank's token count, so pad up to it.
    num_tokens = int(num_tokens_across_dp.max().item())
    if is_last_ubatch_empty(
        int(num_tokens_across_dp.min().item()), num_tokens, num_ubatches
    ):
        # The smallest rank has too few tokens to fill every microbatch.
        logger.debug(
            "Skipping microbatching: %d tokens do not fill %d microbatches of %d",
            int(num_tokens_across_dp.min().item()),
            num_ubatches,
            num_tokens,
        )
        return None

    # Microbatched steps run eager for now; no CUDA graphs are captured for
    # them yet, so there is nothing to dispatch to.
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_ubatches=num_ubatches,
    )


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
    num_active_loras: int = 0,
    ubatch_runner: UBatchRunner | None = None,
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
        )

    if dp_size == 1:
        # Microbatching needs the DP handshake to agree on it, so it is only
        # available with more than one DP rank (as in the V1 runner).
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        num_active_loras=num_active_loras,
        wants_ubatch=(
            ubatch_runner is not None
            and ubatch_runner.wants_ubatch(num_tokens, uniform_token_count)
        ),
        num_ubatches=ubatch_runner.num_ubatches if ubatch_runner is not None else 1,
    )
