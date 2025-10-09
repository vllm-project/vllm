# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    create_ubatch_slices,
    is_second_ubatch_empty,
)

logger = init_logger(__name__)


def _get_device_and_group(parallel_config: ParallelConfig):
    device = current_platform.device_type
    group = get_dp_group().device_group

    # Transfering this tensor from GPU to CPU will introduce a GPU sync
    # point that could adversely affect performance of vllm with asynch
    # scheduling. This environment variable exists to quickly disable
    # this optimization if we run into this case.
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once("Using CPU all reduce to syncronize DP padding between ranks.")
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    tensor = torch.zeros(3, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0
    dist.all_reduce(tensor, group=group)
    return tensor


def _post_process_ubatch(tensor: torch.Tensor) -> bool:
    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    # First determine if we are going to be ubatching.
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    # If the DP ranks are planning to ubatch, make sure that
    # there are no "empty" second ubatches
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_second_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    parallel_config: ParallelConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding.
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded

    # First we coordinate between the DP ranks via an All Reduce
    # to determine the total number of tokens that each rank
    # will run and if we are using ubatching or not.
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        parallel_config=parallel_config,
    )

    # Ensure that each rank is processing the same nuber of tokens
    num_tokens_across_dp = tensor[1, :]
    max_num_tokens = int(num_tokens_across_dp.max().item())
    num_tokens_after_padding = torch.tensor(
        [max_num_tokens] * len(num_tokens_across_dp), device="cpu", dtype=torch.int32
    )

    should_ubatch = _post_process_ubatch(tensor)

    return should_ubatch, num_tokens_after_padding


def coordinate_batch_across_dp(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    parallel_config: ParallelConfig,
    allow_microbatching: bool,
    uniform_decode: bool,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding.
    ]

    """
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return None, None

    # Check preconditions for microbatching
    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )

    # If the caller has explicitly disabled microbatching.
    if not allow_microbatching:
        should_attempt_ubatching = False

    (should_ubatch, num_tokens_after_padding) = _synchronize_dp_ranks(
        num_tokens_unpadded,
        num_tokens_padded,
        should_attempt_ubatching,
        parallel_config,
    )

    # Don't microbatch unless every other DP worker is also microbatching
    if not should_ubatch:
        return (None, num_tokens_after_padding)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_tokens_after_padding is not None
    token_split_point = int(num_tokens_after_padding[0].item()) // 2

    ubatch_slices = create_ubatch_slices(
        num_scheduled_tokens_per_request, token_split_point
    )

    return (ubatch_slices, num_tokens_after_padding)
