# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import numpy as np
import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices

logger = init_logger(__name__)


def check_ubatch_thresholds(config: ParallelConfig, num_tokens: int,
                            uniform_decode: bool) -> bool:
    if not config.enable_dbo:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


def create_ubatch_slices(num_scheduled_tokens: np.ndarray, split_point: int) \
    -> UBatchSlices:
    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    first_ubatch_token_slice = slice(0, split_point)
    second_ubatch_token_slice = slice(split_point, cu_num_tokens[-1])

    # Determine request slices using exclusive stop semantics
    # First ubatch includes requests whose tokens overlap [0, split_point)
    first_ubatch_req_stop = int(
        np.searchsorted(cu_num_tokens, split_point, side="left"))
    first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

    # Second ubatch starts at the request that contains the split_point
    # or the request starting exactly at split_point (if on boundary)
    second_ubatch_req_start = int(
        np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
    second_ubatch_req_slice = slice(second_ubatch_req_start,
                                    len(cu_num_tokens) - 1)

    return [
        UBatchSlice(first_ubatch_req_slice, first_ubatch_token_slice),
        UBatchSlice(second_ubatch_req_slice, second_ubatch_token_slice)
    ]


def ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    parallel_config: ParallelConfig,
    allow_microbatching: bool,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to 
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if ubatch_slices is None
    ]

    """
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank

    # Check preconditions for microbatching
    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        True,  #TODO Fix
    )

    if not allow_microbatching:
        should_attempt_ubatching = False
    # Don't microbatch unless every other DP worker is also microbatching
    (should_ubatch, num_tokens_after_padding) = coordinate_batch_across_dp(
        num_tokens_unpadded, num_tokens_padded, should_attempt_ubatching,
        dp_size, dp_rank)
    if not should_ubatch:
        return (None, num_tokens_after_padding)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_tokens_after_padding is not None
    token_split_point = int(num_tokens_after_padding[0].item()) // 2

    ubatch_slices = create_ubatch_slices(num_scheduled_tokens_per_request,
                                         token_split_point)

    return (ubatch_slices, num_tokens_after_padding)
