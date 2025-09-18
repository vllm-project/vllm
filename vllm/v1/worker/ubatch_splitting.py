# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm.v1.worker.utils import coordinate_batch_across_dp

logger = init_logger(__name__)


def ubatch_split(
    max_num_scheduled_tokens: int,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    parallel_config: ParallelConfig,
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
    should_attempt_ubatching = \
        parallel_config.enable_dbo and \
        num_tokens_unpadded >= \
        parallel_config.dbo_decode_token_threshold \
        and max_num_scheduled_tokens == 1

    # Don't microbatch unless every other DP worker is also microbatching
    num_tokens_after_padding = None
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
    total_num_tokens_per_ubatch = int(num_tokens_after_padding[0].item())
    padded_first_ubatch_slice = slice(0, total_num_tokens_per_ubatch)
    padded_second_ubatch_slice = slice(total_num_tokens_per_ubatch,
                                       num_tokens_unpadded)

    # Note there's an assumption here that there's 1 token per request
    ubatch_slices = [
        UBatchSlice(padded_first_ubatch_slice, padded_first_ubatch_slice),
        UBatchSlice(padded_second_ubatch_slice, padded_second_ubatch_slice)
    ]

    return (ubatch_slices, num_tokens_after_padding)
