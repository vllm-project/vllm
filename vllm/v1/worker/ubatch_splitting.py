# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.config import ParallelConfig
from vllm.forward_context import DPMetadata
from vllm.logger import init_logger
from vllm.utils import round_up
from vllm.v1.worker.ubatch_utils import (UBatchSlice, UBatchSlices,
                                         is_second_ubatch_empty)

logger = init_logger(__name__)


def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[bool, Optional[torch.Tensor]]:
    return DPMetadata.coordinate_batch_across_dp(should_ubatch,
                                                 orig_num_tokens_per_ubatch,
                                                 padded_num_tokens_per_ubatch,
                                                 dp_size, dp_rank)


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    dp_size: int,
    dp_rank: int,
) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded
    if dp_size == 1:
        # Early exit.
        return False, None

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = should_attempt_ubatching

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(
            "Empty second µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
        should_ubatch=should_ubatch,
        orig_num_tokens_per_ubatch=num_tokens_unpadded // 2,
        padded_num_tokens_per_ubatch=num_tokens_per_ubatch,
        dp_size=dp_size,
        dp_rank=dp_rank)

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    return should_ubatch, num_tokens_after_padding


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
