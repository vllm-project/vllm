# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.forward_context import DPMetadata
from vllm.logger import init_logger
from vllm.utils import round_up

from vllm.v1.worker.ubatch_utils import UBatchSlices, UbatchSlice, is_second_ubatch_empty

logger = init_logger(__name__)

def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return DPMetadata.should_ubatch_across_dp(
        should_ubatch, orig_num_tokens_per_ubatch,
        padded_num_tokens_per_ubatch, dp_size, dp_rank)


def get_dp_padding_ubatch(
    num_tokens_unpadded: int, 
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    vllm_config: VllmConfig
) -> tuple[bool, int, Optional[torch.Tensor]]:
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        # Early exit.
        return False, 0, None

    if not should_attempt_ubatching:
        (should_ubatch,
            num_tokens_across_dp) = should_ubatch_with_num_tokens(
                False, 0, 0, vllm_config)
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, 0, num_tokens_across_dp

    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(f"Aborting ubatching {num_tokens_unpadded} {num_tokens_padded}")
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch,
        num_tokens_across_dp) = should_ubatch_with_num_tokens(
            should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch, vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, 0, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    num_pad_tokens = max_tokens_across_dp_cpu - num_tokens_per_ubatch
    num_pad_tokens = ((num_pad_tokens + num_tokens_per_ubatch) * 2) - \
        num_tokens_unpadded
    assert num_pad_tokens >= 0
    return should_ubatch, num_pad_tokens, num_tokens_after_padding

def ubatch_split(max_num_scheduled_tokens: int,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        vllm_config: VllmConfig,
    ) -> tuple[Optional[UBatchSlices], int, Optional[torch.Tensor]]:
    parallel_config = vllm_config.parallel_config
    # Don't bother with the should_ubatch handshaking unless microbatching
    # is enabled
    if not parallel_config.enable_microbatching:
        return (None, 0, None)

    # Check preconditions for microbatching
    should_attempt_ubatching = \
        parallel_config.enable_microbatching and \
        num_tokens_unpadded >= \
        parallel_config.microbatching_token_threshold \
        and max_num_scheduled_tokens == 1

    # Don't microbatch unless every other DP worker is also microbatching
    num_pad_tokens = 0
    num_tokens_after_padding = None
    (should_ubatch, num_pad_tokens,
        num_tokens_after_padding) = get_dp_padding_ubatch(num_tokens_unpadded, 
                                                          num_tokens_padded, 
                                                          should_attempt_ubatching, 
                                                          vllm_config)
    if not should_ubatch:
        return (None, 0, None)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_pad_tokens < num_tokens_unpadded,\
        f"num_pad_tokens {num_pad_tokens} "\
        f"original_num_tokens {num_tokens_unpadded}"
    total_num_tokens_per_ubatch = (num_tokens_unpadded +
                                    num_pad_tokens) // 2
    padded_first_ubatch_slice = slice(0, total_num_tokens_per_ubatch)
    padded_second_ubatch_slice = slice(total_num_tokens_per_ubatch,
                                        num_tokens_unpadded)

    # Note there's an assumption here that there's 1 token per request
    ubatch_slices = [
        UbatchSlice(padded_first_ubatch_slice, padded_first_ubatch_slice),
        UbatchSlice(padded_second_ubatch_slice, padded_second_ubatch_slice)
    ]

    return (ubatch_slices, num_pad_tokens, num_tokens_after_padding)