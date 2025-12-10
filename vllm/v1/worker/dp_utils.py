# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_second_ubatch_empty,
)

logger = init_logger(__name__)


def _get_device_and_group(parallel_config: ParallelConfig):
    # Use the actual device assigned to the DP group, not just the device type
    device = get_dp_group().device
    group = get_dp_group().device_group

    # Transferring this tensor from GPU to CPU will introduce a GPU sync
    # point that could adversely affect performance of vllm with asynch
    # scheduling. This environment variable exists to quickly disable
    # this optimization if we run into this case.
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks."
        )
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    should_dp_pad: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    tensor = torch.zeros(5, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0
    tensor[3][dp_rank] = 1 if should_dp_pad else 0
    tensor[4][dp_rank] = cudagraph_mode
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


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # If DP padding is enabled, ensure that each rank is processing the same number
        # of tokens
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[4, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    should_attempt_dp_padding: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    When running microbatched or if should_attempt_dp_padding is True, all
    ranks will be padded out so that the run with the same number of tokens

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including any DP padding.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded

    # Coordinate between the DP ranks via an All Reduce
    # to determine the total number of tokens that each rank
    # will run and if we are using ubatching or not.
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        should_dp_pad=should_attempt_dp_padding,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    should_dp_pad = bool(torch.all(tensor[3] == 1).item())

    # DP ranks should all have the same value for should_attempt_dp_padding.
    assert should_attempt_dp_padding == should_dp_pad

    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor)

    if should_ubatch and not should_dp_pad:
        logger.debug_once(
            "Microbatching has been triggered and requires DP padding. "
            "Enabling DP padding even though it has been explicitly "
            "disabled.",
            scope="global",
        )
        should_dp_pad = True

    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    # Synchronize cudagraph_mode across ranks (take min)
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    allow_dp_padding: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    num_scheduled_tokens_per_request: np.ndarray | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        allow_dp_padding: If all DP ranks should be padded up to the same value
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        num_scheduled_tokens_per_request: Only used if allow_microbatching is True. The
            number of tokens per request.
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL)

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when allow_dp_padding
        is True.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode

    # If the caller has explicitly enabled microbatching.
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
        _synchronize_dp_ranks(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            allow_dp_padding,
            cudagraph_mode,
            parallel_config,
        )
    )

    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
