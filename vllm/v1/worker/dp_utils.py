# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

logger = init_logger(__name__)

# Conservative decode-only gate for CPU DP padding (see _cpu_dp_pad_worthwhile):
# only pad when every rank's batch is already small and the ranks are close in
# size, so padding to max adds at most a couple of GEMM rows per rank.
CPU_DP_PAD_MAX_TOKENS = 4
CPU_DP_PAD_MAX_SPREAD = 2


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
            "Using CPU all reduce to synchronize DP padding between ranks.",
        )
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    # Populate this rank's contribution on CPU to reduce GPU syncs.
    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)
    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
    tensor_cpu[3][dp_rank] = cudagraph_mode

    # On CPU, route through the DP group's SHM all-reduce when available to
    # avoid a per-step gloo round-trip. The SHM kernel is float-only and
    # reduces in FP32, so round-trip via fp32: the coordination values are
    # tiny integers (< 2^24) and each rank writes only its own column, so the
    # SUM stays exact per column.
    dp_group = get_dp_group()
    comm = dp_group.device_communicator
    use_shm = (
        current_platform.is_cpu()
        and comm is not None
        and getattr(comm, "supports_tensor_dict", False)
    )
    if use_shm:
        tensor_fp32 = tensor_cpu.to(torch.float32)
        dp_group.all_reduce(tensor_fp32)  # in place -> shm_allreduce
        return tensor_fp32.round().to(torch.int32)

    device, group = _get_device_and_group(parallel_config)
    tensor = tensor_cpu.to(device, non_blocking=True)
    dist.all_reduce(tensor, group=group)
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
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
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # Pad every rank up to the synchronized max token count.
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """Synchronize cudagraph_mode across DP ranks by taking the minimum."""
    return int(tensor[3, :].min().item())


def _cpu_dp_pad_worthwhile(tensor: torch.Tensor) -> bool:
    # Decode-only, near-uniform gate: pad to max ONLY when every rank's batch
    # is already small and the ranks are close in size, so padding to max adds
    # at most a couple of GEMM rows per rank. Never pad ragged prefill batches
    # (large max, wide spread) where padding would add real GEMM work.
    counts = tensor[1, :]
    mx = int(counts.max().item())
    mn = int(counts.min().item())
    return mx <= CPU_DP_PAD_MAX_TOKENS and (mx - mn) <= CPU_DP_PAD_MAX_SPREAD


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    When running microbatched or if cudagraph is enabled (synced across ranks),
    all ranks will be padded out so that they run with the same number of tokens.

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including any DP padding.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded

    # Coordinate token counts, ubatching, and cudagraph mode across DP ranks.
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    # Synchronize cudagraph_mode across ranks first (take min).
    # This is needed before DP padding decision since we use the synced
    # cudagraph mode to determine whether DP padding is needed.
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    # DP padding is required for synced cudagraph execution and for ubatching,
    # which still assumes uniform per-rank token counts. On CPU, also pad when
    # it's cheap (small, near-uniform decode batches) to unlock the uniform
    # all-gather fast path and skip the ragged trim.
    should_dp_pad = (
        synced_cudagraph_mode != 0
        or should_ubatch
        or (current_platform.is_cpu() and _cpu_dp_pad_worthwhile(tensor))
    )

    # Return either the synchronized max or the unpadded per-rank counts.
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL).
            DP padding is enabled when synced cudagraph mode across ranks is not NONE.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when cudagraph is enabled.
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
            cudagraph_mode,
            parallel_config,
        )
    )

    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
