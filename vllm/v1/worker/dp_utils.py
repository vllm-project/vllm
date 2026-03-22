# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""数据并行工具函数模块。

本模块提供数据并行（DP）相关的辅助函数，负责：
- 同步 DP ranks 之间的批次信息
- 协调微批次（microbatching）决策
- 管理 DP padding

主要函数：
- _get_device_and_group: 获取 DP 设备和组
- _run_ar: 执行 all-reduce 同步
- _post_process_ubatch: 后处理微批次决策
- _post_process_dp_padding: 后处理 DP padding
- _post_process_cudagraph_mode: 后处理 CUDA Graph 模式
- _synchronize_dp_ranks: 同步 DP ranks
- coordinate_batch_across_dp: 协调跨 DP 的批次
"""


import numpy as np
import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

logger = init_logger(__name__)


def _get_device_and_group(parallel_config: ParallelConfig):
    """获取 DP 组和对应的设备。

    使用 DP 组实际分配的设备，而不仅仅是设备类型。
    如果禁用了 NCCL 用于 DP 同步，则回退到 CPU all-reduce。

    Args:
        parallel_config: 并行配置

    Returns:
        (设备，组) 元组
    """
    # 使用 DP 组分配的实际设备，而不仅仅是设备类型
    device = get_dp_group().device
    group = get_dp_group().device_group

    # 将此张量从 GPU 传输到 CPU 会引入 GPU 同步点，
    # 这可能会对异步调度的 vllm 性能产生不利影响。
    # 如果我们遇到这种情况，此环境变量可以快速禁用此优化。
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "使用 CPU all reduce 来同步 ranks 之间的 DP padding。",
            scope="local",
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
    """执行 all-reduce 同步 DP ranks 的信息。

    Args:
        should_ubatch: 是否应该微批次
        orig_num_tokens_per_ubatch: 每个微批次的原始 token 数
        padded_num_tokens_per_ubatch: 每个微批次的 padding 后 token 数
        cudagraph_mode: CUDA Graph 模式
        parallel_config: 并行配置

    Returns:
        同步后的张量
    """
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    tensor = torch.zeros(4, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0
    tensor[3][dp_rank] = cudagraph_mode
    dist.all_reduce(tensor, group=group)
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    """后处理微批次决策。

    Args:
        tensor: all-reduce 同步后的张量
        num_ubatches: 微批次数量

    Returns:
        是否应该微批次
    """
    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    # 首先确定是否要微批次
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    # 如果 DP ranks 计划微批次，确保没有"空"的第二个微批次
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        logger.debug(
            "中止微批次 %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    """后处理 DP padding。

    Args:
        tensor: all-reduce 同步后的张量
        should_dp_pad: 是否应该 DP padding

    Returns:
        padding 后的 token 数量张量
    """
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # 如果启用了 DP padding，确保每个 rank 处理相同数量的 token
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """后处理 CUDA Graph 模式。

    通过取最小值来同步 DP ranks 之间的 cudagraph_mode。
    如果任何 rank 有 NONE (0)，所有 ranks 都使用 NONE。
    这确保所有 ranks 发送一致的值（全部 padding 或全部不 padding）。

    Args:
        tensor: all-reduce 同步后的张量

    Returns:
        同步后的 CUDA Graph 模式
    """
    return int(tensor[3, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """同步 DP ranks 之间的信息。

    此函数执行以下操作：
    1. 决定每个 DP rank 是否使用微批次。要么所有 ranks 都使用微批次，
       要么都不使用。
    2. 确定每个 rank 将运行的 token 总数。
       当运行微批次或启用 cudagraph 时（跨 ranks 同步），
       所有 ranks 都将 padding 到相同数量的 token。
    3. 通过取最小值来同步 cudagraph_mode。

    Args:
        num_tokens_unpadded: 未 padding 的 token 数量
        num_tokens_padded: padding 后的 token 数量
        should_attempt_ubatching: 是否应该尝试微批次
        cudagraph_mode: CUDA Graph 模式
        parallel_config: 并行配置

    Returns: tuple[
        should_ubatch: 所有 DP ranks 是否都使用微批次
        num_tokens_after_padding: 包含 DP padding 的每个 DP rank 的微批次 token 总数张量
        synced_cudagraph_mode: 同步后的 cudagraph 模式（跨 ranks 取最小值）
    ]
    """
    assert num_tokens_padded >= num_tokens_unpadded

    # 通过 All Reduce 在 DP ranks 之间协调
    # 以确定每个 rank 将运行的 token 总数
    # 以及是否使用微批次。
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    # 首先同步 cudagraph_mode（取最小值）
    # 这需要在 DP padding 决策之前使用，因为我们使用同步后的
    # cudagraph 模式来确定是否需要 DP padding。
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    # 检查微批次条件
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    # 当启用 cudagraph 时（跨 ranks 同步）
    # 或当激活微批次/DBO 时（微批次当前需要跨 DP ranks 的统一批次大小）
    # 需要 DP padding。
    # 使用同步后的运行时 cudagraph 模式而不是编译配置
    # 这样我们可以在步长未启用 cudagraph 时避免 padding。
    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch

    # 如果 should_dp_pad 为 True，将所有 DP ranks padding 到
    # 跨 ranks 的最大 token 数
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
    num_scheduled_tokens_per_request: np.ndarray | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """协调所有 DP ranks 以确定是否以及如何将完整批次分割成微批次。

    Args:
        num_tokens_unpadded: 未考虑 padding 的 token 数量
        allow_microbatching: 是否应该尝试微批次
        parallel_config: 并行配置
        num_tokens_padded: 包括任何非 DP padding 的 token 数量（CUDA graphs、TP 等）
        uniform_decode: 仅在 allow_microbatching 为 True 时使用。
            如果批次仅包含单个 token decode 则为 True
        num_scheduled_tokens_per_request: 仅在 allow_microbatching 为 True 时使用。
            每个请求的 token 数量
        cudagraph_mode: 此 rank 的 cudagraph 模式（0=NONE, 1=PIECEWISE, 2=FULL）。
            当跨 ranks 同步后的 cudagraph 模式不为 NONE 时启用 DP padding。

    Returns: tuple[
        ubatch_slices: 如果设置则表示所有 DP ranks 已同意进行微批次
        num_tokens_after_padding: 包含 padding 的每个 DP rank 的微批次 token 总数张量。
            当启用 cudagraph 时将 padding 到所有 DP ranks 中的最大值
        synced_cudagraph_mode: 同步后的 cudagraph 模式（跨 ranks 取最小值）
    ]
    """
    if parallel_config.data_parallel_size == 1:
        # 提前退出
        return False, None, cudagraph_mode

    # 如果调用者明确启用了微批次
    should_attempt_ubatching = False
    if allow_microbatching:
        # 检查微批次的前提条件
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
