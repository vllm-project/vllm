# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""微批次（Ubatching）工具函数模块。

本模块提供了微批次相关的工具函数，负责：
- 创建和管理微批次切片
- 处理注意力元数据的微批次分割
- 支持 DBO（Dual Batch Overlap）执行
- 处理 DP 填充和 CUDA Graph 兼容性

主要类：
- UBatchSlice: 微批次切片数据类

主要函数：
- maybe_create_ubatch_slices: 创建微批次切片
- split_attn_metadata: 分割注意力元数据
- slice_query_start_locs: 切片 query_start_loc
"""

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


class UBatchSlice:
    """微批次切片数据类。

    表示一个微批次的请求和 token 范围。

    Attributes:
        request_slice: 请求范围的 slice 对象
        token_slice: token 范围的 slice 对象
    """
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        """检查切片是否为空。

        Returns:
            如果切片为空返回 True
        """
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        """获取 token 数量。

        Returns:
            token 数量
        """
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    """检查最后一个微批次是否为空。

    Args:
        orig_num_tokens: 原始 token 数量
        padded_num_tokens: 填充后的 token 数量
        num_ubatches: 微批次数量

    Returns:
        最后一个微批次是否为空
    """
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    """检查是否满足微批次阈值。

    Args:
        config: 并行配置
        num_tokens: token 数量
        uniform_decode: 是否为均匀解码（只有 decode 请求）

    Returns:
        是否应该使用微批次
    """
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    """填充微批次切片到总 token 数。

    这将最后一个微批次切片扩展到总 token 数
    （num_tokens + padding），因为我们在应用 DP 填充之前
    调用 create_ubatch_slices。

    Args:
        ubatch_slices: 微批次切片列表
        num_total_tokens: 总 token 数
        num_reqs_padded: 填充后的请求数

    Returns:
        填充后的微批次切片列表
    """
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    """可能创建微批次切片。

    Args:
        should_ubatch: 是否应该使用微批次
        num_scheduled_tokens: 每个请求调度的 token 数量
        num_tokens_padded: 填充后的 token 总数
        num_reqs_padded: 填充后的请求数
        num_ubatches: 微批次数量
        split_point: 分割点（可选）

    Returns:
        (微批次切片列表，填充后的微批次切片列表)，如果不使用微批次则为 (None, None)
    """
    if not should_ubatch:
        return None, None

    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): 重构 gpu_model_runner.py 以便直接传入 cu_num_tokens
    # （即 query_start_loc）
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # 将终点添加到分割点以便迭代
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # 使用独占停止语义确定请求范围
        # 微批次包括 token 重叠 [start_token, end_token) 的请求

        # 从包含 start_token 的请求开始
        # 或者从正好在 start_token 开始的请求开始（如果在边界上）
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # 在开始于或超过 end_token 的请求处停止
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """创建与 request_slice 对应的新的 query_start_loc。

    注意：此函数创建一个新张量来保存新的 query_start_locs。
    这将破坏 cudagraph 兼容性。

    Args:
        query_start_loc: 原始 query_start_loc 张量
        request_slice: 请求范围的 slice

    Returns:
        新的 query_start_loc 张量
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """创建与 UBatchSlice 对应的新的 CommonAttentionMetadata。

    Args:
        ubatch_slice: 微批次切片
        attn_metadata: 原始注意力元数据

    Returns:
        新的 CommonAttentionMetadata

    Raises:
        AssertionError: 如果 ubatch_slice 为空
    """
    assert not ubatch_slice.is_empty(), f"微批次切片 {ubatch_slice} 为空"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token 切片开始不在第一个请求范围内"
    )
    # 注意：如果我们有 CG 填充，最后一个 token 可能在最后一个请求之外

    # 如果请求在微批次之间分割，我们必须调整元数据
    # splits_first_request: 此切片中的第一个请求是在前一个切片中开始的请求的延续
    # splits_last_request: 此切片中的最后一个请求延续到下一个切片
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc 必须至少有 2 个元素，得到 {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        # 注意：我们使用 start_locs（原始 query_start_loc_cpu）来计算
        # 跳过的 token，因为如果 splits_first_request 为 True，
        # query_start_loc_cpu 可能已被修改
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # 确保不修改 seq_lens 张量（不兼容 cudagraph）
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    max_seq_len = int(seq_lens_cpu.max())
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # 这是为了解释我们在 dummy run 中且 query_start_loc_cpu 全是 0 的情况
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """创建与每个 UBatchSlice 对应的 CommonAttentionMetadata 实例列表。

    注意：此函数不修改 common_attn_metadata

    Args:
        ubatch_slices: 微批次切片列表
        common_attn_metadata: 原始注意力元数据

    Returns:
        CommonAttentionMetadata 列表
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
