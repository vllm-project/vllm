# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention 后端模块。

本模块实现了基于 PyTorch FlexAttention 的注意力后端，负责：
- 实现 FlexAttention 后端类
- 支持块级注意力掩码
- 支持滑动窗口注意力
- 支持 Prefix LM 注意力
- 支持物理到逻辑块索引映射

主要类：
- FlexAttentionBackend: Flex Attention 后端类
- FlexAttentionMetadata: Flex Attention 元数据类
- FlexAttentionMetadataBuilder: 元数据构建器
- FlexAttentionImpl: 后端实现类

主要函数：
- physical_to_logical_mapping: 物理块到逻辑块的逆映射
- unique_static_unsorted: 去重并保持顺序
- causal_mask_mod: 因果掩码
- get_kernel_options: 获取内核选项
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import torch
import torch._dynamo.decorators
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    is_quantized_kv_cache,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

torch._dynamo.config.recompile_limit = 16
create_block_mask_compiled = torch.compile(
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _offsets_to_doc_ids_tensor(offsets: torch.Tensor) -> torch.Tensor:
    """从累积偏移量生成文档 ID 张量。

    Args:
        offsets: 累积偏移量张量

    Returns:
        文档 ID 张量
    """
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int):
    """将张量填充到指定倍数的长度。

    Args:
        x: 输入张量
        multiple: 目标倍数
        dim: 填充维度

    Returns:
        填充后的张量
    """
    difference = (multiple - (x.shape[dim] % multiple)) % multiple
    if difference == 0:
        return x

    dim = dim if dim >= 0 else x.ndim + dim
    pad_list = []

    for i in range(x.ndim - 1, dim - 1, -1):
        if i == dim:
            pad_list.extend([0, difference])
        else:
            pad_list.extend([0, 0])

    return F.pad(x, pad_list, mode="constant", value=0)


class FlexAttentionBackend(AttentionBackend):
    """FlexAttention 后端类。

    基于 PyTorch FlexAttention 实现的注意力后端。
    支持因果掩码、滑动窗口、Prefix LM 等多种注意力模式。

    Class Attributes:
        accept_output_buffer: 是否接受输出缓冲区
        supported_dtypes: 支持的数据类型
        supported_kv_cache_dtypes: 支持的 KV 缓存数据类型
        forward_includes_kv_cache_update: 前向传播是否包含 KV 缓存更新
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "FLEX_ATTENTION"
        """
        return "FLEX_ATTENTION"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlexAttention 支持解码器和仅编码器注意力。

        Args:
            attn_type: 注意力类型

        Returns:
            是否支持
        """
        return attn_type in (AttentionType.DECODER, AttentionType.ENCODER_ONLY)

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """FlexAttention 支持多模态 prefix 的完整注意力。

        Returns:
            True
        """
        return True

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            FlexAttentionImpl 类
        """
        return FlexAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """获取 KV 缓存形状。

        Args:
            num_blocks: 块数量
            block_size: 块大小
            num_kv_heads: KV 头数量
            head_size: 头大小
            cache_dtype_str: 缓存数据类型

        Returns:
            KV 缓存形状元组
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            FlexAttentionMetadataBuilder 类
        """
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """判断是否使用 cascade 注意力。

        Returns:
            False（不支持 cascade 注意力）
        """
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取支持的头大小列表。

        Returns:
            空列表（动态支持）
        """
        return []


# @torch.compile(fullgraph=True, mode="reduce-overhead")
def physical_to_logical_mapping(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> torch.Tensor:
    """创建从物理块位置到逻辑索引的逆映射。

    原始 block_table 将逻辑块映射到物理位置：

    逻辑到物理（原始 block_table）:
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Logical Blocks:  0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Physical Blocks: 3  5  1  7  4  2  0  6   │
    └───────────────────────────────────────────┘

    此函数创建逆映射：

    物理到逻辑（逆映射）:
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Physical Blocks: 0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Logical Blocks:  6  2  5  0  4  1  7  3   │
    └───────────────────────────────────────────┘

    如果多个逻辑块映射到同一个物理块，此函数返回最新的（最大）逻辑块索引。

    如果物理块没有被任何逻辑块映射，其在结果中的值将为 -1。

    重要：垃圾值保护
    ────────────────────────────────────
    block_table 张量可能在未使用的位置包含垃圾值
    （超出实际序列长度）。例如，如果一个序列只需要 3 个块，
    但表有 8 个块的空间：

        block_table[0] = [10, 25, 7, 999, 1234, 888, ...]
                                    ^^^^^^^^^^^^^^^^^^^^
                                    垃圾值

    这些垃圾值会导致问题，因为：
    1. 它们可能偶然映射到有效的物理块
    2. scatter_ 操作将为它们分配逻辑索引
    3. 后续的注意力计算将错误地访问这些块

    为防止这种情况，我们使用 seq_lens 和 block_size 来屏蔽未使用的
    条目，确保只处理有效的块引用。

    重要：复用的物理块（滑动窗口/混合注意力）
    ────────────────────────────────────────────────────────────────────
    对于某些注意力类型，物理缓存块可以随时间复用。
    这可能导致相同的物理块 ID 在 block_table 的某一行中
    在不同的逻辑块索引处多次出现。在这种情况下，只有最新的
    逻辑块索引对应于该物理块的当前内容。因此，逆映射必须
    为每个物理块 ID 选择最大的逻辑块索引。

    Args:
        block_table: 形状为 [max_reqs, max_num_blocks] 的张量，
            将逻辑块映射到物理位置。可能在未使用的位置包含垃圾值。
        seq_lens: 每个请求的序列长度张量。用于确定每个序列实际需要多少块。
        block_size: 每个块的 token 大小。与 seq_lens 一起用于计算每个序列的有效块数。
        total_blocks: 可用的物理块总数

    Returns:
        形状为 [max_reqs, total_blocks] 的张量，其中每个条目
        physical_to_logical[req_id, physical_block] 包含该物理块的逻辑块索引，
        如果未使用则为 -1。
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    physical_to_logical = torch.full(
        (max_reqs, total_blocks), -1, dtype=torch.long, device=device
    )

    # 只处理有效块以避免垃圾值
    num_blocks_per_seq: torch.Tensor = cdiv(seq_lens, block_size)
    mask = (
        torch.arange(max_num_blocks, device=device)[None, :]
        < num_blocks_per_seq[:, None]
    )

    valid_block_table = torch.where(mask, block_table, 0)
    valid_logical_indices = torch.where(
        mask, torch.arange(max_num_blocks, device=device)[None, :], 0
    )

    physical_to_logical.scatter_reduce_(
        -1, valid_block_table.to(torch.int64), valid_logical_indices, reduce="amax"
    )
    # 注意：块 0 似乎总是空的，所以我们手动重置它
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def unique_static_unsorted(
    x: torch.Tensor,
    *,
    M: int,  # 最大正值（0 表示”跳过”）
    dim: int = -1,  # 去重轴
    ignored_val: int = 0,  # 要忽略的值
    pad_val: int = -1,  # 未使用槽位的标记
) -> torch.Tensor:
    “””
    去重并保持顺序，然后左打包唯一值，其余用 pad_val 填充。

    - 保持每个非零值的第一次出现，同时保持顺序
    - 返回 (packed, keep_mask)，形状与 x 相同
    - 要求所有值在 [0, M] 范围内
    - 跳过 ignored_val

    在 CPU 或 GPU 上工作，无 Python 循环，O(B·N) 时间 / O(B·M) 内存。

    示例：
    x =[3, 1, 0, 1, 2], M=3, ignored_val=0 => [3, 1, 2, -1, -1]

    Args:
        x: 输入张量
        M: 最大正值
        dim: 去重轴
        ignored_val: 要忽略的值
        pad_val: 未使用槽位的标记

    Returns:
        去重并打包后的张量
    “””
    if not (-1 <= pad_val <= M):
        raise ValueError("`pad_val` must lie in [-1, M]")

    # ── move `dim` to the end so we can treat tensor as [B, N] ──────────
    dim = dim % x.ndim
    x_perm = x.movedim(dim, -1)  # shape [..., N]
    B, N = x_perm.numel() // x_perm.shape[-1], x_perm.shape[-1]
    x_flat = x_perm.reshape(B, N)  # [B, N]

    device = x.device
    idx = torch.arange(N, device=device).expand(B, N)  # per-row indices

    # ── build first-occurrence table for every v ∈ [0, M] ───────────────
    first_idx = torch.full((B, M + 1), N, device=device)  # “∞”
    # scatter_reduce_: first_idx[b, v] = min(first_idx[b, v], i) for each i
    first_idx.scatter_reduce_(1, x_flat, idx, reduce="amin")

    # ── keep mask: first occurrence *and* value ≠ 0 ─────────────────────
    keep = (x_flat != ignored_val) & (idx == first_idx.gather(1, x_flat))  # [B, N]

    # ── left-pack uniques into a fresh tensor ───────────────────────────
    dest_pos = torch.cumsum(keep.to(torch.long), dim=1) - 1  # where to go
    packed_flat = torch.full_like(x_flat, pad_val)

    rows, src_cols = torch.nonzero(keep, as_tuple=True)
    packed_flat[rows, dest_pos[rows, src_cols]] = x_flat[rows, src_cols]

    # ── restore original layout ─────────────────────────────────────────
    packed = packed_flat.reshape(x_perm.shape).movedim(-1, dim)
    return packed


def causal_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    """因果掩码函数。

    用于 FlexAttention 的因果掩码，确保每个位置只能注意到之前的位置。

    Args:
        b: 批次索引
        h: 注意力头索引
        q_idx: Query 位置索引
        kv_idx: Key/Value 位置索引

    Returns:
        因果掩码布尔值：q_idx >= kv_idx 时为 True
    """
    return q_idx >= kv_idx


@dataclass
class FlexAttentionMetadata:
    """FlexAttention 元数据类。

    存储 FlexAttention 前向传播所需的所有元数据。
    """
    causal: bool  # 是否使用因果掩码
    num_actual_tokens: int  # 实际 token 数量（不包括 padding）
    max_query_len: int  # 最大 query 长度
    query_start_loc: torch.Tensor  # 每个 query 的起始位置累积和
    max_seq_len: int  # 最大序列长度
    seq_lens: torch.Tensor  # 每个序列的实际长度
    block_table: torch.Tensor  # 块表，将逻辑块映射到物理块
    slot_mapping: torch.Tensor  # slot 映射

    use_cascade: bool  # 是否使用 cascade 注意力（暂未实现）
    common_prefix_len: int  # 公共前缀长度（暂未实现）
    cu_prefix_query_lens: torch.Tensor | None  # 前缀 query 长度累积和（暂未实现）
    prefix_kv_lens: torch.Tensor | None  # 前缀 KV 长度（暂未实现）
    suffix_kv_lens: torch.Tensor | None  # 后缀 KV 长度（暂未实现）

    # 块相关信息
    total_cache_tokens: int  # 缓存中的总 token 数
    block_size: int  # 块大小
    max_possible_sequence_length: int  # 最大可能的序列长度
    num_reqs: int  # 请求数量
    physical_to_logical: torch.Tensor  # 物理块到逻辑块的映射
    decode_offset: torch.Tensor  # 解码偏移量
    num_blocks_per_seq: torch.Tensor  # 每个序列的块数

    # 用于日志记录
    num_input_tokens: int = 0  # 输入 token 数量（包括 padding）

    # Flex 元数据
    num_blocks: int = 0  # 块数量
    block_mask: BlockMask | None = None  # 块掩码
    score_mod: _score_mod_signature | None = None  # 分数修改函数
    logical_mask_mod: _mask_mod_signature = causal_mask_mod  # 逻辑掩码函数
    doc_ids: torch.Tensor | None = None  # 文档 ID
    direct_build: bool = True  # 是否直接构建
    q_block_size: int = 16  # Query 块大小
    kv_block_size: int = 16  # KV 块大小
    transformed_score_mod: _score_mod_signature | None = None  # 转换后的分数修改函数
    sliding_window: int | None = None  # 滑动窗口大小
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None  # 多模态前缀范围

    @cached_property
    def logical_block_ids(self):
        """获取逻辑块 ID 序列。

        Returns:
            从 0 到最大块数的逻辑块 ID 张量
        """
        return torch.arange(
            cdiv(self.max_seq_len, self.block_size),
            device=self.block_table.device,
            dtype=torch.long,
        )

    def _convert_physical_to_logical(
        self,
        request_lookup: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将物理索引转换为逻辑索引（包括 query 和 kv）。

        注意：is_within_lower_bound 检查序列是否从块边界开始。

        Args:
            request_lookup: 请求查找表
            q_idx: Query 索引
            physical_kv_idx: 物理 KV 索引

        Returns:
            (is_valid, logical_q_idx, logical_kv_idx) 元组
            - is_valid: KV 索引是否有效
            - logical_q_idx: 逻辑 query 索引
            - logical_kv_idx: 逻辑 KV 索引
        """
        # 将 query 索引映射到对应的请求索引
        q_req = request_lookup[q_idx]

        # 将物理 KV 索引转换为逻辑索引
        physical_kv_block = physical_kv_idx // self.block_size
        physical_kv_offset = physical_kv_idx % self.block_size
        logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
        logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

        # 确定有效的 kv 索引
        live_block = logical_block_idx >= 0
        within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
        within_lower_bound = logical_kv_idx >= 0
        is_valid = live_block & within_upper_bound & within_lower_bound

        # 将物理 query 索引转换为逻辑索引
        local_q_idx = q_idx - self.query_start_loc[q_req]
        logical_q_idx = local_q_idx + self.decode_offset[q_req]

        return is_valid, logical_q_idx, logical_kv_idx

    def get_causal_mask_mod(self) -> _mask_mod_signature:
        """为 FlexAttention 创建因果掩码函数。

        此函数创建组合掩码函数，处理：
        1. Paged attention 块映射
        2. 从打包的 query 序列到逻辑 query 条目的映射

        默认情况下，它还会向 query 索引添加解码偏移量。
        有了这些信息，我们可以创建传递给掩码函数的"逻辑"索引，
        使掩码函数对 query 和 key/value 张量的布局不可知。

        Returns:
            最终掩码函数
        """
        assert self.doc_ids is not None

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            # 仅对有效索引应用掩码修改
            return torch.where(
                is_valid,
                self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def get_bidirectional_mask_mod(self) -> _mask_mod_signature:
        """为 FlexAttention 创建编码器双向掩码函数。

        由于编码器双向注意力不使用 KV 缓存运行，
        此函数基于打包的 query 序列创建掩码。

        Returns:
            双向掩码函数
        """
        # 创建从 query 索引到请求编号的查找映射
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return request_lookup[q_idx] == request_lookup[kv_idx]

        return final_mask_mod

    def get_sliding_window_mask_mod(self) -> _mask_mod_signature:
        """为 FlexAttention 创建滑动窗口掩码函数。

        注意：这里的滑动窗口掩码是双向的，我们需要将其与
        编码器/解码器的双向/因果掩码结合使用。

        Returns:
            滑动窗口掩码函数
        """

        if self.sliding_window is None:
            raise ValueError("sliding_window must be set for sliding window attention")

        def sliding_window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return torch.abs(q_idx - kv_idx) < self.sliding_window

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return torch.where(
                is_valid,
                sliding_window_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod if self.causal else sliding_window_mask_mod

    def get_prefix_lm_mask_mod(self) -> _mask_mod_signature:
        """为 FlexAttention 创建 Prefix LM 掩码函数。

        Returns:
            Prefix LM 掩码函数
        """

        assert self.doc_ids is not None
        request_lookup = self.doc_ids

        def prefix_lm_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            cu_q_idx: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ):
            mask = torch.zeros_like(q_idx, dtype=torch.bool)
            for req, doc_range_lst in (self.mm_prefix_range or {}).items():
                req_mask = request_lookup[cu_q_idx] == req
                for start, end in doc_range_lst:
                    doc_mask_q = (q_idx >= start) & (q_idx <= end)
                    doc_mask_kv = (kv_idx >= start) & (kv_idx <= end)
                    mask = mask | (req_mask & doc_mask_q & doc_mask_kv)
            return mask

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx)
            )
            return torch.where(
                is_valid,
                prefix_lm_mask_mod(b, h, q_idx, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def get_mask_mod(self):
        """获取组合掩码函数。

        分阶段构建掩码：
        - Stage-1: 初始化基础掩码（解码器用因果掩码，编码器用双向掩码）
        - Stage-2: 为特殊注意力类型添加额外的掩码

        Returns:
            组合后的掩码函数
        """
        # Stage-1: 初始化基础掩码函数
        # （解码器使用因果掩码，编码器使用双向掩码）
        if self.causal:
            mask_mod = self.get_causal_mask_mod()
        else:
            mask_mod = self.get_bidirectional_mask_mod()
        # Stage-2: 在前向运行期间为特殊注意力添加额外的 mask_mod，
        # 创建组合的 mask_mod
        if self.sliding_window is not None:
            # 为滑动窗口注意力添加滑动窗口掩码
            sliding_window_mask_mod = self.get_sliding_window_mask_mod()
            mask_mod = and_masks(mask_mod, sliding_window_mask_mod)
        if self.mm_prefix_range:
            # 为视觉 - 语言 Prefix LM 注意力添加 Prefix LM 掩码
            prefix_lm_mask_mod = self.get_prefix_lm_mask_mod()
            mask_mod = or_masks(mask_mod, prefix_lm_mask_mod)
        return mask_mod

    def get_transformed_score_mod(self) -> _score_mod_signature | None:
        """为 FlexAttention 创建转换后的分数修改函数。

        此函数包装用户的 score_mod 以处理物理到逻辑索引的转换，
        类似于 get_mask_mod 处理掩码函数的方式。

        Returns:
            转换后的分数修改函数，如果未设置则返回 None
        """
        if self.score_mod is None:
            return None

        # 创建从 query 索引到请求编号的查找映射
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)
        user_score_mod = self.score_mod

        def transformed_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx, logical_kv_idx) = (
                self._convert_physical_to_logical(
                    request_lookup, q_idx, physical_kv_idx
                )
            )

            return torch.where(
                is_valid,
                user_score_mod(
                    score, b, h, logical_q_idx, logical_kv_idx, physical_q=q_idx
                ),
                -float("inf"),
            )

        return transformed_score_mod

    def _build_block_mask_direct(self) -> BlockMask:
        """直接构建块掩码，用于标准因果注意力。

        此方法使用 BlockMask.from_kv_blocks 直接构建块掩码，
        比通用的 create_block_mask 方法更高效。

        直接路径的工作原理如下：
        1. 对于每个 query token，使用 max_seq_len 从 block_table 获取块，
           如果需要，排除滑动窗口之外的块
           （这会为较短的序列获取比需要更多的块）
        2. 将 query token 分组为 q_block_size 大小的块
        3. 对于每组，使用 unique_static_unsorted 去重
        4. 使用去重后的块索引创建 BlockMask

        当一组 q_block_size token 包含多个序列 ID（doc_ids）时，
        会发生高估。在这种情况下，我们为组中的每个序列获取所有块，
        即使单个 query token 可能只需要这些块的子集（基于因果掩码
        和它们的位置）。

        Returns:
            构建的块掩码
        """
        page_to_block_ratio = self.kv_block_size // self.block_size
        if page_to_block_ratio != 1:
            raise ValueError(
                f"FlexAttention currently requires the cache block size "
                f"({self.block_size}) to be equal to the kv_block_size "
                f"({self.kv_block_size}). Please check your model's "
                f"configuration."
            )

        used_pages = self.block_table[
            self.doc_ids, : cdiv(self.max_seq_len, self.block_size)
        ]

        if self.sliding_window and self.causal:
            device = used_pages.device
            assert self.doc_ids is not None
            token_indices = torch.arange(
                self.doc_ids.shape[0], device=device, dtype=torch.long
            )
            logical_q_idx = (
                token_indices
                - self.query_start_loc[self.doc_ids]
                + self.decode_offset[self.doc_ids]
            )
            min_kv_idx = torch.clamp(logical_q_idx - (self.sliding_window - 1), min=0)
            min_block_idx = min_kv_idx // self.block_size
            sliding_mask = self.logical_block_ids >= min_block_idx[:, None]
            used_pages.masked_fill_(~sliding_mask, 0)

        used_pages_padded = pad_to_multiple(
            used_pages, multiple=self.q_block_size, dim=0
        )
        used_pages_padded = used_pages_padded.reshape(
            used_pages_padded.shape[0] // self.q_block_size, -1
        )
        used_pages_padded = used_pages_padded // page_to_block_ratio
        kv_indices = unique_static_unsorted(
            (used_pages_padded.long()), M=self.num_blocks
        ).to(torch.int32)

        kv_num_blocks = (kv_indices >= 0).sum(dim=-1).to(torch.int32)
        block_mask_kwargs = {
            "seq_lengths": (self.num_actual_tokens, self.total_cache_tokens),
            "kv_num_blocks": kv_num_blocks[None, None],
            "kv_indices": kv_indices[None, None],
            "full_kv_num_blocks": None,
            "full_kv_indices": None,
            "BLOCK_SIZE": (self.q_block_size, self.kv_block_size),
            "mask_mod": self.mask_mod,
        }

        # compute_q_blocks 参数在 PyTorch 2.9+ 中可用
        if is_torch_equal_or_newer("2.9.0.dev0"):
            block_mask_kwargs["compute_q_blocks"] = False
        return BlockMask.from_kv_blocks(**block_mask_kwargs)

    def build_block_mask(self) -> BlockMask:
        """构建块掩码。

        Returns:
            构建的块掩码
        """
        mask_mod = self.get_mask_mod()
        kv_len = self.total_cache_tokens if self.causal else self.num_actual_tokens
        return create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )

    def __post_init__(self):
        """初始化后处理。

        设置文档 ID、块数量、掩码函数等。
        """
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        # 创建从 query 索引到请求编号的查找映射
        self.doc_ids = _offsets_to_doc_ids_tensor(self.query_start_loc)
        self.num_blocks = self.total_cache_tokens // self.block_size

        self.mask_mod = self.get_mask_mod()
        self.transformed_score_mod = self.get_transformed_score_mod()

        if self.direct_build and self.causal:
            self.block_mask = self._build_block_mask_direct()
        else:
            self.block_mask = self.build_block_mask()


class FlexAttentionMetadataBuilder(AttentionMetadataBuilder[FlexAttentionMetadata]):
    """FlexAttention 元数据构建器类。

    负责构建 FlexAttention 运行所需的元数据对象。
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        supports_small_blocks = is_torch_equal_or_newer("2.9.0.dev0")
        self.direct_build: bool = supports_small_blocks
        self.q_block_size: int = 16 if supports_small_blocks else 128
        self.kv_block_size: int = self.block_size if supports_small_blocks else 128

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        """构建 FlexAttention 元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建

        Returns:
            构建的 FlexAttentionMetadata 对象
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks_per_seq = cdiv(seq_lens, self.block_size)

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        block_size = self.kv_cache_spec.block_size
        max_possible_seq_len = self.model_config.max_model_len
        num_gpu_blocks = self.cache_config.num_gpu_blocks

        assert num_gpu_blocks is not None, (
            "FlexAttention requires num_gpu_blocks to be set"
        )
        total_cache_tokens = num_gpu_blocks * block_size

        inverse_block_table = physical_to_logical_mapping(
            block_table_tensor, seq_lens, block_size, num_gpu_blocks
        )

        offset_tensor = common_attn_metadata.compute_num_computed_tokens()

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            num_blocks_per_seq=num_blocks_per_seq,
            # FIXME(Isotr0py): direct build 在构建编码器模型的 bidirectional
            # 注意力块掩码时有问题，暂时禁用。
            # 参见：https://github.com/vllm-project/vllm/pull/27329#issuecomment-3431484053
            direct_build=(self.direct_build and common_attn_metadata.causal),
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
        )
        return out

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        """判断是否使用 cascade 注意力。

        Returns:
            False（FlexAttention 不支持 cascade 注意力）
        """
        return False


class FlexAttentionImpl(AttentionImpl):
    """FlexAttention 实现类。

    基于 PyTorch FlexAttention 的注意力实现。
    """
    sliding_window: int | None  # 滑动窗口大小
    alibi_slopes: torch.Tensor | None  # ALIBI 斜率
    logits_soft_cap: float | None  # Logits 软化上限
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None  # 多模态前缀范围

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        """初始化 FlexAttention 实现。

        Args:
            num_heads: 注意力头数量
            head_size: 头大小
            scale: 缩放因子
            num_kv_heads: KV 头数量
            alibi_slopes: ALIBI 斜率列表
            sliding_window: 滑动窗口大小
            kv_cache_dtype: KV 缓存数据类型
            logits_soft_cap: Logits 软化上限
            attn_type: 注意力类型
            kv_sharing_target_layer_name: KV 共享目标层名称
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
            raise NotImplementedError(
                f"FlexAttention does not support {attn_type} attention"
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlexAttention does not support alibi slopes yet."
            )
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window

        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "FlexAttention does not support logits soft cap yet."
            )

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FlexAttention does not support kv sharing yet.")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet"
            )

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """将 3D 张量视为 4D。

        Args:
            tensor: 输入张量

        Returns:
            4D 张量
        """
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """更新 KV 缓存。

        Args:
            layer: 注意力层
            key: Key 张量
            value: Value 张量
            kv_cache: KV 缓存
            slot_mapping: Slot 映射
        """
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return

        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用 FlexAttention 进行前向传播。

        Args:
            layer: 注意力层
            query: 形状 = [num_tokens, num_heads, head_size]
            key: 形状 = [num_tokens, num_kv_heads, head_size]
            value: 形状 = [num_tokens, num_kv_heads, head_size]
            kv_cache: 形状 = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: 注意力元数据
            output: 输出张量
            output_scale: 输出缩放因子（不支持）
            output_block_scale: 输出块缩放因子（不支持）

        Returns:
            形状 = [num_tokens, num_heads * head_size] 的输出张量
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlexAttentionImpl"
            )

        enable_gqa = self.num_kv_heads != self.num_heads

        if attn_metadata is None:
            # 性能分析运行
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        needs_rebuild_block_mask = False
        if attn_metadata.sliding_window != self.sliding_window:
            attn_metadata.sliding_window = self.sliding_window
            if attn_metadata.direct_build:
                # 更新注意力元数据中的掩码函数
                attn_metadata.mask_mod = attn_metadata.get_mask_mod()
            needs_rebuild_block_mask = True

        if self.mm_prefix_range != getattr(attn_metadata, "mm_prefix_range", None):
            self.mm_prefix_range = attn_metadata.mm_prefix_range
            attn_metadata.mask_mod = attn_metadata.get_mask_mod()
            needs_rebuild_block_mask = True

        if needs_rebuild_block_mask:
            if attn_metadata.direct_build and attn_metadata.causal:
                attn_metadata.block_mask = attn_metadata._build_block_mask_direct()
            else:
                attn_metadata.block_mask = attn_metadata.build_block_mask()

        if not attn_metadata.causal:
            assert self.attn_type == AttentionType.ENCODER_ONLY

            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key, value),
            )

            query = query[:, :, :num_actual_tokens, :]
            if (key_tensor.size(-2) > num_actual_tokens) or (
                value_tensor.size(-2) > num_actual_tokens
            ):
                # 在仅编码器模型中使用 torch.compile 时，
                # qkv 可能会被填充，这可能导致异常。
                # 参见：https://github.com/vllm-project/vllm/pull/24872#discussion_r2353252290
                key_tensor = key_tensor[:, :, :num_actual_tokens, :]
                value_tensor = value_tensor[:, :, :num_actual_tokens, :]

        else:
            assert self.attn_type == AttentionType.DECODER
            key_cache, value_cache = kv_cache.unbind(0)

            # 去除 block_size 维度
            key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key_cache, value_cache),
            )

            query = query[:, :, :num_actual_tokens, :]

        # 暂时不起作用 -> 违反约束
        # torch._dynamo.try_mark_dynamic(query, 2)

        assert attn_metadata.block_mask is not None
        block_m, block_n = attn_metadata.block_mask.BLOCK_SIZE

        kernel_options = get_kernel_options(
            query, block_m, block_n, attn_metadata.direct_build
        )
        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

        # Flex 目前没有 out 变体，依赖于结尾融合
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output[:num_actual_tokens, :, :].copy_(out)
        return output


def get_kernel_options(
    query, block_m, block_n, use_direct_build: bool
) -> dict[str, int | bool]:
    """获取 FlexAttention 内核选项。

    根据硬件配置和输入张量特性确定最优的块大小和其他内核选项。

    Args:
        query: Query 张量
        block_m: Query 块大小
        block_n: KV 块大小
        use_direct_build: 是否使用直接构建

    Returns:
        内核选项字典
    """
    kernel_options: dict[str, int | bool] = {
        "FORCE_USE_FLEX_ATTENTION": True,
    }

    def ensure_divisible(candidate: int, block_size: int) -> int:
        """选择一个能整除逻辑块的核块大小。

        Args:
            candidate: 候选块大小
            block_size: 逻辑块大小

        Returns:
            合适的块大小
        """
        if block_size <= 0:
            return candidate
        candidate = min(candidate, block_size)
        if candidate <= 0:
            return block_size
        if block_size % candidate == 0:
            return candidate

        candidate = math.gcd(candidate, block_size)
        if candidate <= 1:
            return block_size
        return candidate

    if vllm_is_batch_invariant():
        kernel_options["BLOCK_M"] = 16
        kernel_options["BLOCK_N"] = 16
        kernel_options["IS_DIVISIBLE"] = False
        return kernel_options
    if use_direct_build:
        kernel_options["BLOCK_M"] = block_m
        kernel_options["BLOCK_N"] = block_n
        return kernel_options
    else:
        preferred_block = 32 if query.dtype == torch.float32 else 64
        block_lower_bound = 16

        block_m_candidate = ensure_divisible(preferred_block, block_m)
        block_n_candidate = ensure_divisible(preferred_block, block_n)

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties()
            # ROCm 没有暴露 shared_memory_per_block_optin 属性
            # AMD GPU 每个工作组通常有 64KB LDS（本地数据共享）
            if hasattr(device_props, "shared_memory_per_block_optin"):
                max_shared_memory = device_props.shared_memory_per_block_optin
            elif current_platform.is_rocm():
                # ROCm 回退：使用 64KB
                max_shared_memory = 65536
            else:
                raise RuntimeError(
                    "Unable to determine shared memory size on this hardware."
                )

            if max_shared_memory < 144 * 1024:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)
        block_n_candidate = max(block_n_candidate, block_lower_bound)

        kernel_options["BLOCK_M"] = block_m_candidate
        kernel_options["BLOCK_N"] = block_n_candidate

    return kernel_options
