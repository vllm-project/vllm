# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM Aiter Flash Attention 后端模块。

本模块实现了基于 ROCm Aiter Flash Attention 的注意力后端，负责：
- 实现 Aiter Flash Attention 后端类
- 支持解码、预填充和扩展（extend）请求
- 支持 FP8 KV 缓存和 shuffle 布局
- 支持滑动窗口注意力
- 支持级联注意力

主要类：
- AiterFlashAttentionBackend: Aiter Flash Attention 后端类
- AiterFlashAttentionMetadata: Aiter Flash Attention 元数据类
- AiterFlashAttentionMetadataBuilder: 元数据构建器
- AiterFlashAttentionImpl: 后端实现类

辅助类：
- AiterFlashAttentionDecodeMetadata: 解码元数据
- AiterFlashAttentionPrefillMetadata: 预填充元数据
- AiterChunkSlidingWindowMetadata: 分块滑动窗口元数据
- AiterChunkContextMetadata: 分块上下文元数据
- AiterFlashAttentionChunkPrefillMetadata: 分块预填充元数据
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_prefills_and_extends,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024
if current_platform.is_rocm():
    from vllm.triton_utils import tl, triton

    def block_size(x, head_dim):
        """计算块大小。

        Args:
            x: 元素大小相关参数
            head_dim: 头维度

        Returns:
            计算得到的块大小
        """
        return min(65536 // x.element_size(), triton.next_power_of_2(head_dim))

    def num_programs(total_tokens):
        """计算程序数量。

        Args:
            total_tokens: 总 token 数

        Returns:
            计算单元数量
        """
        return min(total_tokens, num_compute_units())

    @triton.jit
    def cp_mha_gather_cache_kernel(
        key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
        value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
        key_ptr,  # [num_tokens, num_heads, head_size]
        value_ptr,  # [num_tokens, num_heads, head_size]
        block_table_ptr,  # [num_batches, max_block_num]
        cu_seqlens_kv_ptr,  # [num_batches + 1]
        token_to_batch_ptr,  # [max_cum_tokens]
        seq_start_ptr,  # [num_batches]
        k_scale_ptr,  # [1] / [num_blocks, num_kv_heads, page_size]
        v_scale_ptr,
        num_heads,
        head_size,
        x,
        max_block_num,
        DEQUANT: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        CACHE_FORMAT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel 用于从 KV 缓存中 gather 数据。

        支持 NHD 和 SHUFFLE 两种缓存布局格式。

        Args:
            key_cache_ptr: K 缓存指针
            value_cache_ptr: V 缓存指针
            key_ptr: K 输出指针
            value_ptr: V 输出指针
            block_table_ptr: 块表指针
            cu_seqlens_kv_ptr: KV 累积序列长度指针
            token_to_batch_ptr: token 到批次映射指针
            seq_start_ptr: 序列起始指针
            k_scale_ptr: K 缩放因子指针
            v_scale_ptr: V 缩放因子指针
            num_heads: 头数量
            head_size: 头大小
            x: 分块参数
            max_block_num: 最大块数
            DEQUANT: 是否反量化
            PAGE_SIZE: 页面大小
            CACHE_FORMAT: 缓存格式
            BLOCK_SIZE: 块大小
        """
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        key_ptr_offset = (
            key_ptr + token_id * head_size * num_heads + head_id * head_size
        )
        value_ptr_offset = (
            value_ptr + token_id * head_size * num_heads + head_id * head_size
        )
        batch_idx = tl.load(token_to_batch_ptr + token_id)
        batch_start = tl.load(seq_start_ptr + batch_idx)
        token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
        batch_offset = token_id - token_start + batch_start
        block_offset = batch_offset // PAGE_SIZE
        block_id = tl.load(
            block_table_ptr + max_block_num * batch_idx + block_offset
        ).to(tl.int64)
        slot_id = batch_offset % PAGE_SIZE

        if CACHE_FORMAT == "NHD":
            # KV 缓存布局为
            # K: [num_blocks, page_size, num_head, head_dim]
            # V: [num_blocks, page_size, num_head, head_dim]
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + slot_id * num_heads * head_size
                + head_id * head_size
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + slot_id * num_heads * head_size
                + head_id * head_size
            )
            k_reg = tl.load(key_cache_ptr_offset + col_offsets)
            v_reg = tl.load(value_cache_ptr_offset + col_offsets)
            if DEQUANT:
                k_scale = tl.load(k_scale_ptr)
                v_scale = tl.load(v_scale_ptr)
                k_dtype = k_reg.dtype
                v_dtype = v_reg.dtype
                k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
                v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
            tl.store(key_ptr_offset + col_offsets, k_reg)
            tl.store(value_ptr_offset + col_offsets, v_reg)

        elif CACHE_FORMAT == "SHUFFLE":
            # KV 缓存布局为
            # K: [num_blocks, num_head, head_dim // x, page_size, x]
            # V: [num_blocks, num_head, page_size // x, head_dim, x]
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + head_id * head_size * PAGE_SIZE
                + slot_id * x
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + head_id * head_size * PAGE_SIZE
                + (slot_id // x) * head_size * x
                + slot_id % x
            )
            k_reg_offset = col_offsets // x * PAGE_SIZE * x + col_offsets % x
            v_reg_offset = col_offsets * x
            k_reg = tl.load(key_cache_ptr_offset + k_reg_offset)
            v_reg = tl.load(value_cache_ptr_offset + v_reg_offset)
            if DEQUANT:
                k_scale = 1.0
                v_scale = 1.0
                k_reg = k_reg.to(tl.float32) * k_scale
                v_reg = v_reg.to(tl.float32) * v_scale
            tl.store(key_ptr_offset + col_offsets, k_reg)
            tl.store(value_ptr_offset + col_offsets, v_reg)

    def cp_mha_gather_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        token_to_batch: torch.Tensor,
        seq_starts: torch.Tensor,
        dequant: bool,
        kv_cache_layout: str,
        total_tokens: int,
    ):
        """从 KV 缓存中 gather 数据的封装函数。

        Args:
            key_cache: K 缓存
            value_cache: V 缓存
            key: K 输出
            value: V 输出
            block_tables: 块表
            k_scales: K 缩放因子
            v_scales: V 缩放因子
            cu_seqlens_kv: KV 累积序列长度
            token_to_batch: token 到批次映射
            seq_starts: 序列起始
            dequant: 是否反量化
            kv_cache_layout: KV 缓存布局
            total_tokens: 总 token 数
        """
        assert kv_cache_layout in ["NHD", "SHUFFLE"], (
            "kv_cache_layout 仅支持 NHD, SHUFFLE"
        )
        head_dim = key.shape[2]
        x = 16 // key_cache.element_size()
        # 目前仅支持带反量化的 gather cache
        assert dequant is True, "目前仅支持带反量化的 gather cache"
        # 对于 K 缓存布局：[num_blocks, num_heads, page_size, head_dim]
        assert head_dim == key_cache.shape[3], (
            "我们假设你的 KV 缓存布局是 [num_blocks, "
            "page_size, num_heads, head_dim]，但实际不是"
        )
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]

        grid = lambda meta: (total_tokens, num_heads)
        cp_mha_gather_cache_kernel[grid](
            key_cache,
            value_cache,
            key,
            value,
            block_tables,
            cu_seqlens_kv,
            token_to_batch,
            seq_starts,
            k_scales,
            v_scales,
            num_heads,
            head_dim,
            x,
            block_tables.size(1),
            DEQUANT=dequant,
            PAGE_SIZE=page_size,
            CACHE_FORMAT=kv_cache_layout,
            BLOCK_SIZE=head_dim,
        )

    @triton.jit
    def reshape_and_cache_shuffle_kernel(
        key_ptr,  # [num_tokens, num_kv_heads, head_size]
        value_ptr,  # [num_tokens, num_kv_heads, head_size]
        key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
        slot_mapping_ptr,  # [num_tokens]
        k_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
        v_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
        x,
        k_stride0,
        v_stride0,
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        QUANT: tl.constexpr,
        IS_FNUZ: tl.constexpr,
    ):
        """Triton kernel 用于 shuffle 布局的 KV 缓存更新。

        Args:
            key_ptr: K 输入指针
            value_ptr: V 输入指针
            key_cache_ptr: K 缓存指针
            value_cache_ptr: V 缓存指针
            slot_mapping_ptr: 槽位映射指针
            k_scale_ptr: K 缩放因子指针
            v_scale_ptr: V 缩放因子指针
            x: 分块参数
            k_stride0: K 步幅
            v_stride0: V 步幅
            block_size: 块大小
            head_size: 头大小
            num_kv_heads: KV 头数量
            BLOCK_SIZE: 块大小
            QUANT: 是否量化
            IS_FNUZ: 是否 FNUZ 格式
        """
        tid = tl.program_id(0)
        head_id = tl.program_id(1)
        offset = tl.arange(0, BLOCK_SIZE)
        src_offset_k = tid * k_stride0 + head_id * head_size
        src_offset_v = tid * v_stride0 + head_id * head_size
        slot_id = tl.load(slot_mapping_ptr + tid)
        if slot_id < 0:
            return
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        dst_offset = (
            block_id * num_kv_heads * head_size * block_size
            + head_id * head_size * block_size
        )
        dst_k_shuffle_offset = (
            dst_offset + offset // x * block_size * x + block_offset * x + offset % x
        )
        dst_v_shuffle_offset = (
            dst_offset
            + block_offset // x * head_size * x
            + offset * x
            + block_offset % x
        )
        k_val = tl.load(key_ptr + src_offset_k + offset)
        v_val = tl.load(value_ptr + src_offset_v + offset)
        if QUANT:
            k_scale = 1.0
            v_scale = 1.0
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
        tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
        tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)

    def reshape_and_cache_shuffle_triton(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ):
        """Shuffle 布局的 KV 缓存更新封装函数。

        Args:
            key: K 输入
            value: V 输入
            key_cache: K 缓存
            value_cache: V 缓存
            slot_mapping: 槽位映射
            kv_cache_dtype: KV 缓存数据类型
            k_scales: K 缩放因子
            v_scales: V 缩放因子
        """
        num_tokens = slot_mapping.shape[0]
        _, num_kv_heads, head_size = key.shape
        num_blocks, block_size, _, _ = key_cache.shape
        x = 16 // key_cache.element_size()
        k_cache_template = torch.empty(
            [num_blocks, num_kv_heads, head_size // x, block_size, x],
            dtype=key_cache.dtype,
            device="meta",
        )
        v_cache_template = torch.empty(
            [num_blocks, num_kv_heads, block_size // x, head_size, x],
            dtype=value_cache.dtype,
            device="meta",
        )
        new_key_cache = key_cache.view_as(k_cache_template)
        new_value_cache = value_cache.view_as(v_cache_template)
        QUANT = False
        if kv_cache_dtype.startswith("fp8"):
            QUANT = True
        grid = (
            num_tokens,
            num_kv_heads,
        )
        reshape_and_cache_shuffle_kernel[grid](
            key,
            value,
            new_key_cache,
            new_value_cache,
            slot_mapping,
            k_scales,
            v_scales,
            x,
            key.stride(0),
            value.stride(0),
            block_size,
            head_size,
            num_kv_heads,
            BLOCK_SIZE=head_size,
            QUANT=QUANT,
            IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        )


logger = init_logger(__name__)


@dataclass
class AiterFlashAttentionDecodeMetadata:
    """Aiter Flash Attention 解码元数据。

    Attributes:
        max_query_len: 最大 query 长度
        min_query_len: 最小 query 长度
        max_seq_len: 最大序列长度
        query_start_loc: query 起始位置
    """
    max_query_len: int
    """最大 query 长度。"""

    min_query_len: int
    """最小 query 长度。"""

    max_seq_len: int
    """最大序列长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""


@dataclass
class AiterFlashAttentionPrefillMetadata:
    """Aiter Flash Attention 预填充元数据。

    Attributes:
        max_query_len: 最大 query 长度
        min_query_len: 最小 query 长度
        max_seq_len: 最大序列长度
        query_start_loc: query 起始位置
    """
    max_query_len: int
    """最大 query 长度。"""

    min_query_len: int
    """最小 query 长度。"""

    max_seq_len: int
    """最大序列长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""


@dataclass
class AiterChunkSlidingWindowMetadata:
    """Aiter 分块滑动窗口元数据。

    Attributes:
        swa_seqlens: 滑动窗口序列长度
        swa_cu_seqlens: 滑动窗口累积序列长度
        swa_seq_starts: 滑动窗口序列起始
        swa_token_to_batch: 滑动窗口 token 到批次映射
        swa_max_seqlens: 滑动窗口最大序列长度
        swa_total_tokens: 滑动窗口总 token 数
        swa_workspace: 滑动窗口工作空间
    """
    swa_seqlens: torch.Tensor
    """滑动窗口序列长度张量。"""

    swa_cu_seqlens: torch.Tensor
    """滑动窗口累积序列长度张量。"""

    swa_seq_starts: torch.Tensor
    """滑动窗口序列起始张量。"""

    swa_token_to_batch: torch.Tensor
    """滑动窗口 token 到批次映射张量。"""

    swa_max_seqlens: int
    """滑动窗口最大序列长度。"""

    swa_total_tokens: int
    """滑动窗口总 token 数。"""

    swa_workspace: torch.Tensor
    """滑动窗口工作空间张量。"""


@dataclass
class AiterChunkContextMetadata:
    """Aiter 分块上下文元数据。

    Attributes:
        workspace: 工作空间
        cu_seq_lens_chunk: 分块累积序列长度
        chunk_starts: 块起始
        token_to_batch: token 到批次映射
        seq_tot: 每个块的总序列长度
        max_seq_lens: 每个块的最大序列长度
        seq_lens: 序列长度
        num_chunks: 块数量
        total_token_per_batch: 每个批次的总 token 数
        swa_metadata: 滑动窗口元数据
    """
    workspace: torch.Tensor
    """工作空间张量。"""

    cu_seq_lens_chunk: torch.Tensor
    """分块累积序列长度张量。"""

    chunk_starts: torch.Tensor
    """块起始张量。"""

    token_to_batch: torch.Tensor
    """token 到批次映射张量。"""

    seq_tot: list[int]
    """每个块的总序列长度列表。"""

    max_seq_lens: list[int]
    """每个块的最大序列长度列表。"""

    seq_lens: torch.Tensor
    """序列长度张量。"""

    num_chunks: int
    """块数量。"""

    total_token_per_batch: list[int]
    """每个批次的总 token 数列表。"""

    swa_metadata: AiterChunkSlidingWindowMetadata | None
    """滑动窗口元数据。"""


@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    """Aiter Flash Attention 分块预填充元数据。

    Attributes:
        max_query_len: 最大 query 长度
        min_query_len: 最小 query 长度
        max_seq_len: 最大序列长度
        query_start_loc: query 起始位置
        chunk_context_metadata: 分块上下文元数据
    """
    max_query_len: int
    """最大 query 长度。"""

    min_query_len: int
    """最小 query 长度。"""

    max_seq_len: int
    """最大序列长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""

    chunk_context_metadata: AiterChunkContextMetadata
    """分块上下文元数据。"""


@dataclass
class AiterFlashAttentionMetadata:
    """Aiter Flash Attention 元数据类。

    存储 Aiter Flash Attention 前向传播所需的元数据信息。

    Attributes:
        num_actual_tokens: 实际 token 数（不包括 padding）
        num_actual_kv_tokens: 实际 KV token 数
        max_query_len: 最大 query 长度
        query_start_loc: query 起始位置
        max_seq_len: 最大序列长度
        seq_lens: 序列长度
        slot_mapping: 槽位映射
        block_table: 块表
        num_decodes: 解码请求数
        num_decode_tokens: 解码 token 数
        num_prefills: 预填充请求数
        num_prefill_tokens: 预填充 token 数
        num_extends: 扩展请求数
        num_extend_tokens: 扩展 token 数
        decode_metadata: 解码元数据
        prefill_metadata: 预填充元数据
        extend_metadata: 扩展元数据
        use_cascade: 是否使用级联注意力
        common_prefix_len: 公共前缀长度
        total_tokens: 总 token 数
        k_scale: K 缩放因子
        v_scale: V 缩放因子
    """
    # NOTE(sang): context_len、query_len 和 seq_len 的定义：
    # |---------- N-1 次迭代 --------|
    # |---------------- N 次迭代 ---------------------|
    # |- tokenA -|......................|-- 新 token ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int
    """实际 token 数（不包括 padding）。"""

    num_actual_kv_tokens: int
    """实际 KV token 数。"""

    max_query_len: int
    """最大 query 长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""

    max_seq_len: int
    """最大序列长度。"""

    seq_lens: torch.Tensor
    """序列长度张量。"""

    slot_mapping: torch.Tensor
    """槽位映射张量。"""

    block_table: torch.Tensor
    """块表张量。"""

    # 预填充和解码分割
    num_decodes: int
    """解码请求数。"""

    num_decode_tokens: int
    """解码 token 数。"""

    num_prefills: int
    """预填充请求数。"""

    num_prefill_tokens: int
    """预填充 token 数。"""

    num_extends: int
    """扩展请求数。"""

    num_extend_tokens: int
    """扩展 token 数。"""

    decode_metadata: AiterFlashAttentionDecodeMetadata | None
    """解码元数据。"""

    prefill_metadata: AiterFlashAttentionPrefillMetadata | None
    """预填充元数据。"""

    extend_metadata: AiterFlashAttentionChunkPrefillMetadata | None
    """扩展元数据。"""

    # 用于级联注意力
    use_cascade: bool
    """是否使用级联注意力。"""

    common_prefix_len: int
    """公共前缀长度。"""

    total_tokens: int
    """总 token 数。"""

    # 仅用于启用了 shuffle_kv_cache 的 fp8 shuffle 布局 KV 缓存
    # 我们为每一层分配 kv_scale，因为我们未来可能会为 KV 缓存集成每 token 量化
    k_scale: dict[str, torch.Tensor] | None
    """K 缩放因子的字典。"""

    v_scale: dict[str, torch.Tensor] | None
    """V 缩放因子的字典。"""


class AiterFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AiterFlashAttentionMetadata]
):
    """Aiter Flash Attention 元数据构建器类。

    负责构建 Aiter Flash Attention 运行所需的元数据对象。
    支持解码、预填充和扩展请求的元数据构建。
    """
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Aiter Flash Attention 元数据构建器。

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
        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        self.total_tokens: int = 0
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        sliding_window_configs: set[tuple[int, int] | None] = set()
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for name, layer in layers.items():
            if name not in layer_names:
                continue
            assert isinstance(layer.impl, AiterFlashAttentionImpl), (
                "Aiter Flash Attention Metadata Builder can only be used "
                "with Aiter Flash Attention Impl."
            )
            sliding_window_configs.add(layer.impl.sliding_window)

        while len(sliding_window_configs) > 0:
            sliding_window_config = sliding_window_configs.pop()
            if sliding_window_config is not None and sliding_window_config[0] != -1:
                assert self.aot_sliding_window is None, (
                    "Aiter Flash ATTENTION can only support one valid sliding window!"
                )
                self.aot_sliding_window = sliding_window_config

        self.extend_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.headdim],
            dtype=self.model_config.dtype,
            device=device,
        )
        self.scale = torch.tensor([1.0], dtype=torch.float, device=self.device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """为 CUDA 图捕获构建元数据。

        Args:
            common_attn_metadata: 通用注意力元数据

        Returns:
            构建的 AiterFlashAttentionMetadata 对象
        """
        self.total_tokens = (
            self.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        )
        res = self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.total_tokens = 0
        return res

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "AiterFlashAttentionMetadata":
        """构建 Aiter Flash Attention 元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建

        Returns:
            构建的 AiterFlashAttentionMetadata 对象
        """
        assert self.reorder_batch_threshold is not None
        split_ret = split_decodes_prefills_and_extends(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )
        # Allocate scales for fp8 shuffle kv cache with shuffle_kv_cache enabled
        if (
            rocm_aiter_ops.is_shuffle_kv_cache_enabled()
            and self.scale.numel() == 1
            and self.vllm_config.cache_config.cache_dtype.startswith("fp8")
        ):
            layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            first_layer_name = [k for k in layers][0]
            kv_cache_shape = (
                self.vllm_config.compilation_config.static_forward_context[
                    first_layer_name
                ]
                .kv_cache[0]
                .shape
            )
            num_blocks = kv_cache_shape[1]
            self.scale = torch.ones(
                [num_blocks, self.num_heads_kv, self.block_size],
                dtype=torch.float32,
                device=self.device,
            )
        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            num_prefill_tokens,
        ) = split_ret

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        seq_lens = common_attn_metadata.seq_lens.cpu()

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=query_lens_cpu[:num_decodes].max().item(),
                min_query_len=query_lens_cpu[:num_decodes].min().item(),
                max_seq_len=seq_lens[:num_decodes].max().item(),
                query_start_loc=common_attn_metadata.query_start_loc[: num_decodes + 1],
            )

        prefill_metadata = None
        if num_prefills > 0:
            query_lens_for_prefill = query_lens_cpu[num_decodes + num_extends :]
            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes + num_extends :
            ]
            prefill_metadata = AiterFlashAttentionPrefillMetadata(
                max_query_len=query_lens_for_prefill.max().item(),
                min_query_len=query_lens_for_prefill.min().item(),
                max_seq_len=seq_lens[num_decodes + num_extends :].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
            )

        extend_metadata = None
        if num_extends > 0:
            num_extends_slice = slice(num_decodes, num_decodes + num_extends)
            query_lens_for_extend = query_lens_cpu[num_extends_slice]
            seq_lens_for_extend = seq_lens[num_extends_slice]
            computed_kv_lens = seq_lens_for_extend - query_lens_for_extend
            swa_metadata = None
            if self.aot_sliding_window is not None:
                swa_seqlen_for_extend = torch.minimum(
                    seq_lens_for_extend,
                    query_lens_for_extend + self.aot_sliding_window[0] + 1,
                )
                cu_seq_lens = torch.zeros(
                    num_extends + 1,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                torch.cumsum(
                    swa_seqlen_for_extend,
                    dim=0,
                    dtype=cu_seq_lens.dtype,
                    out=cu_seq_lens[1:],
                )
                token_to_seq = torch.arange(
                    0,
                    num_extends,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                token_to_seq = torch.repeat_interleave(
                    token_to_seq, swa_seqlen_for_extend
                )
                fetched_shape = cu_seq_lens[-1].item()
                # TODO(ganyi): Maybe reuse these 2 buffer from extend_workspace
                swa_workspace = torch.empty(
                    (2, fetched_shape, self.num_heads_kv, self.headdim),
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )

                seq_starts = seq_lens_for_extend - swa_seqlen_for_extend
                max_seqlen_k = swa_seqlen_for_extend.max().item()
                total_tokens = cu_seq_lens[-1].item()

                swa_metadata = AiterChunkSlidingWindowMetadata(
                    swa_seqlens=swa_seqlen_for_extend.to(
                        self.device, non_blocking=True
                    ),
                    swa_cu_seqlens=cu_seq_lens.to(self.device, non_blocking=True),
                    swa_seq_starts=seq_starts.to(self.device, non_blocking=True),
                    swa_token_to_batch=token_to_seq.to(self.device, non_blocking=True),
                    swa_max_seqlens=max_seqlen_k,
                    swa_total_tokens=total_tokens,
                    swa_workspace=swa_workspace,
                )

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = (
                torch.arange(num_chunks, dtype=torch.int32)
                .unsqueeze(1)
                .expand(-1, num_extends)
                * max_context_chunk
            )
            chunk_ends = torch.min(
                computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
            )
            chunk_seq_lens = (chunk_ends - chunk_starts).clamp(
                min=0
            )  # [num_chunks, num_extends]
            cu_seq_lens_cpu = torch.zeros(
                [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
            )
            torch.cumsum(
                chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
            )
            max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()

            range_idx = torch.arange(max_cum_tokens, dtype=torch.int32)[None, None, :]
            idx_to_batch_tensor = range_idx == cu_seq_lens_cpu[:, 1:][:, :, None]
            idx_to_batch_tensor = idx_to_batch_tensor.sum(
                dim=1
            )  # [num_chunks, max_cum_tokens]
            token_to_batch_tensor = torch.cumsum(idx_to_batch_tensor, dim=1)

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.extend_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                chunk_starts=chunk_starts.to(self.device, non_blocking=True),
                seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                seq_lens=chunk_seq_lens,
                token_to_batch=token_to_batch_tensor.to(self.device, non_blocking=True),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
                swa_metadata=swa_metadata,
            )

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes : num_decodes + num_extends + 1
            ]
            seq_lens_device = common_attn_metadata.seq_lens[num_extends_slice]
            cu_seq_lens = torch.zeros(
                num_extends + 1, dtype=torch.int32, device=seq_lens_device.device
            )
            torch.cumsum(
                seq_lens_device, dim=0, dtype=cu_seq_lens.dtype, out=cu_seq_lens[1:]
            )
            extend_metadata = AiterFlashAttentionChunkPrefillMetadata(
                max_query_len=query_lens_for_extend.max().item(),
                min_query_len=query_lens_for_extend.min().item(),
                max_seq_len=seq_lens[num_extends_slice].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata,
            )

        num_actual_kv_tokens = torch.sum(seq_lens).item()

        use_cascade = common_prefix_len > 0

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            extend_metadata=extend_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
            k_scale=self.scale,
            v_scale=self.scale,
        )
        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> AiterFlashAttentionMetadata:
        """为预测草稿构建元数据。

        为 EAGLE 预测草稿构建注意力元数据，无需 CPU-GPU 同步。
        所有请求都是统一解码，因此可以跳过 split_decodes_prefills_and_extends()
        和所有会破坏 CUDA 图捕获的.cpu()/.item() 调用。

        Args:
            common_attn_metadata: 通用注意力元数据
            draft_index: 草稿索引

        Returns:
            构建的 AiterFlashAttentionMetadata 对象
        """
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        decode_metadata = AiterFlashAttentionDecodeMetadata(
            max_query_len=common_attn_metadata.max_query_len,
            min_query_len=common_attn_metadata.max_query_len,  # uniform batch
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
        )

        return AiterFlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            num_actual_kv_tokens=0,  # not used in unified_attention path
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_reqs,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            num_extends=0,
            num_extend_tokens=0,
            decode_metadata=decode_metadata,
            prefill_metadata=None,
            extend_metadata=None,
            use_cascade=False,
            common_prefix_len=0,
            total_tokens=self.total_tokens,
            k_scale=self.scale,
            v_scale=self.scale,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        """检查是否使用级联注意力。

        Returns:
            False（不支持级联注意力）
        """
        return False


class AiterFlashAttentionBackend(AttentionBackend):
    """Aiter Flash Attention 后端类。

    基于 ROCm Aiter Flash Attention 实现的注意力后端。
    支持解码、预填充和扩展请求，支持 FP8 KV 缓存和 shuffle 布局。
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """检查是否支持指定的注意力类型。

        ROCM AITER FA 支持解码器注意力和编码器 - 解码器（交叉）注意力。

        Args:
            attn_type: 注意力类型

        Returns:
            是否支持
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的内核块大小列表。

        Returns:
            支持的块大小列表 [16, 32]
        """
        return [16, 32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取支持的头大小列表。

        Returns:
            支持的头大小列表 [64, 128, 256]
        """
        return [64, 128, 256]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "FLASH_ATTN"
        """
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            AiterFlashAttentionImpl 类
        """
        return AiterFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            AiterFlashAttentionMetadataBuilder 类
        """
        return AiterFlashAttentionMetadataBuilder

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

        Raises:
            ValueError: 如果块大小不是 16 的倍数
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持指定的计算能力。

        Args:
            capability: 设备计算能力

        Returns:
            是否支持（仅支持 MI3XX 系列）
        """
        from vllm.platforms.rocm import on_mi3xx

        # DeviceCapability is currently created using torch.cuda.get_device_capability()
        # which is known to be buggy on rocm systems. on_mi3xx uses amd-smi which is
        # more reliable.
        return on_mi3xx()


class AiterFlashAttentionImpl(AttentionImpl):
    """Aiter Flash Attention 实现类。

    基于 ROCm Aiter Flash Attention 实现的注意力后端。
    支持解码、预填充和扩展请求，支持滑动窗口和 FP8 KV 缓存。
    """
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
        kv_sharing_target_layer_name: int | None = None,
    ) -> None:
        """初始化 Aiter Flash Attention 实现。

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
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0.0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type not in [AttentionType.DECODER, AttentionType.ENCODER_DECODER]:
            raise NotImplementedError(
                "Encoder self-attention is not implemented for FlashAttentionImpl"
            )

    def extend_for_sliding_window(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: float,
        v_scale: float,
    ):
        """为滑动窗口执行扩展注意力计算。

        Args:
            attn_metadata: 注意力元数据
            query: Query 张量
            key_cache: Key 缓存
            value_cache: Value 缓存
            output: 输出张量
            cu_seqlens_q: Query 累积序列长度
            max_seqlen_q: 最大 Query 序列长度
            block_table: 块表
            k_scale: K 缩放因子
            v_scale: V 缩放因子
        """
        assert attn_metadata.extend_metadata is not None
        assert attn_metadata.extend_metadata.chunk_context_metadata is not None
        chunked_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        swa_metadata = chunked_metadata.swa_metadata
        assert swa_metadata is not None
        swa_cu_seqlens = swa_metadata.swa_cu_seqlens
        swa_seq_starts = swa_metadata.swa_seq_starts
        swa_token_to_batch = swa_metadata.swa_token_to_batch
        swa_max_seqlens = swa_metadata.swa_max_seqlens
        swa_total_tokens = swa_metadata.swa_total_tokens
        key_fetched, value_fetched = (
            swa_metadata.swa_workspace[0],
            swa_metadata.swa_workspace[1],
        )
        cp_mha_gather_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key_fetched,
            value=value_fetched,
            block_tables=block_table,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=swa_cu_seqlens,
            token_to_batch=swa_token_to_batch,
            seq_starts=swa_seq_starts,
            dequant=self.kv_cache_dtype.startswith("fp8"),
            kv_cache_layout="NHD",
            total_tokens=swa_total_tokens,
        )

        rocm_aiter_ops.flash_attn_varlen_func(
            q=query,
            k=key_fetched,
            v=value_fetched,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=swa_cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=swa_max_seqlens,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=False,
            out=output,
        )

    def extend_forward(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ):
        """执行扩展前向传播。

        处理扩展请求的注意力计算，支持滑动窗口和分块上下文。

        Args:
            attn_metadata: 注意力元数据
            query: Query 张量
            key: Key 张量
            value: Value 张量
            key_cache: Key 缓存
            value_cache: Value 缓存
            output: 输出张量
            cu_seqlens_q: Query 累积序列长度
            max_seqlen_q: 最大 Query 序列长度
            max_seqlen_k: 最大 Key 序列长度
            min_seqlen_q: 最小 Query 序列长度
            block_table: 块表
            slot_mapping: 槽位映射
            k_scale: K 缩放因子
            v_scale: V 缩放因子
        """
        if self.sliding_window[0] != -1:
            self.extend_for_sliding_window(
                attn_metadata,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                block_table,
                k_scale,
                v_scale,
            )
            return
        out, lse = rocm_aiter_ops.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=True,
        )
        assert attn_metadata.extend_metadata is not None
        chunk_context_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        chunked_output = None
        chunked_lse = None
        for chunk_idx in range(num_chunks):
            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key_fetched,
                value=value_fetched,
                block_tables=block_table,
                k_scales=k_scale,
                v_scales=v_scale,
                cu_seqlens_kv=cu_seqlens_kv[chunk_idx],
                token_to_batch=token_to_batch[chunk_idx],
                seq_starts=chunk_starts[chunk_idx],
                dequant=self.kv_cache_dtype.startswith("fp8"),
                kv_cache_layout="SHUFFLE"
                if rocm_aiter_ops.is_shuffle_kv_cache_enabled()
                else "NHD",
                total_tokens=total_token_per_batch[chunk_idx],
            )

            suf_out, suf_lse = rocm_aiter_ops.flash_attn_varlen_func(
                q=query,
                k=key_fetched,
                v=value_fetched,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlens[chunk_idx],
                min_seqlen_q=min_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                return_lse=True,
            )
            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(
                    output=tmp_output,
                    output_lse=tmp_lse,
                    prefix_output=chunked_output,
                    prefix_lse=chunked_lse,
                    suffix_output=suf_out,
                    suffix_lse=suf_lse,
                )
                chunked_output = tmp_output
                chunked_lse = tmp_lse

        merge_attn_states(
            output=output,
            prefix_output=chunked_output,
            prefix_lse=chunked_lse,
            suffix_output=out,
            suffix_lse=lse,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用 AiterFlashAttention 进行前向传播。

        Args:
            layer: 注意力层
            query: 形状 = [num_tokens, num_heads, head_size]
            key: 形状 = [num_tokens, num_kv_heads, head_size]
            value: 形状 = [num_tokens, num_kv_heads, head_size]
            kv_cache: 形状 = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: 注意力元数据
            output: 输出张量
            output_scale: 输出缩放因子
            output_block_scale: 输出块缩放因子

        Returns:
            形状 = [num_tokens, num_heads * head_size] 的输出张量

        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is
        # executed in eager-mode PyTorch. Thus, we need to be careful
        # about any CPU overhead in this method. For example, `view`
        # and `slice` (or `[:n]`) operations are surprisingly slow even
        # in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # decode:extend:prefill
        query = query[:num_actual_tokens]
        if key is not None:
            key = key[:num_actual_tokens]
        if value is not None:
            value = value[:num_actual_tokens]

        output_actual_tokens = output[:num_actual_tokens]

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_extends = attn_metadata.num_extends

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens
        if not attn_metadata.use_cascade:
            # calculate for pure prefills
            if num_prefills > 0:
                assert attn_metadata.prefill_metadata is not None

                prefill_query = query[num_decode_tokens + num_extend_tokens :]
                prefill_key = key[num_decode_tokens + num_extend_tokens :]
                prefill_value = value[num_decode_tokens + num_extend_tokens :]

                rocm_aiter_ops.flash_attn_varlen_func(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    cu_seqlens_q=attn_metadata.prefill_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.prefill_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.prefill_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_metadata.max_seq_len,
                    min_seqlen_q=1,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    out=output_actual_tokens[num_decode_tokens + num_extend_tokens :],
                )

            # calculate for extends
            if num_extends > 0:
                assert attn_metadata.extend_metadata is not None
                extend_tokens_slice = slice(
                    num_decode_tokens, num_decode_tokens + num_extend_tokens
                )
                extend_queries = query[extend_tokens_slice]
                extend_keys = key[extend_tokens_slice]
                extend_values = value[extend_tokens_slice]
                extend_outputs = output[extend_tokens_slice]
                k_scale = layer._k_scale
                v_scale = layer._v_scale
                if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
                    k_scale = attn_metadata.k_scale
                    v_scale = attn_metadata.v_scale
                self.extend_forward(
                    attn_metadata=attn_metadata,
                    query=extend_queries,
                    key=extend_keys,
                    value=extend_values,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=extend_outputs,
                    cu_seqlens_q=attn_metadata.extend_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.extend_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.extend_metadata.max_seq_len,
                    min_seqlen_q=1,
                    block_table=attn_metadata.block_table[
                        num_decodes : num_decodes + num_extends
                    ],
                    slot_mapping=attn_metadata.slot_mapping[
                        num_decodes : num_decodes + num_extends
                    ],
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            # calculate for decodes
            if num_decodes > 0:
                assert attn_metadata.decode_metadata is not None
                decode_max_query_len = attn_metadata.decode_metadata.max_query_len

                # Use unified_attention for speculative decoding (multi-token)
                if decode_max_query_len > 1:
                    assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
                        "Shuffle KV cache layout is not supported with "
                        "speculative decoding (multi-token decode)."
                    )
                    from aiter.ops.triton.unified_attention import (
                        unified_attention,
                    )

                    descale_shape = (
                        attn_metadata.query_start_loc[:num_decodes].shape[0] - 1,
                        key_cache.shape[2],
                    )
                    unified_attention(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=attn_metadata.query_start_loc[:num_decodes],
                        max_seqlen_q=decode_max_query_len,
                        seqused_k=attn_metadata.seq_lens[:num_decodes],
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=attn_metadata.block_table[:num_decodes],
                        softcap=self.logits_soft_cap,
                        q_descale=None,
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
                    return

                # The ll4mi kernel in paged_attention_v1 requires
                # HEAD_SIZE >= 16 * NWARPS (= 64 on ROCm with NWARPS=4).
                # For smaller head sizes or sliding window attention,
                # fall back to the unified_attention triton kernel which
                # handles both correctly.
                _MIN_HEAD_SIZE_FOR_LL4MI = 64
                use_unified_attention = self.head_size < _MIN_HEAD_SIZE_FOR_LL4MI

                if use_unified_attention:
                    assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
                        "unified_attention fallback with shuffle layout "
                        "is not supported yet."
                    )
                    from aiter.ops.triton.unified_attention import (
                        unified_attention,
                    )

                    decode_cu_seqlens_q = attn_metadata.query_start_loc[
                        : num_decodes + 1
                    ]
                    descale_shape = (
                        num_decodes,
                        key_cache.shape[2],
                    )
                    unified_attention(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=decode_cu_seqlens_q,
                        max_seqlen_q=1,
                        seqused_k=attn_metadata.seq_lens[:num_decodes],
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=attn_metadata.block_table[:num_decodes],
                        softcap=self.logits_soft_cap,
                        q_descale=None,
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
                elif rocm_aiter_ops.is_shuffle_kv_cache_enabled():
                    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
                    x = 16 // key_cache.element_size()
                    k_cache_template = torch.empty(
                        [num_blocks, num_kv_heads, head_size // x, block_size, x],
                        dtype=key_cache.dtype,
                        device="meta",
                    )
                    v_cache_template = torch.empty(
                        [num_blocks, num_kv_heads, block_size // x, head_size, x],
                        dtype=value_cache.dtype,
                        device="meta",
                    )
                    new_key_cache = key_cache.view_as(k_cache_template)
                    new_value_cache = value_cache.view_as(v_cache_template)
                    rocm_aiter_ops.pa_fwd_asm(
                        Q=query[:num_decode_tokens],
                        K=new_key_cache,
                        V=new_value_cache,
                        block_tables=attn_metadata.block_table[:num_decodes],
                        context_lens=attn_metadata.seq_lens[:num_decodes],
                        block_tables_stride0=attn_metadata.block_table[
                            :num_decodes
                        ].stride(0),
                        K_QScale=attn_metadata.k_scale,
                        V_QScale=attn_metadata.v_scale,
                        out_=output[:num_decode_tokens],
                    )
                else:
                    _, num_heads, head_size = query.shape
                    nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
                    num_seqs = attn_metadata.seq_lens.shape[0]
                    max_num_partitions = (
                        attn_metadata.max_seq_len + _PARTITION_SIZE_ROCM - 1
                    ) // _PARTITION_SIZE_ROCM

                    workspace_buffer = torch.empty(
                        (num_seqs * num_heads * max_num_partitions * head_size)
                        * nbytes_per_qo_elem
                        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
                        dtype=torch.uint8,
                        device=output.device,
                    )

                    # import so that aiter register the op to the namespace of
                    # torch.ops.aiter
                    import aiter  # noqa: F401

                    torch.ops.aiter.paged_attention_v1(
                        output[:num_decode_tokens],
                        workspace_buffer,
                        query[:num_decode_tokens],
                        key_cache,
                        value_cache,
                        self.scale,
                        attn_metadata.block_table[:num_decodes],
                        attn_metadata.query_start_loc[:num_decodes],
                        attn_metadata.seq_lens[:num_decodes],
                        attn_metadata.max_seq_len,
                        self.alibi_slopes,
                        self.kv_cache_dtype,
                        "NHD",
                        self.logits_soft_cap,
                        layer._k_scale,
                        layer._v_scale,
                        None,
                        _PARTITION_SIZE_ROCM,
                        1,
                        self.sliding_window[0] + 1,
                    )
        else:
            raise NotImplementedError(
                "Cascade attention is not implemented for ROCM AITER"
            )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """执行 KV 缓存更新。

        Args:
            layer: 注意力层
            key: Key 张量
            value: Value 张量
            kv_cache: KV 缓存张量
            slot_mapping: 槽位映射
        """
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping
        # is not padded. However, we don't need to do
        # key[:num_actual_tokens] and value[:num_actual_tokens] because
        # the reshape_and_cache_flash op uses the slot_mapping's shape
        # to determine the number of actual tokens.
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            # We may calculate per token quant scale in
            # reshape_and_cache_shuffle_triton which might differ from
            # vllm's style when shuffle layout is used.
            k_scale = layer._k_scale
            v_scale = layer._v_scale
            assert k_scale is not None and v_scale is not None, (
                "k_scale and v_scale are required for shuffled update"
            )
            reshape_and_cache_shuffle_triton(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
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

    def fused_rope_kvcache_supported(self):
        """检查是否支持融合 RoPE KV 缓存。

        仅在不使用 shuffle KV 缓存布局时支持融合；
        shuffle 布局使用不同的缓存更新路径。

        Returns:
            是否支持
        """
        # Only support fusion when shuffle KV cache layout is not used;
        # shuffle layout uses a different cache update path.
        return (
            rocm_aiter_ops.is_enabled()
            and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """执行 RoPE 和 KV 缓存更新。

        Args:
            layer: 注意力层
            query: Query 张量
            key: Key 张量
            value: Value 张量
            positions: 位置
            cos_sin_cache: RoPE cos/sin 缓存
            is_neox: 是否 NeoX 格式
            kv_cache: KV 缓存张量
            layer_slot_mapping: 层槽位映射
        """
        key_cache, value_cache = kv_cache.unbind(0)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
