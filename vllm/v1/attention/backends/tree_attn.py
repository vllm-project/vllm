# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""树形注意力后端模块。

本模块实现了基于 Tree Attention 的注意力后端，负责：
- 实现树形注意力后端类
- 支持预测解码中的树形结构注意力
- 使用 Triton unified attention kernel

主要类：
- TreeAttentionBackend: 树形注意力后端类
- TreeAttentionMetadata: 树形注意力元数据类
- TreeAttentionMetadataBuilder: 元数据构建器
- TreeAttentionImpl: 后端实现类
"""

import ast
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TreeAttentionBackend(AttentionBackend):
    """树形注意力后端类。

    基于 Tree Attention 实现的注意力后端，支持预测解码中的树形结构。
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的内核块大小列表。

        Returns:
            支持的块大小列表 [MultipleOf(16)]
        """
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取支持的头大小列表。

        Returns:
            支持的头大小列表
        """
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "TREE_ATTN"
        """
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            TreeAttentionImpl 类
        """
        return TreeAttentionImpl

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
            raise ValueError("块大小必须是 16 的倍数。")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            TreeAttentionMetadataBuilder 类
        """
        return TreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """检查是否使用 cascade 注意力。

        Returns:
            False（树形注意力不支持 cascade）
        """
        return False


@dataclass
class TreeAttentionMetadata:
    """树形注意力元数据类。

    存储树形注意力前向传播所需的元数据信息。

    Attributes:
        num_actual_tokens: 实际 token 数（不包括 padding）
        max_query_len: 最大 query 长度
        query_start_loc: query 起始位置
        max_seq_len: 最大序列长度
        seq_lens: 序列长度
        block_table: 块表
        slot_mapping: 槽位映射
        num_prefill_tokens: 预填充 token 数
        num_decode_tokens: 解码 token 数
        num_prefills: 预填充请求数
        num_decodes: 解码请求数
        tree_attn_bias: 树形注意力偏置
        _cached_prefill_metadata: 缓存的预填充元数据
        _cached_decode_metadata: 缓存的解码元数据
    """
    num_actual_tokens: int
    """实际 token 数（不包括 padding）。"""

    max_query_len: int
    """最大 query 长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""

    max_seq_len: int
    """最大序列长度。"""

    seq_lens: torch.Tensor
    """序列长度张量。"""

    block_table: torch.Tensor
    """块表张量。"""

    slot_mapping: torch.Tensor
    """槽位映射张量。"""

    num_prefill_tokens: int = 0
    """预填充 token 数。"""

    num_decode_tokens: int = 0
    """解码 token 数。"""

    num_prefills: int = 0
    """预填充请求数。"""

    num_decodes: int = 0
    """解码请求数。"""

    tree_attn_bias: torch.Tensor | None = None
    """树形注意力偏置张量。"""

    # 缓存的预填充/解码元数据
    _cached_prefill_metadata: "TreeAttentionMetadata | None" = None
    """缓存的预填充元数据。"""

    _cached_decode_metadata: "TreeAttentionMetadata | None" = None
    """缓存的解码元数据。"""

    @property
    def prefill_metadata(self) -> "TreeAttentionMetadata | None":
        """获取预填充元数据。

        如果批次中没有预填充请求则返回 None。
        否则返回缓存的预填充元数据或构建新的元数据。

        Returns:
            预填充元数据或 None
        """
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # 恢复缓存的预填充阶段注意力元数据结构
            return self._cached_prefill_metadata

        # 为预填充请求提取相关张量
        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # 构建并缓存预填充阶段注意力元数据结构
        self._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "TreeAttentionMetadata | None":
        """获取解码元数据。

        如果批次中没有解码请求则返回 None。
        否则返回缓存的解码元数据或构建新的元数据。

        Returns:
            解码元数据或 None
        """
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # 恢复缓存的解码阶段注意力元数据结构
            return self._cached_decode_metadata

        # 为解码请求提取相关张量
        q_start_loc = self.query_start_loc[: self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[: self.num_decodes]
        # 构建并缓存解码阶段注意力元数据结构
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc,
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            tree_attn_bias=self.tree_attn_bias,
        )
        return self._cached_decode_metadata


class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    """树形注意力元数据构建器类。

    负责构建树形注意力运行所需的元数据对象。
    支持树形注意力偏置的构建和管理。
    """
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化树形注意力元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        # 获取预测解码的树形结构配置
        spec_config = vllm_config.speculative_config
        spec_token_tree: str | None = None
        if spec := spec_config:
            spec_token_tree = spec.speculative_token_tree
        # 解析树形选择
        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree) if spec_token_tree is not None else [(0,)]
        )
        # 构建树形注意力偏置
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        # 设置重排序批次阈值
        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TreeAttentionMetadata:
        """构建树形注意力元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建

        Returns:
            构建的 TreeAttentionMetadata 对象
        """
        # 使用树形注意力偏置的大小作为解码阈值
        decode_threshold = self.tree_attn_bias.shape[0]
        # 分割解码和预填充请求
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=decode_threshold
            )
        )

        # 提取通用元数据字段
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # 构建树形注意力元数据
        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TreeAttentionMetadata:
        """为预测草稿构建元数据。

        Args:
            common_attn_metadata: 通用注意力元数据
            draft_index: 草稿索引

        Returns:
            构建的 TreeAttentionMetadata 对象
        """
        # 缓存原始树形注意力偏置
        orig_tree_attn_bias = self.tree_attn_bias

        if draft_index == 0:
            # 在根级别使用预填充进行草稿
            self.tree_attn_bias = torch.empty(0)
        else:
            # 为草稿切片树形注意力偏置，排除根级别
            start, end = 1, 1 + common_attn_metadata.max_query_len
            self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()

        # 构建注意力元数据
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # 重置树形注意力偏置为原始值
        self.tree_attn_bias = orig_tree_attn_bias
        return attn_metadata


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    """计算树形结构中每个深度的选择数量。

    Args:
        sorted_tree_choices: 排序后的树形选择列表

    Returns:
        每个深度的计数列表
    """
    # 计算树形结构中每个深度的选择数量
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    """准备树形注意力偏置张量。

    构建树形注意力掩码，其中：
    - 对角线设置为 0（每个 token 关注自身）
    - 根节点设置为 0（所有 token 关注根节点）
    - 所有祖先节点设置为 0

    Args:
        sorted_tree_choices: 排序后的树形选择列表
        depth_counts: 每个深度的计数列表
        dtype: 数据类型
        device: 设备类型

    Returns:
        树形注意力偏置张量
    """
    # +1 来自额外的根节点
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # 设置对角线为全 0，每个 token 应该关注自身
    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    # 设置根节点为全 0，所有 token 关注它
    tree_attn_mask[:, 0] = mask_val

    # 设置所有祖先节点为 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # 获取祖先位置
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


class TreeAttentionImpl(AttentionImpl):
    """树形注意力实现类。

    基于 Triton unified attention kernel 实现的树形注意力后端。
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
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        """初始化树形注意力实现。

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
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # logits_soft_cap 设置为 0 表示没有软化上限
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TreeAttentionImpl 不支持编码器自注意力和 "
                "编码器/解码器交叉注意力。"
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """执行 KV 缓存更新。

        Args:
            layer: 注意力层
            key: Key 张量
            value: Value 张量
            kv_cache: KV 缓存张量
            slot_mapping: 槽位映射
        """
        key_cache, value_cache = kv_cache.unbind(0)

        # 重新塑造输入键和值并将其存储在缓存中。
        # NOTE(woosuk): 这里 key 和 value 被填充而 slot_mapping 没有填充。
        # 但是，我们不需要做 key[:num_actual_tokens] 和 value[:num_actual_tokens]，
        # 因为 reshape_and_cache_flash op 使用 slot_mapping 的形状来确定实际 token 数。
        ops.reshape_and_cache_flash(
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
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用树形注意力进行前向传播。

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
        """
        assert output is not None, "必须提供输出张量。"

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "TreeAttentionImpl 尚未支持融合输出量化"
            )

        if attn_metadata is None:
            # 性能分析运行
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])

        # 处理预填充请求
        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # 不支持
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        # 处理解码请求
        if decode_meta := attn_metadata.decode_metadata:
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # 不支持
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
        return output
