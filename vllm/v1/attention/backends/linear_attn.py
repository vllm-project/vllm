# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear Attention 注意力后端模块。

本模块实现了基于 Linear Attention 的注意力后端，负责：
- 实现 Linear Attention 后端类
- 支持 Mamba 状态空间模型

主要类：
- LinearAttentionBackend: Linear Attention 后端类
- LinearAttentionMetadata: Linear Attention 元数据类
- LinearAttentionMetadataBuilder: 元数据构建器
"""
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class LinearAttentionBackend(AttentionBackend):
    """Linear Attention 注意力后端类。

    基于 Linear Attention 实现的注意力后端。
    """
    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "LINEAR_ATTN"
        """
        return "LINEAR_ATTN"

    @staticmethod
    def get_builder_cls() -> type["LinearAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            LinearAttentionMetadataBuilder 类
        """
        return LinearAttentionMetadataBuilder


@dataclass
class LinearAttentionMetadata:
    """Linear Attention 元数据类。

    存储 Linear Attention 前向传播所需的元数据信息。

    Attributes:
        num_prefills: 预填充请求数
        num_prefill_tokens: 预填充 token 数
        num_decodes: 解码请求数
        num_decode_tokens: 解码 token 数
        query_start_loc: query 起始位置
        seq_lens: 序列长度
        state_indices_tensor: 状态索引张量
    """
    num_prefills: int
    """预填充请求数。"""

    num_prefill_tokens: int
    """预填充 token 数。"""

    num_decodes: int
    """解码请求数。"""

    num_decode_tokens: int
    """解码 token 数。"""

    query_start_loc: torch.Tensor
    """Query 起始位置张量。"""

    seq_lens: torch.Tensor
    """序列长度张量。"""

    state_indices_tensor: torch.Tensor
    """状态索引张量，形状：[batch,]"""


class LinearAttentionMetadataBuilder(AttentionMetadataBuilder[LinearAttentionMetadata]):
    """Linear Attention 元数据构建器类。

    负责构建 Linear Attention 运行所需的元数据对象。

    Class Attributes:
        reorder_batch_threshold: 重排序批次阈值
        _cudagraph_support: CUDA 图支持级别
    """
    reorder_batch_threshold: int = 1

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Linear Attention 元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LinearAttentionMetadata:
        """构建 Linear Attention 元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建

        Returns:
            构建的 LinearAttentionMetadata 对象
        """
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        # 获取状态索引张量
        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]

        # 分割解码和预填充请求
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        # 构建并返回注意力元数据
        attn_metadata = LinearAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata
