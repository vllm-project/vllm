# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba1 注意力后端模块。

本模块实现了基于 Mamba1 的注意力后端，负责：
- 实现 Mamba1 注意力后端类
- 构建 Mamba1 专用的元数据

主要类：
- Mamba1AttentionBackend: Mamba1 注意力后端类
- Mamba1AttentionMetadata: Mamba1 元数据类
- Mamba1AttentionMetadataBuilder: 元数据构建器
"""

from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class Mamba1AttentionBackend(AttentionBackend):
    """Mamba1 注意力后端类。

    基于 Mamba1 实现的注意力后端。
    """
    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "MAMBA1_ATTN"
        """
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            Mamba1AttentionMetadataBuilder 类
        """
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    """Mamba1 注意力元数据类。

    继承自 BaseMambaAttentionMetadata，用于存储 Mamba1 注意力
    前向传播所需的元数据信息。
    """
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    """Mamba1 元数据构建器类。

    负责构建 Mamba1 注意力运行所需的元数据对象。
    继承自 BaseMambaAttentionMetadataBuilder。

    Class Attributes:
        metadata_cls: 元数据类
    """
    metadata_cls = Mamba1AttentionMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        """构建 Mamba1 注意力元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建
            **kwargs: 其他参数

        Returns:
            构建的 Mamba1AttentionMetadata 对象
        """
        # 计算通用元数据
        common = self._compute_common_metadata(common_attn_metadata)

        # 如果存在预填充请求且启用了前缀缓存（all 模式），
        # 需要构建分块元数据张量
        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            # 构建分块元数据张量
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size,
                    common,
                    common_attn_metadata,
                )
            )
            # 返回添加了分块元数据的对象
            return replace(
                common,
                cu_chunk_seqlen_p=cu_chunk_seqlen_p,
                last_chunk_indices_p=last_chunk_indices_p,
            )

        # 否则直接返回通用元数据
        return common
