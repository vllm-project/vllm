# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short Conv 注意力后端模块。

本模块实现了基于 Short Convolution 的注意力后端，负责：
- 实现 Short Conv 注意力后端类
- 构建 Short Conv 专用的元数据

主要类：
- ShortConvAttentionBackend: Short Conv 注意力后端类
- ShortConvAttentionMetadata: Short Conv 元数据类
- ShortConvAttentionMetadataBuilder: 元数据构建器
"""
from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class ShortConvAttentionBackend(AttentionBackend):
    """Short Conv 注意力后端类。

    基于 Short Convolution 实现的注意力后端。
    """
    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "SHORT_CONV_ATTN"
        """
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            ShortConvAttentionMetadataBuilder 类
        """
        return ShortConvAttentionMetadataBuilder


@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    """Short Conv 注意力元数据类。

    继承自 BaseMambaAttentionMetadata，用于存储 Short Conv 注意力
    前向传播所需的元数据信息。
    """
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    """Short Conv 元数据构建器类。

    负责构建 Short Conv 注意力运行所需的元数据对象。
    继承自 BaseMambaAttentionMetadataBuilder。

    Class Attributes:
        metadata_cls: 元数据类
    """
    metadata_cls = ShortConvAttentionMetadata
