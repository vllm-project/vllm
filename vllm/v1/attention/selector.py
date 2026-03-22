# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力后端选择器模块。

本模块实现了注意力后端的选择和懒加载机制，负责：
- 根据配置和硬件特性选择合适的注意力后端
- 使用缓存机制避免重复选择
- 支持 Mamba 状态空间模型的后端选择
- 验证 KV 缓存数据类型和布局

主要类：
- AttentionSelectorConfig: 注意力选择器配置数据类

主要函数：
- get_attn_backend: 根据参数选择注意力后端
- get_mamba_attn_backend: 选择 Mamba 注意力后端
"""

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    """注意力选择器配置数据类。

    封装选择注意力后端所需的所有配置参数。

    Attributes:
        head_size: 注意力头大小
        dtype: 数据类型
        kv_cache_dtype: KV 缓存数据类型
        block_size: 块大小（可选）
        use_mla: 是否使用 MLA（多头潜在注意力）
        has_sink: 是否有 sink
        use_sparse: 是否使用稀疏注意力
        use_mm_prefix: 是否使用多模态前缀
        use_per_head_quant_scales: 是否使用每头量化缩放因子
        attn_type: 注意力类型（默认为 DECODER）
    """

    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER

    def __repr__(self):
        """返回配置的字符串表示。"""
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type})"
        )


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """根据参数选择注意力后端并懒加载。

    此函数根据提供的配置参数选择合适的注意力后端实现。
    它会验证 KV 缓存数据类型的有效性，并使用缓存机制
    避免重复选择相同的后端。

    Args:
        head_size: 注意力头大小
        dtype: 数据类型
        kv_cache_dtype: KV 缓存数据类型字符串
        use_mla: 是否使用 MLA
        has_sink: 是否有 sink
        use_sparse: 是否使用稀疏注意力
        use_mm_prefix: 是否使用多模态前缀
        use_per_head_quant_scales: 是否使用每头量化缩放因子
        attn_type: 注意力类型（可选）
        num_heads: 注意力头数量（可选）

    Returns:
        选定的注意力后端类

    Raises:
        AssertionError: 如果 kv_cache_dtype 无效
    """
    # 验证 KV 缓存数据类型
    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"无效的 kv_cache_dtype: {kv_cache_dtype}。"
            f"有效值为：{valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    # 如果用户指定了 block_size 则使用，否则设为 None 让后端选择默认值
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    # 构建注意力选择器配置
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
    )

    # 使用缓存的后端选择函数
    return _cached_get_attn_backend(
        backend=vllm_config.attention_config.backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """带缓存的注意力后端选择函数。

    使用 functools.cache 装饰器缓存结果，避免重复选择。

    Args:
        backend: 后端名称
        attn_selector_config: 注意力选择器配置
        num_heads: 注意力头数量（可选）

    Returns:
        选定的注意力后端类

    Raises:
        ValueError: 如果当前平台不支持选定的后端
    """
    from vllm.platforms import current_platform

    # 从平台获取注意力后端类
    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"无效的注意力后端：{current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # 如果选定的后端需要特定的 KV 缓存布局，则进行调整
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "为 %s 后端使用 %s KV 缓存布局。",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    """选择 Mamba 注意力后端并懒加载。

    Args:
        mamba_type: Mamba 类型

    Returns:
        选定的 Mamba 注意力后端类
    """
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    """带缓存的 Mamba 注意力后端选择函数。

    Args:
        mamba_type: Mamba 类型

    Returns:
        选定的 Mamba 注意力后端类

    Raises:
        AssertionError: 如果 mamba_type 不是字符串
        ValueError: 如果 mamba_type 无效
    """
    assert mamba_type and isinstance(mamba_type, str)

    selected_backend = None
    try:
        # 从映射表获取后端名称
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]
        selected_backend = MambaAttentionBackendEnum[backend_name]
    except KeyError as e:
        raise ValueError(
            f"无效的 Mamba 注意力后端类型：'{backend_name}'。有效 "
            f"后端为：{list(MambaAttentionBackendEnum.__members__.keys())}"
        ) from e

    mamba_attn_backend = selected_backend.get_class()
    return mamba_attn_backend
