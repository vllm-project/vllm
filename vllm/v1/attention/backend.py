# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力后端抽象基类模块。

本模块定义了注意力后端的抽象基类和相关的元数据结构，负责：
- 定义注意力后端的统一接口（AttentionBackend）
- 提供注意力元数据基类（AttentionMetadata）
- 定义通用注意力元数据结构（CommonAttentionMetadata）
- 提供 CUDA Graph 支持级别枚举（AttentionCGSupport）
- 定义注意力元数据构建器接口（AttentionMetadataBuilder）
- 定义注意力层协议（AttentionLayer）
- 定义注意力实现基类（AttentionImplBase, AttentionImpl, MLAAttentionImpl）

主要类：
- AttentionType: 注意力类型枚举
- MultipleOf: 块大小倍数辅助类
- AttentionBackend: 注意力后端抽象基类
- AttentionMetadata: 注意力元数据基类
- CommonAttentionMetadata: 通用注意力元数据
- AttentionCGSupport: CUDA Graph 支持级别枚举
- AttentionMetadataBuilder: 注意力元数据构建器抽象基类
- AttentionLayer: 注意力层协议
- AttentionImplBase: 注意力实现基类
- AttentionImpl: 标准注意力实现抽象类
- MLAAttentionImpl: MLA 注意力实现抽象类
- SparseMLAAttentionImpl: 稀疏 MLA 注意力实现抽象类

主要函数：
- is_quantized_kv_cache: 检查 KV 缓存是否量化
- subclass_attention_backend: 派生注意力后端类
- subclass_attention_backend_with_overrides: 带覆盖的派生注意力后端类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import torch
from typing_extensions import deprecated

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.utils import KVCacheLayoutType
    from vllm.v1.kv_cache_interface import AttentionSpec


class AttentionType(str, Enum):
    """注意力类型枚举。

    使用字符串类型以兼容 `torch.compile`。

    Attributes:
        DECODER: 解码器注意力，处理前一层 Q/K/V 之间的注意力
        ENCODER: 编码器注意力，用于编码器 - 解码器架构中前一层 Q/K/V 之间的注意力
        ENCODER_ONLY: 仅编码器注意力，处理前一层 Q/K/V 之间的注意力
        ENCODER_DECODER: 编码器 - 解码器注意力，处理解码器 Q 和编码器 K/V 之间的注意力
    """

    DECODER = "decoder"
    """解码器注意力，处理前一层 Q/K/V 之间的注意力。"""
    ENCODER = "encoder"
    """编码器注意力，用于编码器 - 解码器架构中前一层 Q/K/V 之间的注意力。"""
    ENCODER_ONLY = "encoder_only"
    """仅编码器注意力，处理前一层 Q/K/V 之间的注意力。"""
    ENCODER_DECODER = "encoder_decoder"
    """编码器 - 解码器注意力，处理解码器 Q 和编码器 K/V 之间的注意力。"""


class MultipleOf:
    """块大小倍数辅助类。

    用于表示内核支持的块大小是某个基数的倍数。

    Attributes:
        base: 基数
    """

    base: int

    def __init__(self, base: int):
        """初始化倍数类。

        Args:
            base: 基数
        """
        self.base = base


class AttentionBackend(ABC):
    """注意力后端抽象基类。

    定义了所有注意力后端实现必须遵循的接口。

    Class Attributes:
        accept_output_buffer: 是否接受输出缓冲区（用于自定义操作前分配输出张量）
        supported_dtypes: 支持的数据类型列表
        supported_kv_cache_dtypes: 支持的 KV 缓存数据类型列表
        forward_includes_kv_cache_update: 前向传播是否包含 KV 缓存更新
    """

    # 对于某些注意力后端，我们在调用自定义操作之前分配输出张量。
    # 当启用分段 CUDA Graph 时，这确保输出张量在 CUDA Graph 内部被分配。
    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # 注意力前向传播是否包含 KV 缓存更新
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的块大小列表。

        Returns:
            支持的块大小列表，可以是整数或 MultipleOf 对象
        """
        return [MultipleOf(1)]

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称字符串
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        """获取注意力实现类。

        Returns:
            注意力实现类
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            元数据构建器类
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
            cache_dtype_str: 缓存数据类型字符串

        Returns:
            KV 缓存形状元组
        """
        raise NotImplementedError

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        """获取 KV 缓存块维度。

        由于不同后端的维度布局不同，此函数用于确定哪个维度是块索引。

        Args:
            block_size: 块大小
            num_kv_heads: KV 头数量
            head_size: 头大小
            cache_dtype_str: 缓存数据类型字符串

        Returns:
            块维度索引
        """
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(_S)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """获取 KV 缓存的物理（内存布局）维度顺序。

        例如，如果 KV 缓存形状是 [2, num_blocks, block_size, num_heads, head_size]，
        且 get_kv_cache_stride_order 返回 (1, 3, 0, 2, 4)，
        那么物理维度顺序是 [num_blocks, num_heads, 2, block_size, head_size]。

        如果此函数未实现或引发 NotImplementedError，
        KV 缓存的物理布局将与逻辑形状匹配。

        Args:
            include_num_layers_dimension: 是否包含 num_layers 维度。
                如果为 True，假设 num_layers 维度前置到逻辑 KV 缓存形状中。
                例如，返回 (2, 4, 0, 1, 3, 5) 对应于
                [num_blocks, num_heads, num_layers, 2, block_size, head_size]。

                如果返回的元组中不包含额外维度，
                物理布局将不包含层维度。

        Returns:
            一个整数元组，是 range(len(shape)) 的排列。
        """
        raise NotImplementedError

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        """获取完整的类名。

        Returns:
            (模块名，限定类名) 元组
        """
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取支持的头大小列表。

        Returns:
            支持的头大小列表
        """
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """检查是否支持给定的头大小。

        Args:
            head_size: 头大小

        Returns:
            是否支持
        """
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """检查是否支持给定的数据类型。

        Args:
            dtype: 数据类型

        Returns:
            是否支持
        """
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        """检查是否支持给定的 KV 缓存数据类型。

        Args:
            kv_cache_dtype: KV 缓存数据类型

        Returns:
            是否支持
        """
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """检查是否支持给定的块大小。

        Args:
            block_size: 块大小

        Returns:
            是否支持
        """
        if block_size is None:
            return True

        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True

        for supported_size in supported_kernel_block_sizes:
            if isinstance(supported_size, MultipleOf):
                supported_size = supported_size.base
            # 使用 hybrid_blocks 特性时，框架级别的块大小
            # 只需要是内核要求的倍数，即使内核要求固定的块大小。
            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        """获取首选块大小。

        Args:
            default_block_size: 默认块大小

        Returns:
            首选块大小
        """
        supported_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_sizes:
            return default_block_size

        if cls.supports_block_size(default_block_size):
            return default_block_size

        return min(s.base if isinstance(s, MultipleOf) else s for s in supported_sizes)

    @classmethod
    def is_mla(cls) -> bool:
        """检查是否为 MLA（多头潜在注意力）。

        Returns:
            是否为 MLA
        """
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        """检查是否支持 sink。

        Returns:
            是否支持
        """
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        """检查是否支持 ALiBi sqrt。

        Returns:
            是否支持
        """
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """检查是否支持多模态前缀。

        Returns:
            是否支持
        """
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        """检查是否为稀疏注意力。

        Returns:
            是否为稀疏
        """
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        """检查是否支持每头量化缩放因子。

        Returns:
            是否支持
        """
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """检查是否支持给定的注意力类型。

        默认只支持解码器注意力。
        子类应重写此方法以支持其他注意力类型。

        Args:
            attn_type: 注意力类型

        Returns:
            是否支持
        """
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        """检查是否支持给定的计算能力。

        Args:
            capability: 设备计算能力

        Returns:
            是否支持
        """
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        """检查是否支持给定的配置组合。

        Args:
            head_size: 头大小
            dtype: 数据类型
            kv_cache_dtype: KV 缓存数据类型
            block_size: 块大小
            use_mla: 是否使用 MLA
            has_sink: 是否有 sink
            use_sparse: 是否使用稀疏
            device_capability: 设备计算能力

        Returns:
            不支持的原因字符串，如果支持则返回 None
        """
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
    ) -> list[str]:
        """验证配置是否有效。

        Args:
            head_size: 头大小
            dtype: 数据类型
            kv_cache_dtype: KV 缓存数据类型
            block_size: 块大小
            use_mla: 是否使用 MLA
            has_sink: 是否有 sink
            use_sparse: 是否使用稀疏
            use_mm_prefix: 是否使用多模态前缀
            use_per_head_quant_scales: 是否使用每头量化缩放因子
            device_capability: 设备计算能力
            attn_type: 注意力类型

        Returns:
            无效原因列表
        """
        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("头大小不支持")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("数据类型不支持")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("KV 缓存数据类型不支持")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("块大小不支持")
        if use_mm_prefix and not cls.supports_mm_prefix():
            invalid_reasons.append(
                "部分多模态 token 全注意力不支持"
            )
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("不支持 MLA")
            else:
                invalid_reasons.append("不支持非 MLA")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("不支持注意力 sink")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("不支持稀疏")
            else:
                invalid_reasons.append("不支持非稀疏")
        if use_per_head_quant_scales and not cls.supports_per_head_quant_scales():
            invalid_reasons.append("不支持每头量化缩放因子")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("计算能力不支持")
        if not cls.supports_attn_type(attn_type):
            invalid_reasons.append(f"不支持注意力类型 {attn_type}")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        """获取所需的 KV 缓存布局。

        Returns:
            KV 缓存布局类型，如果无特殊要求则返回 None
        """
        return None


class AttentionMetadata:
    """注意力元数据基类。

    用于在注意力层之间传递批次级别的元数据。
    """

    pass


T = TypeVar("T", bound=AttentionMetadata)


@dataclass
class CommonAttentionMetadata:
    """每个批次的注意力元数据，跨层和后端共享。

    AttentionMetadataBuilder 实例使用它来构建每层元数据。

    对于许多张量，我们同时保留 GPU 和 CPU 版本。

    Attributes:
        query_start_loc: 查询起始位置 (GPU)
        query_start_loc_cpu: 查询起始位置 (CPU)
        seq_lens: 序列长度 (GPU)
        num_reqs: 请求数量
        num_actual_tokens: 实际 token 总数
        max_query_len: 批次中最长的查询
        max_seq_len: 最长的上下文长度（可能是上限）
        block_table_tensor: 块表张量
        slot_mapping: 槽映射
        causal: 是否因果注意力
        logits_indices_padded: 填充的 logits 索引（用于 FastPrefillAttentionBuilder）
        num_logits_indices: logits 索引数量
        encoder_seq_lens: 编码器序列长度（用于 CrossAttentionBuilder）
        encoder_seq_lens_cpu: 编码器序列长度 (CPU)
        dcp_local_seq_lens: DCP 世界中本地 rank 的序列长度
        dcp_local_seq_lens_cpu: DCP 本地序列长度 (CPU)
        is_prefilling: 是否仍在预填充阶段
        _seq_lens_cpu: 序列长度 (CPU) - 已弃用
        _num_computed_tokens_cpu: 已计算的 token 数 (CPU) - 已弃用
        _num_computed_tokens_cache: 已计算的 token 数缓存
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,)，每个请求在查询张量中的起始位置"""

    seq_lens: torch.Tensor
    """(batch_size,)，每个请求已计算的 token 数"""

    num_reqs: int
    """请求数量"""
    # TODO(lucas): 重命名为 num_tokens，因为可能有填充，当前名称会误导
    num_actual_tokens: int
    """批次中的实际 token 总数"""
    max_query_len: int
    """批次中最长的查询"""
    max_seq_len: int
    """最长的上下文长度（可能是上限）"""

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool = True

    # 用于 FastPrefillAttentionBuilder
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None

    # 用于 CrossAttentionBuilder
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None

    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """解码上下文并行世界中本地 rank 的序列长度"""

    is_prefilling: torch.Tensor | None = None
    """(batch_size,) 布尔张量：如果请求仍在预填充阶段则为 True
    (num_computed_tokens < num_prompt_tokens)。某些后端用它来
    区分实际的 decode 和短的 extend。"""

    # 警告：已弃用的字段。将在未来版本（v0.15.0）中移除
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    _num_computed_tokens_cache: torch.Tensor | None = None

    def batch_size(self) -> int:
        """获取批次大小。

        Returns:
            批次大小（请求数量）
        """
        return self.seq_lens.shape[0]

    def naive_query_lens(self) -> torch.Tensor:
        """计算查询长度（简单方法，假设查询结束于下一个查询开始处）。

        Returns:
            查询长度张量
        """
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    def replace(self, **kwargs) -> "CommonAttentionMetadata":
        """替换字段值。

        Args:
            **kwargs: 要替换的字段

        Returns:
            新的 CommonAttentionMetadata 实例
        """
        return replace(self, **kwargs)

    @property
    @deprecated(
        """
    优先直接使用设备上的 seq_lens，以避免隐式的 H<>D 同步。
    如果需要 CPU 副本，请使用 `seq_lens.cpu()`。
    将在未来版本中移除，请尽快迁移。
    """
    )
    def seq_lens_cpu(self) -> torch.Tensor:
        """获取 seq_lens 的 CPU 副本（已弃用）。

        Returns:
            seq_lens 的 CPU 副本
        """
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu

    @property
    @deprecated(
        """
    优先直接使用设备上的 seq_lens，以避免破坏完全异步调度的隐式 H<>D 同步。
    如果需要 CPU 副本，可以从 query_start_loc_cpu 和 seq_lens 派生。
    将在未来版本中移除，请尽快迁移。
    """
    )
    def num_computed_tokens_cpu(self) -> torch.Tensor:
        """获取已计算的 token 数 (CPU)（已弃用）。

        Returns:
            已计算的 token 数
        """
        if self._num_computed_tokens_cpu is None:
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu

    def compute_num_computed_tokens(self) -> torch.Tensor:
        """在设备上计算 num_computed_tokens (seq_lens - query_lens)。

        Returns:
            已计算的 token 数张量
        """
        if self._num_computed_tokens_cache is None:
            query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
            self._num_computed_tokens_cache = self.seq_lens - query_lens
        return self._num_computed_tokens_cache

    # TODO(lucas): 移除，一旦我们有 FULL-CG spec-decode 支持
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        """获取未填充的元数据副本。

        Args:
            num_actual_tokens: 实际 token 数量
            num_actual_reqs: 实际请求数量

        Returns:
            未填充的 CommonAttentionMetadata 实例
        """
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            _seq_lens_cpu=self._seq_lens_cpu[:num_actual_reqs]
            if self._seq_lens_cpu is not None
            else None,
            _num_computed_tokens_cpu=self._num_computed_tokens_cpu[:num_actual_reqs]
            if self._num_computed_tokens_cpu is not None
            else None,
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            encoder_seq_lens=maybe_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=maybe_slice_reqs(self.encoder_seq_lens_cpu),
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
            is_prefilling=maybe_slice_reqs(self.is_prefilling),
        )


M = TypeVar("M")


class AttentionCGSupport(Enum):
    """注意力后端 CUDA Graph 支持级别常量。

    这里不考虑级联注意力，因为目前它从不是 CUDA Graph 支持的。

    Attributes:
        ALWAYS: 始终支持 CUDA Graph；支持混合预填充/解码
        UNIFORM_BATCH: 支持均匀批次的 CUDA Graph，可用于 spec-decode
        UNIFORM_SINGLE_TOKEN_DECODE: 仅支持 query_len==1 解码批次的 CUDA Graph
        NEVER: 不支持 CUDA Graph
    """

    ALWAYS = 3
    """始终支持 CUDA Graph；支持混合预填充/解码"""
    UNIFORM_BATCH = 2
    """支持均匀批次的 CUDA Graph，批次中只包含相同查询长度的请求，
    可用于 spec-decode，即"解码"是 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """支持仅包含 query_len==1 解码批次的 CUDA Graph"""
    NEVER = 0
    """不支持 CUDA Graph"""


class AttentionMetadataBuilder(ABC, Generic[M]):
    """注意力元数据构建器抽象基类。

    用于构建注意力前向传播所需的元数据。

    Class Attributes:
        _cudagraph_support: CUDA Graph 支持级别
        reorder_batch_threshold: 批次重排序阈值
        supports_update_block_table: 是否支持更新元数据中的块表
    """

    # 此后端/构建器是否支持注意力 CUDA Graph（默认：不支持）
    # 不要直接访问。调用 get_cudagraph_support() 代替。
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # 此后端/构建器是否重新排序批次？
    # 如果不，设为 None。否则设为将被拉到批次前面的查询长度。
    reorder_batch_threshold: int | None = None
    # 此后端/构建器是否支持更新元数据中的块表
    supports_update_block_table: bool = False

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        """初始化元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备
        """
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        """获取此构建器类的 CUDA Graph 支持级别。

        Args:
            vllm_config: vLLM 配置
            kv_cache_spec: KV 缓存规格

        Returns:
            CUDA Graph 支持级别
        """
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        """初始化批次重排序阈值。

        Args:
            reorder_batch_threshold: 重排序阈值
            supports_spec_as_decode: 是否支持 spec-as-decode
            supports_dcp_with_varlen: 是否支持带变长的 DCP
        """
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # 如果后端支持 spec-as-decode 内核，我们可以根据
            # 配置中的推测 token 数量设置 reorder_batch_threshold。
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                max_num_queries_for_spec = (
                    1
                    + (2 if speculative_config.parallel_drafting else 1)
                    * speculative_config.num_speculative_tokens
                )
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    max_num_queries_for_spec,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """构建注意力元数据。

        这是构建注意力元数据的核心方法。
        某些构建器（如 MLA）要求在 build 之前调用 reorder_batch。

        Args:
            common_prefix_len: 批次的前缀公共长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 元数据是否优先考虑构建速度而非执行速度。
                可用于 spec-decode，因为构建结果可能只用于少数层/迭代。
        """
        raise NotImplementedError

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """更新注意力元数据的块表。

        当有多个 kv-cache 组创建几乎相同但块表不同的元数据时，
        这种方法更快。

        仅在 supports_update_block_table 为 True 时需要实现。

        Args:
            metadata: 注意力元数据
            blk_table: 块表
            slot_mapping: 槽映射

        Returns:
            更新后的元数据
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """为 CUDA Graph 捕获构建注意力元数据。

        默认调用 build。
        重写此方法的子类应调用 self.build 或
        super().build_for_cudagraph_capture。

        Args:
            common_attn_metadata: 通用注意力元数据

        Returns:
            用于 CUDA Graph 捕获的元数据
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """为推测模型构建注意力元数据。

        默认调用 build。

        Args:
            common_attn_metadata: 通用注意力元数据
            draft_index: 当前推测操作的索引。
                当推测 token 链时，此索引指第 i 个 token 的草案尝试。
                对于基于树的注意力，此索引指 token 树中第 i 层的草案尝试。

        Returns:
            用于推测的元数据
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        """判断是否使用级联注意力。

        Args:
            common_prefix_len: 公共前缀长度
            query_lens: 查询长度数组
            num_query_heads: 查询头数量
            num_kv_heads: KV 头数量
            use_alibi: 是否使用 ALiBi
            use_sliding_window: 是否使用滑动窗口
            use_local_attention: 是否使用局部注意力
            num_sms: SM 数量
            dcp_world_size: DCP 世界大小

        Returns:
            是否使用级联注意力
        """
        return False


class AttentionLayer(Protocol):
    """注意力层协议。

    定义了注意力层必须实现的接口。

    Attributes:
        _q_scale: Q 缩放因子
        _k_scale: K 缩放因子
        _v_scale: V 缩放因子
        _q_scale_float: Q 缩放因子（浮点数）
        _k_scale_float: K 缩放因子（浮点数）
        _v_scale_float: V 缩放因子（浮点数）
        _prob_scale: 概率缩放因子
    """

    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            kv_cache: KV 缓存
            attn_metadata: 注意力元数据

        Returns:
            输出张量
        """
        ...


class AttentionImplBase(ABC, Generic[T]):
    """注意力实现的基类。

    包含标准 AttentionImpl 和 MLAAttentionImpl 共享的
    通用属性和初始化逻辑。不定义前向方法——
    子类定义自己的前向接口。

    Attributes:
        num_heads: 头数量
        head_size: 头大小
        scale: 缩放因子
        can_return_lse_for_decode: 是否可以为解码返回 softmax LSE
        supports_pcp: 是否支持预填充上下文并行
        supports_mtp_with_cp_non_trivial_interleave_size: 当 cp_kv_cache_interleave_size > 1 时是否支持 MTP
        need_to_return_lse_for_decode: 是否需要为解码返回 LSE
        supports_quant_query_input: 是否支持预量化查询输入
        dcp_world_size: DCP 世界大小
        dcp_rank: DCP rank
        pcp_world_size: PCP 世界大小
        pcp_rank: PCP rank
        total_cp_world_size: 总 CP 世界大小
        total_cp_rank: 总 CP rank
    """

    # 所有实现都应具有的必要属性
    num_heads: int
    head_size: int
    scale: float

    # 注意力实现是否可以为解码返回 softmax LSE。
    # 某些特性（如解码上下文并行）需要 softmax LSE。
    can_return_lse_for_decode: bool = False

    # 注意力实现是否支持预填充上下文并行。
    supports_pcp: bool = False
    # 当 cp_kv_cache_interleave_size > 1 时，注意力实现（或操作）是否支持 MTP
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # 某些注意力后端即使可以返回 LSE，也可能不想总是返回（出于效率考虑）
    need_to_return_lse_for_decode: bool = False

    # 此注意力实现是否支持预量化查询输入。
    # 当为 True 时，注意力层将在传递给后端之前量化查询，
    # 允许 torch.compile 将量化与前面的操作融合。
    # 当使用 FP8 KV 缓存与兼容的注意力内核（如 TRT-LLM）时，
    # 通常会支持此特性。
    # 子类应在 __init__ 中设置此属性。
    # TODO 添加到更多后端支持：
    # https://github.com/vllm-project/vllm/issues/25584
    supports_quant_query_input: bool = False

    dcp_world_size: int
    dcp_rank: int

    pcp_world_size: int
    pcp_rank: int

    total_cp_world_size: int
    total_cp_rank: int

    def __new__(cls, *args, **kwargs):
        """创建新实例。

        使用 __new__ 以便所有子类都会调用此方法。

        Returns:
            新实例
        """
        # 使用 __new__ 以便所有子类都会调用此方法
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP 可能在测试中未初始化
            self.dcp_world_size = 1
            self.dcp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """加载后处理权重。

        Args:
            act_dtype: 激活数据类型
        """
        pass


class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """标准注意力实现，带有 forward 方法。

    定义了标准注意力实现的接口。
    """

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        """初始化注意力实现。

        Args:
            num_heads: 头数量
            head_size: 头大小
            scale: 缩放因子
            num_kv_heads: KV 头数量（可选）
            alibi_slopes: ALiBi 斜率（可选）
            sliding_window: 滑动窗口大小（可选）
            kv_cache_dtype: KV 缓存数据类型
            logits_soft_cap: logits 软上限（可选）
            attn_type: 注意力类型
            kv_sharing_target_layer_name: KV 共享目标层名称（可选）
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """注意力前向传播。

        Args:
            layer: 注意力层
            query: 查询张量
            key: 键张量
            value: 值张量
            kv_cache: KV 缓存
            attn_metadata: 注意力元数据
            output: 输出缓冲区（可选）
            output_scale: 输出缩放因子（可选）
            output_block_scale: 输出块缩放因子（可选）

        Returns:
            输出张量
        """
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        """检查是否支持融合输出量化。

        用于 AttnFusionPass，只将输出量化融合到支持的实现上。

        Args:
            quant_key: 描述量化操作的 QuantKey 对象

        Returns:
            是否支持此类型的量化融合
        """
        return False

    def fused_rope_kvcache_supported(self):
        """检查是否支持 RoPE+KVCache 融合。

        用于 RopeKVCacheFusionPass，只将 RoPE 操作与
        KV 缓存更新融合到支持的实现上。

        Returns:
            是否支持
        """
        return False

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

        如果 `fused_rope_kvcache_supported` 返回 True，
        此方法将被 torch.ops.vllm.fused_rope_and_unified_kv_cache_update
        调用以执行原地 RoPE 和 KV 缓存更新。

        Args:
            layer: 注意力层
            query: 查询张量
            key: 键张量
            value: 值张量
            positions: 位置
            cos_sin_cache: cos/sin 缓存
            is_neox: 是否为 NeoX 风格
            kv_cache: KV 缓存
            layer_slot_mapping: 层槽映射
        """
        raise NotImplementedError


class MLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """MLA 注意力实现，带有 forward_mqa 和 forward_mha 方法。

    多头潜在注意力（MLA）的实现接口。
    """

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA 特定参数
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        """初始化 MLA 注意力实现。

        Args:
            num_heads: 头数量
            head_size: 头大小
            scale: 缩放因子
            num_kv_heads: KV 头数量
            alibi_slopes: ALiBi 斜率
            sliding_window: 滑动窗口大小
            kv_cache_dtype: KV 缓存数据类型
            logits_soft_cap: logits 软上限
            attn_type: 注意力类型
            kv_sharing_target_layer_name: KV 共享目标层名称
            q_lora_rank: Q LoRA rank
            kv_lora_rank: KV LoRA rank
            qk_nope_head_dim: QK 无位置头维度
            qk_rope_head_dim: QK RoPE 头维度
            qk_head_dim: QK 头维度
            v_head_dim: V 头维度
            kv_b_proj: KV 投影
            indexer: 索引器
            q_pad_num_heads: Q 填充头数量
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """MHA 风格的预填充前向传播。

        Args:
            q: 查询张量
            kv_c_normed: 归一化的 KV 压缩
            k_pe: K 位置编码
            kv_c_and_k_pe_cache: KV 压缩和 K 位置编码缓存
            attn_metadata: 注意力元数据
            k_scale: K 缩放因子
            output: 输出张量
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MQA 风格的解码前向传播。

        Args:
            q: 查询张量或张量元组
            kv_c_and_k_pe_cache: KV 压缩和 K 位置编码缓存
            attn_metadata: 注意力元数据
            layer: 注意力层

        Returns:
            (输出，可选的额外输出) 元组
        """
        raise NotImplementedError

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """执行 KV 缓存更新。

        Args:
            kv_c_normed: 归一化的 KV 压缩
            k_pe: K 位置编码
            kv_cache: KV 缓存
            slot_mapping: 槽映射
            kv_cache_dtype: KV 缓存数据类型
            k_scale: K 缩放因子
        """
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )


class SparseMLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """稀疏 MLA 注意力实现，仅带有 forward_mqa 方法。

    稀疏 MLA 实现仅支持解码（MQA 风格）注意力。
    它们不支持预填充（MHA 风格）注意力。
    """

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA 特定参数
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        """初始化稀疏 MLA 注意力实现。

        Args:
            num_heads: 头数量
            head_size: 头大小
            scale: 缩放因子
            num_kv_heads: KV 头数量
            alibi_slopes: ALiBi 斜率
            sliding_window: 滑动窗口大小
            kv_cache_dtype: KV 缓存数据类型
            logits_soft_cap: logits 软上限
            attn_type: 注意力类型
            kv_sharing_target_layer_name: KV 共享目标层名称
            q_lora_rank: Q LoRA rank
            kv_lora_rank: KV LoRA rank
            qk_nope_head_dim: QK 无位置头维度
            qk_rope_head_dim: QK RoPE 头维度
            qk_head_dim: QK 头维度
            v_head_dim: V 头维度
            kv_b_proj: KV 投影
            indexer: 索引器
            q_pad_num_heads: Q 填充头数量
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MQA 风格的解码前向传播。

        Args:
            q: 查询张量或张量元组
            kv_c_and_k_pe_cache: KV 压缩和 K 位置编码缓存
            attn_metadata: 注意力元数据
            layer: 注意力层

        Returns:
            (输出，可选的额外输出) 元组
        """
        raise NotImplementedError

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """执行 KV 缓存更新。

        Args:
            kv_c_normed: 归一化的 KV 压缩
            k_pe: K 位置编码
            kv_cache: KV 缓存
            slot_mapping: 槽映射
            kv_cache_dtype: KV 缓存数据类型
            k_scale: K 缩放因子
        """
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    """检查 KV 缓存是否量化。

    Args:
        kv_cache_dtype: KV 缓存数据类型字符串

    Returns:
        是否量化
    """
    return kv_cache_dtype.startswith("fp8")


def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]:
    """派生一个新的注意力后端子类。

    返回一个新子类，其中 `get_builder_cls` 返回 `builder_cls`。

    Args:
        name_prefix: 名称前缀
        attention_backend_cls: 原始注意力后端类
        builder_cls: 元数据构建器类

    Returns:
        新的注意力后端子类
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(
        name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls}
    )


def subclass_attention_backend_with_overrides(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    overrides: dict[str, Any],
) -> type[AttentionBackend]:
    """派生一个带覆盖的注意力后端子类。

    Args:
        name_prefix: 名称前缀
        attention_backend_cls: 原始注意力后端类
        overrides: 覆盖属性字典

    Returns:
        新的注意力后端子类
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore
    return type(name, (attention_backend_cls,), overrides)
