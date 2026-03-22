# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention DiffKV 后端模块。

本模块实现了支持不同 K/V 头大小的 FlashAttention 后端，负责：
- 实现 DiffKV 版本的 FlashAttention 后端
- 支持 K 和 V 有不同的头大小
- 使用 Triton kernel 进行 KV 缓存更新

主要类：
- FlashAttentionDiffKVBackend: DiffKV FlashAttention 后端类
- FlashAttentionDiffKVImpl: DiffKV 实现类
"""

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from .flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    cascade_attention,
)

logger = init_logger(__name__)


class FlashAttentionDiffKVBackend(FlashAttentionBackend):
    """FlashAttention DiffKV 后端类。

    支持 K 和 V 有不同头大小的 FlashAttention 后端。
    默认 V 的头大小为 128。

    Class Attributes:
        head_size_v: V 的头大小，默认为 128
    """
    # Default to 128 for this backend
    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        """设置 V 的头大小。

        Args:
            head_size_v: V 的头大小
        """
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "FLASH_ATTN_DIFFKV"
        """
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            FlashAttentionDiffKVImpl 类
        """
        return FlashAttentionDiffKVImpl

    # Do not modify the interface of get_kv_cache_shape,
    # but consider head_size_v when returning result.
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
            head_size: K 的头大小
            cache_dtype_str: 缓存数据类型

        Returns:
            KV 缓存形状元组

        Raises:
            ValueError: 如果块大小不是 16 的倍数
        """
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + FlashAttentionDiffKVBackend.head_size_v,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """获取 KV 缓存步幅顺序。

        Args:
            include_num_layers_dimension: 是否包含层数维度

        Returns:
            步幅顺序元组

        Raises:
            ValueError: 如果缓存布局未知
        """
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size,
            # num_kv_heads, head_size + head_size_v)
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers,
            # block_size, head_size + head_size_v)
            return (1, 3, 0, 2, 4)
        elif cache_layout == "HND":
            stride_order = (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


class FlashAttentionDiffKVImpl(FlashAttentionImpl):
    """FlashAttention DiffKV 实现类。

    支持 K 和 V 有不同头大小的 FlashAttention 实现。
    """
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FlashAttention 前向传播（DiffKV 版本）。

        Args:
            layer: 注意力层模块
            query: 形状 = [num_tokens, num_heads, head_size]
            key: 形状 = [num_tokens, num_kv_heads, head_size]
            value: 形状 = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: 形状 = [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
            attn_metadata: 注意力元数据
            output: 输出张量
            output_scale: 输出缩放因子（不支持）
            output_block_scale: 输出块缩放因子（不支持）

        Returns:
            形状 = [num_tokens, num_heads * head_size_v] 的输出张量

        Note:
            FP8 量化时，flash-attn 期望 {q,k,v}_descale 的大小为
            (num_sequences, num_kv_heads)。我们使用 torch 的 .expand()
            来避免复制值。
        """
        assert output is not None, "必须提供输出张量。"
        assert self.vllm_flash_attn_version is not None, (
            "未检测到 FlashAttention 版本。"
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "FlashAttentionImpl 不支持融合输出量化。"
            )

        if attn_metadata is None:
            # 性能分析运行。
            return output.fill_(0)

        attn_type = self.attn_type

        # 重要提示！
        # NOTE(woosuk): 使用分段 CUDA 图时，此方法在 eager-mode PyTorch 中执行。
        # 因此，我们需要小心此方法中的任何 CPU 开销。
        # 例如，`view`和`slice`（或`[:n]`）操作即使在不调用任何 GPU 操作的情况下也慢得惊人。
        # 尽可能减少此方法中的 PyTorch 操作。
        # 在此方法中进行任何更改时，请基准测试性能以确保不会引入任何开销。

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 以不同方式处理编码器注意力 - 不需要 KV 缓存
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # 对于编码器注意力，我们直接使用 Q、K、V 张量而不进行缓存
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # 对于解码器和交叉注意力，按原样使用 KV 缓存
        # K 和 V 有不同的头大小
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]

        # key 和 value 在交叉注意力的情况下可能为 None。
        # 它们基于编码器的输出计算一次，然后缓存在 KV 缓存中。
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # 重塑输入键和值并将其存储在缓存中。
            # 如果与早期注意力层共享 KV 缓存，则跳过此操作。
            # NOTE(woosuk): 在这里，key 和 value 经过 padding，而 slot_mapping 没有。
            # 但是，我们不需要执行 key[:num_actual_tokens] 和 value[:num_actual_tokens]，
            # 因为 reshape_and_cache_flash 操作使用 slot_mapping 的形状来确定
            # 实际 token 的数量。

            # 为不同头大小的 K 和 V 更新 kv_cache
            triton_reshape_and_cache_flash_diffkv(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            # query 在注意力层中量化
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # Cascade 注意力（罕见情况）。
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output
