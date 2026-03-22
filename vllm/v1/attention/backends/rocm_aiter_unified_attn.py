# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM Aiter Unified Attention 后端模块。

本模块实现了基于 ROCm Aiter Unified Attention 的注意力后端，负责：
- 实现 Aiter Unified Attention 后端类
- 使用 Triton unified attention kernel
- 支持编码器和解码器注意力
- 支持 FP8 量化和融合输出量化

主要类：
- RocmAiterUnifiedAttentionBackend: Aiter Unified Attention 后端类
- RocmAiterUnifiedAttentionImpl: 后端实现类
"""

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    """Rocm Aiter Unified Attention 后端类。

    基于 ROCm Aiter Unified Attention 实现的注意力后端。
    支持所有注意力类型（解码器、编码器、编码器 - 解码器）。
    """
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的内核块大小列表。

        Returns:
            支持的块大小列表 [MultipleOf(16)]
        """
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        """获取首选块大小。

        Args:
            default_block_size: 默认块大小

        Returns:
            首选块大小 64
        """
        logger.warning_once(
            "[ROCM_AITER_UNIFIED_ATTN]: Setting kv cache block size to 64."
        )
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """检查是否支持指定的块大小。

        Args:
            block_size: 块大小

        Returns:
            是否支持（必须是 16 的倍数）
        """
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """检查是否支持指定的头大小。

        Args:
            head_size: 头大小

        Returns:
            是否支持（必须 >= 32）
        """
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """检查是否支持 multimodal 前缀。

        Returns:
            True（支持）
        """
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        """检查是否支持 sink token。

        Returns:
            True（支持）
        """
        return True

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "ROCM_AITER_UNIFIED_ATTN"
        """
        return "ROCM_AITER_UNIFIED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            RocmAiterUnifiedAttentionImpl 类
        """
        return RocmAiterUnifiedAttentionImpl

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

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """检查是否使用级联注意力。

        Returns:
            False（不支持级联注意力）
        """
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            RocmAttentionMetadataBuilder 类
        """
        return RocmAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """检查是否支持指定的注意力类型。

        RocmAiterUnifiedAttention 支持所有注意力类型。

        Args:
            attn_type: 注意力类型

        Returns:
            是否支持
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )


class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    """Rocm Aiter Unified Attention 实现类。

    基于 ROCm Aiter Unified Attention 实现的注意力后端。
    使用 Triton unified attention kernel，支持编码器和解码器注意力。
    """
    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持融合输出量化。

        Args:
            quant_key: 量化密钥

        Returns:
            是否支持（仅支持 kFp8StaticTensorSym）
        """
        return quant_key == kFp8StaticTensorSym

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
        sinks: torch.Tensor | None = None,
    ) -> None:
        """初始化 Rocm Aiter Unified Attention 实现。

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
            sinks: sink token 张量
        """
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using aiter unified attention for RocmAiterUnifiedAttentionImpl"
        )
        from aiter.ops.triton.unified_attention import unified_attention

        self.unified_attention = unified_attention
        self.supports_quant_query_input = True

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
        """使用 FlashAttention 进行前向传播。

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
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        softmax_scale = self.scale
        fp8_post_attn_v_rescale = False
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # When Q is FP8, triton kernel skips K/V dequant (for fp8xfp8 matmul).
            # Compensate by absorbing q_scale and k_scale into softmax_scale, and
            # v_scale into output_scale (or post-multiplying if no fusion).
            if query.dtype == self.fp8_dtype:
                softmax_scale = self.scale * layer._q_scale_float * layer._k_scale_float
                if output_scale is not None:
                    output_scale = output_scale / layer._v_scale_float
                else:
                    fp8_post_attn_v_rescale = True

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,
            key.shape[1] if key is not None else self.num_kv_heads,
        )

        self.unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # q_scale absorbed into softmax_scale
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            sinks=self.sinks,
            output_scale=output_scale,
        )

        if fp8_post_attn_v_rescale:
            output[:num_actual_tokens].mul_(layer._v_scale_float)

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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
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

    def fused_rope_kvcache_supported(self):
        """检查是否支持融合 RoPE KV 缓存。

        Returns:
            是否支持（需要 rocm_aiter_ops 启用）
        """
        return rocm_aiter_ops.is_enabled()

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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = kv_cache.unbind(0)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

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
