# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention, differential K/V head sizes."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import AttentionType, LayerConfig
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from .flash_attn import (
    FlashAttentionBackend,
    _fa_sliding_window,
    cascade_attention,
)

logger = init_logger(__name__)


class FlashAttentionDiffKVBackend(FlashAttentionBackend):
    # Default to 128 for this backend
    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_DIFFKV"

    def bind_layer(self, layer_config: LayerConfig) -> None:
        super().bind_layer(layer_config)
        # Re-derive the FA version with diff-kv context so get_flash_attn_version
        # can apply the FA3 -> FA4 upgrade rule for sinks + hdim != hdim_v.
        self._vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=layer_config.alibi_slopes is not None,
            head_size=layer_config.head_size,
            head_size_v=type(self).head_size_v,
            has_sinks=layer_config.extra.get("sinks") is not None,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        layer_config = layer.layer_config
        if layer_config.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            # For encoder attention, Q/K/V are used directly without caching.
            return

        # DiffKV packs K and V into a single tensor along the last dim:
        #   kv_cache shape: [num_blocks, block_size, num_kv_heads,
        #                    head_size_k + head_size_v]
        # (B, H, N, C) -> (B, N, H, C) for kernel compatibility.
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
            slot_mapping,
            layer_config.kv_cache_dtype,
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
        output: torch.Tensor,
        *,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention (differential K/V head sizes).

        Per-step state is read off the microbatch's slot in ``self._step`` (set
        by ``prep_forward``); per-layer config off ``layer.layer_config``.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
        Returns:
            shape = [num_tokens, num_heads * head_size_v]
        """
        layer_config = layer.layer_config
        head_size = layer_config.head_size
        scale = layer_config.scale
        num_kv_heads = layer_config.num_kv_heads
        attn_type = layer_config.attn_type
        kv_cache_dtype = layer_config.kv_cache_dtype
        logits_soft_cap = layer_config.logits_soft_cap or 0
        sliding_window = _fa_sliding_window(layer_config.sliding_window, attn_type)
        sinks = layer_config.extra.get("sinks")
        alibi_slopes = self._alibi_slopes
        fa_version = self._vllm_flash_attn_version
        assert fa_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttention"
            )

        md = self._step[self._current_ubatch_id()]
        assert md is not None, "forward() called before prep_forward()/_build_step()"
        num_actual_tokens = md.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                layer,
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                md,
            )

        # (B, H, N, C) -> (B, N, H, C) for kernel compatibility. K and V are
        # packed along the content dim and split by head_size (not interleaved).
        kv_cache = kv_cache.transpose(1, 2)
        key_cache = kv_cache[..., :head_size]
        value_cache = kv_cache[..., head_size:]
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttentionDiffKV): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        if is_quantized_kv_cache(kv_cache_dtype):
            # queries are quantized in the attention layer
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        if not md.use_cascade:
            cu_seqlens_q = md.query_start_loc
            seqused_k = md.seq_lens
            max_seqlen_q = md.max_query_len
            max_seqlen_k = md.max_seq_len
            block_table = md.block_table
            scheduler_metadata = md.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, num_kv_heads)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    layer,
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    md,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                sliding_window_size = (
                    list(sliding_window) if sliding_window is not None else None
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
                    softmax_scale=scale,
                    causal=md.causal,
                    alibi_slopes=alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=fa_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=md.max_num_splits,
                    s_aux=sinks,
                )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=md.query_start_loc,
            max_query_len=md.max_query_len,
            cu_prefix_query_lens=md.cu_prefix_query_lens,
            prefix_kv_lens=md.prefix_kv_lens,
            suffix_kv_lens=md.suffix_kv_lens,
            max_kv_len=md.max_seq_len,
            softmax_scale=scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            block_table=md.block_table,
            common_prefix_len=md.common_prefix_len,
            max_num_splits=md.max_num_splits,
            fa_version=fa_version,
            prefix_scheduler_metadata=md.prefix_scheduler_metadata,
            suffix_scheduler_metadata=md.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=sinks,
        )
        return output
