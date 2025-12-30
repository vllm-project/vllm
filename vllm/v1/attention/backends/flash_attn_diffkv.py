# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func
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
    # Default to 128 for this backend
    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
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
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size_v]
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

        attn_type = self.attn_type

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
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
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

        # For decoder and cross-attention, use KV cache as before
        # Different head_size for K and V
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.

            # kv_cache update for different head_size K and V
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
            # queries are quantized in the attention layer
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
                    window_size=self.sliding_window,
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

        # Cascade attention (rare case).
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
