from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.prefill_only.layers.attention.backends.abstract import (
    PrefillOnlyAttentionBackend, PrefillOnlyAttentionImpl,
    PrefillOnlyAttentionMetadata, PrefillOnlyAttentionMetadataBuilder)


class PrefillOnlyFlashAttentionBackend(PrefillOnlyAttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "flash_attn"

    @staticmethod
    def get_impl_cls() -> Type["PrefillOnlyFlashAttentionImpl"]:
        return PrefillOnlyFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["PrefillOnlyFlashAttentionMetadata"]:
        return PrefillOnlyFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["PrefillOnlyAttentionMetadataBuilder"]:
        return PrefillOnlyFlashAttentionMetadataBuilder


@dataclass
class PrefillOnlyFlashAttentionMetadata(PrefillOnlyAttentionMetadata):
    pass


class PrefillOnlyFlashAttentionMetadataBuilder(
        PrefillOnlyAttentionMetadataBuilder[PrefillOnlyFlashAttentionMetadata]
):
    pass


class PrefillOnlyFlashAttentionImpl(PrefillOnlyAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")

        from vllm_flash_attn import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if sliding_window is not None:
            # NOTE(woosuk): flash-attn's sliding window does not work with
            # paged KV cache.
            raise ValueError(
                "Sliding window is not supported in FlashAttention.")

        support_head_sizes = (
            PrefillOnlyFlashAttentionBackend.get_supported_head_sizes())
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PrefillOnlyFlashAttentionMetadata,
        kv_cache: Optional[torch.Tensor] = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        if attn_type == AttentionType.ENCODER:
            causal = False
        elif attn_type == AttentionType.DECODER:
            causal = True
        else:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PrefillOnlyFlashAttentionImpl")

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        attn_output = self.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.seq_start_loc,
            cu_seqlens_k=attn_metadata.seq_start_loc,
            max_seqlen_q=attn_metadata.max_seq_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=causal,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            softcap=self.logits_soft_cap,
        )

        # Reshape the output tensor.
        return attn_output.view(num_tokens, hidden_size)
