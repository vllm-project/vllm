from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.encode_only.layers.attention.backends.abstract import (
    EncodeOnlyAttentionBackend, EncodeOnlyAttentionImpl,
    EncodeOnlyAttentionMetadata, EncodeOnlyAttentionMetadataBuilder)


class EncodeOnlyFlashInferBackend(EncodeOnlyAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def get_impl_cls() -> Type["EncodeOnlyFlashInferImpl"]:
        return EncodeOnlyFlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type["EncodeOnlyAttentionMetadata"]:
        return EncodeOnlyFlashInferMetadata

    @staticmethod
    def get_builder_cls() -> Type["EncodeOnlyFlashInferMetadataBuilder"]:
        return EncodeOnlyFlashInferMetadataBuilder


@dataclass
class EncodeOnlyFlashInferMetadata(EncodeOnlyAttentionMetadata):
    pass


class EncodeOnlyFlashInferMetadataBuilder(
        EncodeOnlyAttentionMetadataBuilder[EncodeOnlyFlashInferMetadata]):
    pass


class EncodeOnlyFlashInferImpl(EncodeOnlyAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is not None:
            raise ValueError("Sliding window is not supported in FlashInfer.")
        self.sliding_window = (-1, -1)
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        from vllm_flash_attn import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: EncodeOnlyFlashInferMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashInfer.")

        if attn_type != AttentionType.ENCODER:
            raise NotImplementedError("Decoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "EncodeOnlyFlashInferImpl")

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Because encode only models do not involve kv cache
        # When using Flashinfer backend in encode only models,
        # you are actually using FLASH ATTN backend
        attn_output = self.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.seq_start_loc,
            cu_seqlens_k=attn_metadata.seq_start_loc,
            max_seqlen_q=attn_metadata.max_seq_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
        )

        # Reshape the output tensor.
        return attn_output.view(num_tokens, hidden_size)
