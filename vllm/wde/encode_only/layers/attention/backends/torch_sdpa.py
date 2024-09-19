from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.utils import is_pin_memory_available
from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.encode_only.layers.attention.backends.abstract import (
    EncodeOnlyAttentionBackend, EncodeOnlyAttentionImpl,
    EncodeOnlyAttentionMetadata, EncodeOnlyAttentionMetadataBuilder)

pin_memory = is_pin_memory_available()


class EncodeOnlyTorchSDPABackend(EncodeOnlyAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "torch_sdpa"

    @staticmethod
    def get_impl_cls() -> Type["EncodeOnlyTorchSDPABackendImpl"]:
        return EncodeOnlyTorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["EncodeOnlyTorchSDPAMetadata"]:
        return EncodeOnlyTorchSDPAMetadata

    @staticmethod
    def get_builder_cls() -> Type["EncodeOnlyAttentionMetadataBuilder"]:
        return EncodeOnlyTorchSDPAMetadataBuilder


@dataclass
class EncodeOnlyTorchSDPAMetadata(EncodeOnlyAttentionMetadata):
    pass


class EncodeOnlyTorchSDPAMetadataBuilder(
        EncodeOnlyAttentionMetadataBuilder[EncodeOnlyTorchSDPAMetadata]):
    pass


class EncodeOnlyTorchSDPABackendImpl(EncodeOnlyAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "Torch SPDA does not support block-sparse attention.")
        if logits_soft_cap is not None:
            raise ValueError("Torch SPDA does not support logits soft cap.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: EncodeOnlyTorchSDPAMetadata,  # type: ignore
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in TorchSDPA.")

        if attn_type != AttentionType.ENCODER:
            raise NotImplementedError("Decoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "EncodeOnlyTorchSDPABackendImpl")

        num_tokens, hidden_size = query.shape

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        start = 0
        output = torch.empty((num_tokens, self.num_heads, self.head_size),
                             dtype=query.dtype,
                             device=query.device)

        for seq_len in attn_metadata.seq_lens:
            end = start + seq_len
            sub_out = scaled_dot_product_attention(
                query[None, :, start:end, :],
                key[None, :, start:end, :],
                value[None, :, start:end, :],
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)
