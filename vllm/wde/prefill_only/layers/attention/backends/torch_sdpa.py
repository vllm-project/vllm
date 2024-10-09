from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.utils import is_pin_memory_available
from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.prefill_only.layers.attention.backends.abstract import (
    PrefillOnlyAttentionBackend, PrefillOnlyAttentionImpl,
    PrefillOnlyAttentionMetadata, PrefillOnlyAttentionMetadataBuilder)

pin_memory = is_pin_memory_available()


class PrefillOnlyTorchSDPABackend(PrefillOnlyAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "torch_sdpa"

    @staticmethod
    def get_impl_cls() -> Type["PrefillOnlyTorchSDPABackendImpl"]:
        return PrefillOnlyTorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PrefillOnlyTorchSDPAMetadata"]:
        return PrefillOnlyTorchSDPAMetadata

    @staticmethod
    def get_builder_cls() -> Type["PrefillOnlyAttentionMetadataBuilder"]:
        return PrefillOnlyTorchSDPAMetadataBuilder


@dataclass
class PrefillOnlyTorchSDPAMetadata(PrefillOnlyAttentionMetadata):
    pass


class PrefillOnlyTorchSDPAMetadataBuilder(
        PrefillOnlyAttentionMetadataBuilder[PrefillOnlyTorchSDPAMetadata]):
    pass


class PrefillOnlyTorchSDPABackendImpl(PrefillOnlyAttentionImpl):

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

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PrefillOnlyTorchSDPAMetadata,
        kv_cache: Optional[torch.Tensor] = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in TorchSDPA.")

        if attn_type == AttentionType.ENCODER:
            causal = False
        elif attn_type == AttentionType.DECODER:
            causal = True
        else:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PrefillOnlyTorchSDPABackendImpl")

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
                is_causal=causal,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)
