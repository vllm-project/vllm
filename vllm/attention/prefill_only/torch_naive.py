import math
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.prefill_only.abstract import (AttentionType,
                                                  PrefillOnlyAttentionBackend,
                                                  PrefillOnlyAttentionImpl,
                                                  PrefillOnlyAttentionMetadata)
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


class PrefillOnlyTorchNAIVEBackend(PrefillOnlyAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "torch_naive"

    @staticmethod
    def get_impl_cls() -> Type["PrefillOnlyTorchNaiveBackendImpl"]:
        return PrefillOnlyTorchNaiveBackendImpl


class PrefillOnlyTorchNaiveBackendImpl(PrefillOnlyAttentionImpl):

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
                "Torch Naive does not support block-sparse attention.")
        if logits_soft_cap is not None:
            raise ValueError("Torch Naive does not support logits soft cap.")
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
                "Torch Naive backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: PrefillOnlyAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:

        assert kv_cache is None

        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in Torch Naive.")

        if attn_type == AttentionType.ENCODER:
            causal = False
        elif attn_type == AttentionType.DECODER:
            causal = True
        else:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PrefillOnlyTorchNaiveBackendImpl")

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
                is_causal=causal,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 is_causal=False,
                                 scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool,
                               device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value
