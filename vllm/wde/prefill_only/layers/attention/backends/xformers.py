from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         BlockDiagonalMask)

from vllm.logger import init_logger
from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.prefill_only.layers.attention.backends.abstract import (
    PrefillOnlyAttentionBackend, PrefillOnlyAttentionImpl,
    PrefillOnlyAttentionMetadata, PrefillOnlyAttentionMetadataBuilder)

logger = init_logger(__name__)


class PrefillOnlyXFormersBackend(PrefillOnlyAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "xformers"

    @staticmethod
    def get_impl_cls() -> Type["PrefillOnlyXFormersImpl"]:
        return PrefillOnlyXFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type["PrefillOnlyAttentionMetadata"]:
        return PrefillOnlyXFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type["PrefillOnlyXFormersMetadataBuilder"]:
        return PrefillOnlyXFormersMetadataBuilder


@dataclass
class PrefillOnlyXFormersMetadata(PrefillOnlyAttentionMetadata):
    pass


class PrefillOnlyXFormersMetadataBuilder(
        PrefillOnlyAttentionMetadataBuilder[PrefillOnlyXFormersMetadata]):
    pass


class PrefillOnlyXFormersImpl(PrefillOnlyAttentionImpl):

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
                "XFormers does not support block-sparse attention.")
        if logits_soft_cap is not None:
            raise ValueError(
                "XFormers does not support attention logits soft capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        assert self.alibi_slopes is None
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PrefillOnlyXFormersMetadata,
        kv_cache: Optional[torch.Tensor] = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:

        if attn_type == AttentionType.ENCODER:
            causal = False
        elif attn_type == AttentionType.DECODER:
            causal = True
        else:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PrefillOnlyXFormersImpl")
        original_query = query

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])

        if causal:
            attn_bias = BlockDiagonalCausalMask.from_seqlens(
                attn_metadata.seq_lens)
        else:
            attn_bias = BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens)

        # Add the batch dimension.
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        out = xops.memory_efficient_attention_forward(query,
                                                      key,
                                                      value,
                                                      p=0.0,
                                                      attn_bias=attn_bias,
                                                      scale=self.scale)
        return out.view_as(original_query)
