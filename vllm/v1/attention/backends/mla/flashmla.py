# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)

logger = init_logger(__name__)


class FlashMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl


@dataclass
class FlashMLAMetadata(MLACommonMetadata):
    decode_tile_scheduler_metadata: Optional[tuple[torch.Tensor,
                                                   torch.Tensor]] = None
    decode_num_splits: Optional[torch.Tensor] = None


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):

    def __init__(self, runner):
        super().__init__(runner, cls=FlashMLAMetadata)

        self.num_q_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        m = super().build(num_reqs, num_actual_tokens, max_query_len,
                          common_prefix_len)

        if m.num_decode_tokens is not None and m.num_decode_tokens > 0:
            m.decode_tile_scheduler_metadata, m.decode_num_splits = \
                get_mla_metadata(
                m.seq_lens[:m.num_decode_tokens],
                self.num_q_heads,
                1, # MQA for the decode path
            )

        return m


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

        assert is_flashmla_supported(), \
            "FlashMLA is not supported on this device"

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 FlashMLA not yet supported")

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.block_table[:attn_metadata.num_decodes,
                                                  ...],
            cache_seqlens=attn_metadata.seq_lens[:attn_metadata.
                                                 num_decode_tokens],
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.
            decode_tile_scheduler_metadata,
            num_splits=attn_metadata.decode_num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj_and_o_proj(o)
