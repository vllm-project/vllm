# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
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
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    tile_scheduler_metadata: tuple[torch.Tensor, torch.Tensor]
    num_splits: torch.Tensor


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):

    def __init__(self, runner):
        super().__init__(runner)

        self.num_q_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)

    def _build_decode(self, input_positions: torch.Tensor,
                      block_table: torch.Tensor,
                      seq_lens: torch.Tensor) -> FlashMLADecodeMetadata:
        tile_scheduler_metadata, num_splits = \
            get_mla_metadata(
            seq_lens,
            self.num_q_heads,
            1, # MQA for the decode path
        )

        return FlashMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
        )


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

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.decode.
            tile_scheduler_metadata,
            num_splits=attn_metadata.decode.num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj(o)
