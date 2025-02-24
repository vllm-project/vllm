# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Type, Set, Tuple

import torch

from dataclasses import asdict, dataclass
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.ops.flashmla import (
    is_flashmla_supported,
    flash_mla_with_kvcache,
    get_mla_metadata,
)


class FlashMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashMLAImpl"]:
        return FlashMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["FlashMLAMetadata"]:
        return FlashMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashMLAState"]:
        return FlashMLAState

@dataclass
class FlashMLAMetadata(MLACommonMetadata):
    decode_tile_scheduler_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    decode_num_splits: Optional[torch.Tensor] = None

    _cached_decode_metadata: Optional["MLACommonMetadata"] = None

    @property
    def decode_metadata(self):
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata

        common_decode_metadata = super().decode_metadata
        self._cached_decode_metadata = FlashMLAMetadata(
            # TODO: cached but can this be faster?
            **asdict(common_decode_metadata),
            decode_tile_scheduler_metadata=self.decode_tile_scheduler_metadata,
            decode_num_splits=self.decode_num_splits,
        )
        return self._cached_decode_metadata


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        common_metadata = super().build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        decode_tile_scheduler_metadata, decode_num_splits = None, None
        if common_metadata.num_decode_tokens > 0:
            decode_tile_scheduler_metadata, decode_num_splits = get_mla_metadata(
                common_metadata.seq_lens_tensor[common_metadata.num_prefills:],
                self.runner.model_config.get_num_attention_heads(
                    self.runner.parallel_config),
                1,   
            )

        return FlashMLAMetadata(
            # TODO: not on hotpath but can this be faster?
            **asdict(common_metadata), 
            decode_tile_scheduler_metadata=decode_tile_scheduler_metadata,
            decode_num_splits=decode_num_splits,
        )

class FlashMLAState(MLACommonState):
    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = super().get_graph_input_buffers(attn_metadata,
                                                        is_encoder_decoder_model)
        if attn_metadata.tile_scheduler_metadata is not None:
            tile_scheduler_metadata = attn_metadata.tile_scheduler_metadata
            num_splits = attn_metadata.num_splits
            input_buffers["tile_scheduler_metadata"] = tile_scheduler_metadata
            input_buffers["num_splits"] = num_splits
            
        return input_buffers
    
    

class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
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
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2), # Add head dim of 1
            block_table=decode_meta.block_tables,
            cache_seqlens=decode_meta.seq_lens_tensor,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=decode_meta.decode_tile_scheduler_metadata,
            num_splits=decode_meta.decode_num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj_and_o_proj(o)
