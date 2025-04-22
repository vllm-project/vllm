# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
from vllm.vllm_flash_attn.fa_utils import (flash_attn_supports_mla,
                                           get_flash_attn_version)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class FlashAttnMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHATTN_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["FlashAttnMLAMetadata"]:
        return FlashAttnMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLAMetadataBuilder"]:
        return FlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl


@dataclass
class FlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: Optional[torch.Tensor] = None


@dataclass
class FlashAttnMLAMetadata(MLACommonMetadata[FlashAttnMLADecodeMetadata]):
    pass


class FlashAttnMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashAttnMLAMetadata]):
    # TODO(lucas): tune this value
    decode_threshold: int = 64

    def __init__(self, runner):
        super().__init__(runner)
        self.fa_aot_schedule = (get_flash_attn_version() == 3)
        self.page_size = self.runner.block_size

    def _schedule_decode(self, num_reqs, cu_query_lens, max_query_len, seqlens,
                         max_seq_len, causal):
        if self.fa_aot_schedule:
            return get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                headdim=self.mla_dims.qk_rope_head_dim,
                headdim_v=self.mla_dims.kv_lora_rank,
                page_size=self.page_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
            )
        return None

    def _build_decode(self, seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      input_positions: torch.Tensor,
                      block_table: torch.Tensor) -> FlashAttnMLADecodeMetadata:

        query_lens_cpu = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])
        max_query_len = query_lens_cpu.max().item()
        max_seq_len = seq_lens_cpu.max().item()

        scheduler_metadata = self._schedule_decode(
            num_reqs=seq_lens_cpu.numel(),
            cu_query_lens=query_start_loc_device,
            max_query_len=max_query_len,
            seqlens=seq_lens_device,
            max_seq_len=max_seq_len,
            causal=True,
        )

        return FlashAttnMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=scheduler_metadata,
        )


class FlashAttnMLAImpl(MLACommonImpl[MLACommonMetadata]):

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

        assert flash_attn_supports_mla(), \
            "FlashAttnMLA is not supported on this device"

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
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 FlashMLA not yet supported")

        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        kv_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        o = flash_attn_varlen_func(
            q=q_pe,
            k=kv_pe_cache.unsqueeze(-2),  # Add head dim of 1
            v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
            q_v=q_nope,
            max_seqlen_q=decode_meta.max_query_len,
            cu_seqlens_q=decode_meta.query_start_loc,
            max_seqlen_k=decode_meta.max_seq_len,
            seqused_k=decode_meta.seq_lens,
            block_table=decode_meta.block_table,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,  # only version 3 is supported
            scheduler_metadata=decode_meta.scheduler_metadata,
        )

        return self._v_up_proj_and_o_proj(o)
