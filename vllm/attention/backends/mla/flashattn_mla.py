# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata)
from vllm.vllm_flash_attn.fa_utils import flash_attn_supports_mla
from vllm.vllm_flash_attn import flash_attn_varlen_func

if TYPE_CHECKING:
    pass


class FlashAttnMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHATTN_MLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttnMLAImpl"]:
        return FlashAttnMLAImpl


class FlashAttnMLAImpl(MLACommonImpl[MLACommonMetadata]):

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

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        kv_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        o = flash_attn_varlen_func(
            q=q_pe,
            k=kv_pe_cache.unsqueeze(-2),  # Add head dim of 1
            v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
            q_v=q_nope,
            max_seqlen_q=decode_meta.max_decode_query_len,
            cu_seqlens_q=decode_meta.query_start_loc,
            max_seqlen_k=decode_meta.max_decode_seq_len,
            seqused_k=decode_meta.seq_lens_tensor,
            block_table=decode_meta.block_tables,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3  # only version 3 is supported
        )

        return self._v_up_proj_and_o_proj(o)
