# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch

import vllm._custom_ops as ops
from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)

logger = init_logger(__name__)


class CutlassMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["CutlassMLAImpl"]:
        return CutlassMLAImpl


class CutlassMLAImpl(MLACommonImpl[MLACommonMetadata]):

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
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "CutlassMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "CutlassMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "CutlassMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Cutlass MLA not yet supported")

        B = q_nope.shape[0]

        o = torch.empty((B, self.num_heads, self.kv_lora_rank),
                        dtype=q_nope.dtype,
                        device=q_nope.device)

        # Run MLA
        # Clone q_nope and q_pe to make sure strides computation is correct.
        q_nope = q_nope.clone()
        q_pe = q_pe.clone()
        ops.cutlass_mla_decode(o, q_nope, q_pe, kv_c_and_k_pe_cache,
                               attn_metadata.decode.seq_lens,
                               attn_metadata.decode.block_table, self.scale)

        return self._v_up_proj(o)
