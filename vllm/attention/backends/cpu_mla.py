# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata)
from vllm.attention.ops.cpu_mla_decode import mla_decode_kvcache_cpu


class CPUMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "CPU_MLA"

    @staticmethod
    def get_impl_cls() -> Type["CPUMLAImpl"]:
        return CPUMLAImpl


class CPUMLAImpl(MLACommonImpl[MLACommonMetadata]):

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

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                f"{__class__.__name__} does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      f"{__class__.__name__}")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                f"{__class__.__name__} with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        q = torch.cat([q_nope, q_pe], dim=-1)

        # Run MQA
        o = mla_decode_kvcache_cpu(q, kv_c_and_k_pe_cache,
                                   decode_meta.block_tables,
                                   decode_meta.seq_lens_tensor, self.scale)

        return self._v_up_proj_and_o_proj(o)
