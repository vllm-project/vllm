# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm import envs
from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.attention.ops.triton_flash_attention import triton_attention
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)

logger = init_logger(__name__)


class TritonMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported")

        self.use_triton_flash_attn = envs.VLLM_USE_TRITON_FLASH_ATTN
        self.triton_fa_func = triton_attention if HAS_TRITON else None

    def _flash_attn_varlen_diff_headdims_rocm(self,
                                              q,
                                              k,
                                              v,
                                              softmax_scale=None,
                                              **kwargs):
        assert self.triton_fa_func is not None

        # Triton Attention requires a padded V
        padded_v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)
        # The output of triton_attention is a tuple of
        # [output_tensor, encoded_softmax] where encoded_softmax is always None
        output_tensor, _ = self.triton_fa_func(
            q,
            k,
            padded_v,
            None,  # output
            kwargs["cu_seqlens_q"],
            kwargs["cu_seqlens_k"],
            kwargs["max_seqlen_q"],
            kwargs["max_seqlen_k"],
            kwargs["causal"],
            softmax_scale,
            None,  # bias
        )

        return output_tensor

    def _flash_attn_varlen_diff_headdims(self,
                                         q,
                                         k,
                                         v,
                                         return_softmax_lse=False,
                                         softmax_scale=None,
                                         **kwargs):
        if current_platform.is_rocm() \
            and self.use_triton_flash_attn \
            and not return_softmax_lse:
            return self._flash_attn_varlen_diff_headdims_rocm(
                q, k, v, softmax_scale=softmax_scale, **kwargs)
        else:
            return super()._flash_attn_varlen_diff_headdims(
                q,
                k,
                v,
                return_softmax_lse=return_softmax_lse,
                softmax_scale=softmax_scale,
                **kwargs)

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
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        num_kv_splits = 4  # TODO: heuristic

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
                             attn_metadata.decode.block_table,
                             attn_metadata.decode.seq_lens, attn_logits,
                             num_kv_splits, self.scale, PAGE_SIZE)

        return self._v_up_proj(o)
