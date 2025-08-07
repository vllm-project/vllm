# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

import vllm.envs as envs
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_fwd
from vllm.config import VllmConfig
from vllm.utils import cdiv
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

# yapf: enable


def is_aiter_mla_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER \
        and envs.VLLM_ROCM_USE_AITER_MLA


class AiterMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_metadata_cls() -> type["AiterMLAMetadata"]:
        return AiterMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: Optional[torch.Tensor] = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    attn_cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.PURE_DECODE_ONLY

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         AiterMLAMetadata)
        assert self.kv_cache_spec.block_size == 1, "AITER MLA" \
            "only supports block size 1."

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(vllm_config.model_config.max_model_len,
                                     self.kv_cache_spec.block_size)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        if vllm_config.compilation_config.full_cuda_graph:
            self.paged_kv_indptr = torch.zeros(max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device=device)
            self.paged_kv_indices = torch.zeros(max_num_pages,
                                                dtype=torch.int32,
                                                device=device)
            self.paged_kv_last_page_len = torch.zeros(max_num_reqs,
                                                      dtype=torch.int32,
                                                      device=device)

            self.qo_indptr = torch.arange(0,
                                          max_num_reqs + 1,
                                          dtype=torch.int32,
                                          device=device)

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor) -> AiterMLADecodeMetadata:
        page_size = self.kv_cache_spec.block_size
        block_table_bounds = (seq_lens + page_size - 1) // page_size
        device = self.device
        num_reqs = seq_lens.size(0)

        mask = (torch.arange(block_table_tensor.size(1),
                             dtype=block_table_tensor.dtype,
                             device=device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table_tensor[mask]

        paged_kv_last_page_len = seq_lens % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)

        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=block_table_bounds.dtype, device=device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)
        ])

        if self.compilation_config.full_cuda_graph:

            num_actual_pages = paged_kv_indices.size(0)

            self.paged_kv_indices[:num_actual_pages].copy_(paged_kv_indices,
                                                           non_blocking=True)
            self.paged_kv_indices[num_actual_pages:].fill_(-1)
            paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

            self.paged_kv_indptr[:1 + num_reqs].copy_(paged_kv_indptr,
                                                      non_blocking=True)
            self.paged_kv_indptr[1 + num_reqs:].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[:1 + num_reqs]

            self.paged_kv_last_page_len[:num_reqs].copy_(
                paged_kv_last_page_len, non_blocking=True)
            self.paged_kv_last_page_len[num_reqs:].fill_(1)
            paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

            qo_indptr = self.qo_indptr[:1 + num_reqs]

        else:
            qo_indptr = torch.arange(0,
                                     num_reqs + 1,
                                     step=1,
                                     dtype=torch.int32,
                                     device=device)

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr)

        return attn_metadata


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):

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
        assert (num_heads == 16 or num_heads == 128), (
            f"Aiter MLA only supports 16 or 128 number of heads.\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value.")
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        from aiter import flash_attn_varlen_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(self,
                                         q,
                                         k,
                                         v,
                                         return_softmax_lse=False,
                                         softmax_scale=None,
                                         **kwargs):
        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # max_seqlen_qo must be 1 except for MTP
        # TODO: Find the best value for MTP
        max_seqlen_qo = 1
        aiter_mla_decode_fwd(q, kv_buffer, o, self.scale,
                             attn_metadata.decode.qo_indptr, max_seqlen_qo,
                             attn_metadata.decode.paged_kv_indptr,
                             attn_metadata.decode.paged_kv_indices,
                             attn_metadata.decode.paged_kv_last_page_len)

        return self._v_up_proj(o)
