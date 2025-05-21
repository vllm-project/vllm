# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

import vllm.envs as envs
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_fwd
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

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


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):

    def __init__(self, runner, kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        super().__init__(runner, kv_cache_spec, block_table)
        max_model_len = self.runner.model_config.max_model_len
        assert max_model_len == 32768,\
            "AITER MLA requires max_model_len=32768"
        assert self.kv_cache_spec.block_size == 1, "AITER MLA" \
            "only supports block size 1."

    def _get_paged_kv_tensors(
            self, block_table: torch.Tensor,
            seq_lens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        page_size = self.kv_cache_spec.block_size
        block_table_bounds = (seq_lens + page_size - 1) // page_size

        mask = (torch.arange(block_table.size(1),
                             dtype=block_table.dtype,
                             device=block_table.device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table[mask]

        paged_kv_indptr = torch.cat([
            torch.zeros(1,
                        dtype=block_table_bounds.dtype,
                        device=block_table_bounds.device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)
        ])

        paged_kv_last_page_len = seq_lens % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)
        return (
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_len,
        )

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor) -> AiterMLADecodeMetadata:

        (
            paged_kv_indices,
            paged_kv_indptr,
            paged_last_page_len,
        ) = self._get_paged_kv_tensors(block_table_tensor, seq_lens)

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_last_page_len)

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
            blocksparse_params: Optional[dict[str, Any]],
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
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

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

        aiter_mla_decode_fwd(q, kv_buffer, o, self.scale,
                             attn_metadata.decode.paged_kv_indptr,
                             attn_metadata.decode.paged_kv_indices,
                             attn_metadata.decode.paged_kv_last_page_len)

        return self._v_up_proj(o)
