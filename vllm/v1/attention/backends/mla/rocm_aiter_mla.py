# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

import vllm.envs as envs
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_forward
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder,
                                                   MLACommonPrefillMetadata)

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
    paged_kv_last_page_lens: Optional[torch.Tensor] = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):

    def __init__(self, runner):
        super().__init__(runner)
        max_model_len = self.runner.model_config.max_model_len
        assert max_model_len == 32768,\
            "AITER MLA requires max_model_len=32768"
        assert self.runner.block_size == 1, "AITER MLA" \
            "requires only block size 1."

        self.block_size = self.runner.block_size

    def _get_paged_kv_tensors(
            self, block_table: torch.Tensor,
            seq_lens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        paged_kv_indices: list[int] = []
        paged_kv_indptr: list[int] = [0]
        paged_kv_last_page_lens: list[int] = []

        device = self.runner.device
        block_size = self.runner.block_size
        total_blocks = 0
        for idx in range(seq_lens.shape[0]):
            seq_len = seq_lens[idx]
            total_blocks += seq_len
            block_table_list = block_table.tolist()[idx]
            block_table_list.insert(0, 0)
            block_table_bound = seq_len // block_size + 1 \
                if seq_len % block_size != 0 \
                else seq_len // block_size
            paged_kv_indices.extend(block_table_list[:block_table_bound])
            paged_kv_indptr.append(paged_kv_indptr[-1] + block_table_bound)
            last_page_len = seq_len % block_size
            if last_page_len == 0:
                last_page_len = block_size
            paged_kv_last_page_lens.append(last_page_len)

        if self.runner.use_cuda_graph:
            cudagraph_batch_size = self.runner.cudagraph_batch_sizes[-1]
            last_paged_kv_indptr = paged_kv_indptr[-1]
            paged_kv_indptr.extend([last_paged_kv_indptr] *
                                   cudagraph_batch_size)
            paged_kv_last_page_lens.extend([0] * cudagraph_batch_size)

        paged_kv_indices.extend([0] * (total_blocks - len(paged_kv_indices)))
        paged_kv_indices_tensor = torch.tensor(paged_kv_indices,
                                               device=device,
                                               dtype=torch.int)
        paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr,
                                              device=device,
                                              dtype=torch.int)
        paged_last_page_lens_tensor = torch.tensor(paged_kv_last_page_lens,
                                                   device=device,
                                                   dtype=torch.int)
        return (
            paged_kv_indices_tensor,
            paged_kv_indptr_tensor,
            paged_last_page_lens_tensor,
        )

    def _build_decode(self, input_positions: torch.Tensor,
                      block_table: torch.Tensor,
                      seq_lens: torch.Tensor) -> AiterMLADecodeMetadata:

        (
            paged_kv_indices,
            paged_kv_indptr,
            paged_last_page_lens,
        ) = self._get_paged_kv_tensors(block_table, seq_lens)

        return AiterMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_lens=paged_last_page_lens)


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

    def _get_fwd_prefill_attn_output(self, q: torch.Tensor, k: torch.Tensor,
                                     v: torch.Tensor,
                                     kv_c_and_k_pe_cache: torch.Tensor,
                                     attn_metadata: MLACommonMetadata,
                                     has_context: bool) -> torch.Tensor:
        assert attn_metadata.prefill is not None

        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=attn_metadata.prefill.query_start_loc,
            cu_seqlens_k=attn_metadata.prefill.query_start_loc,
            max_seqlen_q=attn_metadata.prefill.max_query_len,
            max_seqlen_k=attn_metadata.prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        return output.reshape(-1, self.num_heads * v.shape[-1])

    def _get_prefill_ctx_attn_output(
        self, index: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        prefill_metadata: MLACommonPrefillMetadata
    ) -> tuple[torch.Tensor, ...]:
        assert prefill_metadata.chunked_context is not None

        return self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill_metadata.query_start_loc,
            cu_seqlens_k=prefill_metadata.chunked_context.cu_seq_lens[index],
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=prefill_metadata.chunked_context.max_seq_lens[index],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_lse=True,
        )

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

        aiter_mla_decode_forward(q, kv_buffer, o, self.scale,
                                 attn_metadata.decode.paged_kv_indptr,
                                 attn_metadata.decode.paged_kv_indices,
                                 attn_metadata.decode.paged_kv_last_page_lens)

        return self._v_up_proj_and_o_proj(o)
