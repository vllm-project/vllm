# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

import vllm.envs as envs
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_forward
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)

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
        device = self.runner.device
        block_size = self.runner.block_size
        batch_size = seq_lens.shape[0]

        # Calculate total_blocks and number of blocks per sequence
        total_blocks = seq_lens.sum().item()
        max_blocks = ((seq_lens + block_size - 1) // block_size)
        max_blocks_sum = max_blocks.sum().item()

        # Initialize tensors with precomputed sizes
        paged_kv_indices = torch.zeros(max_blocks_sum,
                                       dtype=torch.int,
                                       device=device)
        paged_kv_indptr = torch.zeros(batch_size + 1,
                                      dtype=torch.int,
                                      device=device)
        paged_kv_last_page_lens = torch.zeros(batch_size,
                                              dtype=torch.int,
                                              device=device)

        current_index = 0
        for idx in range(batch_size):
            seq_len = seq_lens[idx].item()
            block_table_slice = block_table[idx].tolist()
            block_table_slice.insert(0, 0)

            block_table_bound = (seq_len + block_size - 1) // block_size

            # Fill paged_kv_indices
            paged_kv_indices[current_index:current_index +
                             block_table_bound] = torch.tensor(
                                 block_table_slice[:block_table_bound],
                                 dtype=torch.int,
                                 device=device)

            # Fill paged_kv_indptr
            paged_kv_indptr[idx + 1] = paged_kv_indptr[idx] + block_table_bound

            # Fill paged_kv_last_page_lens
            last_page_len = seq_len % block_size
            if last_page_len == 0:
                last_page_len = block_size
            paged_kv_last_page_lens[idx] = last_page_len

            current_index += block_table_bound

        if self.runner.use_cuda_graph:
            cudagraph_batch_size = self.runner.cudagraph_batch_sizes[-1]
            last_paged_kv_indptr = paged_kv_indptr[-1].item()
            paged_kv_indptr = torch.cat([
                paged_kv_indptr,
                paged_kv_indptr.new_full((cudagraph_batch_size, ),
                                         last_paged_kv_indptr)
            ])
            paged_kv_last_page_lens = torch.cat([
                paged_kv_last_page_lens,
                paged_kv_last_page_lens.new_zeros((cudagraph_batch_size, ))
            ])

        pad_size = total_blocks - paged_kv_indices.size(0)
        paged_kv_indices = torch.cat([
            paged_kv_indices,
            torch.zeros(pad_size, dtype=torch.int, device=device)
        ])

        return (
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_lens,
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

    def _flash_attn_varlen_diff_headdims(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            softmax_scale: float,
            return_softmax_lse: bool = False,
            **kwargs) -> Union[tuple[torch.Tensor, ...], torch.Tensor]:
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

        aiter_mla_decode_forward(q, kv_buffer, o, self.scale,
                                 attn_metadata.decode.paged_kv_indptr,
                                 attn_metadata.decode.paged_kv_indices,
                                 attn_metadata.decode.paged_kv_last_page_lens)

        return self._v_up_proj_and_o_proj(o)
