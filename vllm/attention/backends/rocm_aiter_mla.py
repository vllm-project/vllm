# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type, Union

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.backends.utils import (compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.rocm_aiter_mla import (aiter_mla_decode_fwd,
                                               get_aiter_mla_metadata)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder


def is_aiter_mla_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER \
        and envs.VLLM_ROCM_USE_AITER_MLA


class AiterMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> Type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AiterMLAMetadata"]:
        return AiterMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["AiterMLAState"]:
        return AiterMLAState


@dataclass
class AiterMLAMetadata(MLACommonMetadata):
    # The following 5 tensors are for current version of AITER MLA
    block_table_bound: Optional[torch.Tensor] = None
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_lens: Optional[torch.Tensor] = None

    # This is just to make new AITER MLA API work
    # -- MTP support is not added yet.
    qo_indptr: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self):
        prefill_metadata = super().prefill_metadata
        self._cached_prefill_metadata = prefill_metadata

        if prefill_metadata is not None:
            prefill_metadata.paged_kv_indptr = self.paged_kv_indptr
            prefill_metadata.paged_kv_indices = self.paged_kv_indices
            prefill_metadata\
                .paged_kv_last_page_lens = self.paged_kv_last_page_lens
            prefill_metadata.block_table_bound = self.block_table_bound
            prefill_metadata.qo_indptr = self.qo_indptr

            # update the cache
            self._cached_prefill_metadata = self.__class__(
                **prefill_metadata.__dict__)

        return self._cached_prefill_metadata

    @property
    def decode_metadata(self):
        decode_metadata = super().decode_metadata

        self._cached_decode_metadata = decode_metadata

        if decode_metadata is not None:
            decode_metadata.paged_kv_indptr = self.paged_kv_indptr
            decode_metadata.paged_kv_indices = self.paged_kv_indices
            decode_metadata\
                .paged_kv_last_page_lens = self.paged_kv_last_page_lens
            decode_metadata.block_table_bound = self.block_table_bound
            decode_metadata.qo_indptr = self.qo_indptr

            # update the cache
            self._cached_decode_metadata = self.__class__(
                **decode_metadata.__dict__)

        return self._cached_decode_metadata

    def _ops_advance_step(self, num_seqs: int, num_queries: int,
                          block_size: int, input_tokens: torch.Tensor,
                          sampled_token_ids: torch.Tensor,
                          input_positions: torch.Tensor) -> None:

        ops.advance_step_flashinfer(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_last_page_lens=self.paged_kv_last_page_lens,
            block_table_bound=self.block_table_bound)


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        super().__init__(input_builder)
        assert self.runner.model_config.max_model_len == 32768,\
                "AITER MLA requires max model len to be set to 32768"
        assert self.block_size == 1, "AITER MLA requires only block size 1."

    def prepare(self):
        super().prepare()
        self.paged_kv_indices: list[int] = []
        self.paged_kv_indptr: list[int] = [0]
        self.paged_kv_last_page_lens: list[int] = []
        self.total_blocks = 0
        self.qo_indptr: list[int] = [0]

    def _add_seq_group(self, inter_data, chunked_prefill_enabled: bool,
                       prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)
            if is_profile_run:
                return

            # Update paged_kv_* tensors only for non-profile run
            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: list[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = seq_len // self.block_size + 1 \
            if seq_len % self.block_size != 0 \
            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)
        self.qo_indptr.append(self.qo_indptr[-1] + 1)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_lens.append(last_page_len)

    def build(self, seq_lens: list[int], query_lens: list[int],
              cuda_graph_pad_size: int, batch_size: int) -> AiterMLAMetadata:
        metadata = super().build(seq_lens, query_lens, cuda_graph_pad_size,
                                 batch_size)
        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        if use_captured_graph:
            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_lens.extend([0] * cuda_graph_pad_size)
            last_qo_indptr = self.qo_indptr[-1]
            self.qo_indptr.extend([last_qo_indptr] * cuda_graph_pad_size)

        # For current version of AITER MLA
        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            self.paged_kv_indices.extend(
                [0] * (self.total_blocks - len(self.paged_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device=device,
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device=device,
                                                  dtype=torch.int)
            paged_kv_last_page_lens_tensor = torch.tensor(
                self.paged_kv_last_page_lens, device=device, dtype=torch.int)
            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device=device,
                                                   dtype=torch.int)

            qo_indptr = torch.tensor(self.qo_indptr,
                                     device=device,
                                     dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_lens_tensor = None
            block_table_bound_tensor = None
            qo_indptr = None

        metadata.paged_kv_indptr = paged_kv_indptr_tensor
        metadata.paged_kv_indices = paged_kv_indices_tensor
        metadata.paged_kv_last_page_lens = paged_kv_last_page_lens_tensor
        metadata.block_table_bound = block_table_bound_tensor
        metadata.qo_indptr = qo_indptr

        return metadata


class AiterMLAState(MLACommonState[AiterMLAMetadata]):

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        kv_indices, kv_indptr, last_page_lens, qo_indptr = \
            get_aiter_mla_metadata(
                max_batch_size=max_batch_size,
                block_size=self.runner.block_size,
                max_block_per_batch=\
                    self.runner.get_max_block_per_batch(),
                device=self.runner.device)
        self._paged_kv_indices_tensor = kv_indices
        self._paged_kv_indptr_tensor = kv_indptr
        self._paged_kv_last_page_lens_tensor = last_page_lens
        self._qo_indptr_tensor = qo_indptr

        with super().graph_capture(max_batch_size):
            yield

        del self._paged_kv_indices_tensor
        del self._paged_kv_indptr_tensor
        del self._paged_kv_last_page_lens_tensor
        del self._qo_indptr_tensor

    def graph_capture_get_metadata_for_batch(
            self,
            batch_size: int,
            is_encoder_decoder_model: bool = False) -> AiterMLAMetadata:

        metadata = super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)

        paged_kv_indptr = self._paged_kv_indptr_tensor[:batch_size + 1]
        paged_kv_indices = self._paged_kv_indices_tensor
        paged_kv_last_page_lens = self._paged_kv_last_page_lens_tensor[:
                                                                       batch_size]
        qo_indptr = self._qo_indptr_tensor[:batch_size + 1]

        metadata.paged_kv_indptr = paged_kv_indptr
        metadata.paged_kv_indices = paged_kv_indices
        metadata.paged_kv_last_page_lens = paged_kv_last_page_lens
        metadata.qo_indptr = qo_indptr

        return metadata

    def get_graph_input_buffers(self,
                                attn_metadata: AiterMLAMetadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = super().get_graph_input_buffers(
            attn_metadata, is_encoder_decoder_model)
        input_buffers[
            'paged_kv_indptr'] = attn_metadata.decode_metadata.paged_kv_indptr
        input_buffers[
            "paged_kv_indices"] = attn_metadata.\
            decode_metadata.paged_kv_indices
        input_buffers[
            "paged_kv_last_page_lens"] = attn_metadata.\
            decode_metadata.paged_kv_last_page_lens
        input_buffers['qo_indptr'] = attn_metadata.qo_indptr

        return input_buffers

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata: AiterMLAMetadata,
                                    is_encoder_decoder_model: bool = False):
        super().prepare_graph_input_buffers(input_buffers, attn_metadata,
                                            is_encoder_decoder_model)

        num_total_blocks = attn_metadata.decode_metadata.paged_kv_indices.shape[
            0]
        input_buffers["paged_kv_indptr"].copy_(
            attn_metadata.decode_metadata.paged_kv_indptr, non_blocking=True)
        input_buffers["paged_kv_indices"][:num_total_blocks].copy_(
            attn_metadata.decode_metadata.paged_kv_indices, non_blocking=True)
        input_buffers["paged_kv_last_page_lens"].copy_(
            attn_metadata.decode_metadata.paged_kv_last_page_lens,
            non_blocking=True)
        input_buffers["qo_indptr"].copy_(
            attn_metadata.decode_metadata.qo_indptr, non_blocking=True)


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
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            softmax_scale: float, return_softmax_lse: bool,
            **kwargs) -> Union[tuple[torch.Tensor, ...], torch.Tensor]:
        output = self.flash_attn_varlen_func(
            q,
            k,
            v,
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

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.empty(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        aiter_mla_decode_fwd(q, kv_buffer, o, self.scale,
                             attn_metadata.qo_indptr,
                             attn_metadata.max_query_len,
                             attn_metadata.paged_kv_indptr,
                             attn_metadata.paged_kv_indices,
                             attn_metadata.paged_kv_last_page_lens)

        return self._v_up_proj(o)
