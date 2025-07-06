# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

import vllm._custom_ops as ops
from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.mla.common import MLACommonImpl, MLACommonState
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.cpu_model_runner import ModelInputForCPUBuilder


class CPUMLABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CPU_MLA"

    @staticmethod
    def get_metadata_cls() -> Type["CPUMLAMetadata"]:
        return CPUMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["CPUMLAMetadataBuilder"]:
        return CPUMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["MLACommonState"]:
        return MLACommonState

    @staticmethod
    def get_impl_cls() -> Type["CPUMLAImpl"]:
        return CPUMLAImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        ops.copy_blocks_mla(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]


@dataclass
class CPUMLAMetadata(TorchSDPAMetadata):
    # New for MLA
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor = None

    # required by MLACommonImpl
    is_profile_run: bool = False


class CPUMLAMetadataBuilder(AttentionMetadataBuilder[CPUMLAMetadata]):

    def __init__(self, input_builder: ModelInputForCPUBuilder) -> None:
        self.chunked_prefill = input_builder.chunked_prefill
        self.input_builder = input_builder
        assert not self.chunked_prefill, \
            "chunked prefill is currently not supported"

    def prepare(self):
        self.input_data = self.input_builder.input_data

    def build(self, seq_lens, query_lens, cuda_graph_pad_size, batch_size):
        input_data = self.input_data
        prefill_seq_lens = seq_lens[0:input_data.num_prefills]
        prefill_query_lens = query_lens[0:input_data.num_prefills]
        slot_mapping = torch.tensor(input_data.slot_mapping,
                                    dtype=torch.long,
                                    device="cpu")

        # metadata for prefill
        if input_data.num_prefills > 0:
            query_lens_tensor = torch.tensor(prefill_query_lens,
                                             dtype=torch.int32,
                                             device="cpu")
            kv_lens_tensor = torch.tensor(prefill_seq_lens,
                                          dtype=torch.int32,
                                          device="cpu")
            query_start_loc = torch.zeros(input_data.num_prefills + 1,
                                          dtype=torch.int32,
                                          device="cpu")
            kv_start_loc = torch.zeros(input_data.num_prefills + 1,
                                       dtype=torch.int32,
                                       device="cpu")
            torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=query_start_loc[1:])
            torch.cumsum(kv_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=kv_start_loc[1:])
            max_query_len = max(prefill_query_lens)
            max_kv_len = max(prefill_seq_lens)

            # for chunked-prefill
            if self.chunked_prefill:
                prefill_block_tables = make_tensor_with_pad(
                    self.input_data.prefill_block_tables,
                    pad=0,
                    dtype=torch.int32,
                    device="cpu",
                )
            else:
                prefill_block_tables = None

        else:
            query_start_loc = None
            kv_start_loc = None
            max_query_len = None
            max_kv_len = None
            prefill_block_tables = None

        # metadata for decode
        if input_data.num_decode_tokens != 0:
            seq_lens_tensor = torch.tensor(
                input_data.seq_lens[input_data.num_prefills:],
                dtype=torch.int32,
                device="cpu",
            )
            block_tables = make_tensor_with_pad(
                self.input_data.decode_block_tables,
                pad=0,
                dtype=torch.int32,
                device="cpu",
            )
        else:
            block_tables = torch.tensor([])
            seq_lens_tensor = torch.tensor(
                input_data.seq_lens[:input_data.num_prefills],
                dtype=torch.int32,
                device="cpu",
            )

        # For multi-modal models
        placeholder_index_maps = None
        if len(input_data.multi_modal_inputs_list) != 0:
            placeholder_index_maps = {
                modality: placeholder_map.index_map()
                for modality, placeholder_map in
                input_data.multi_modal_placeholder_maps.items()
            }

        return CPUMLAMetadata(
            chunked_prefill=self.chunked_prefill,
            seq_lens=prefill_seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_kv_len=max_kv_len,
            prefill_query_start_loc=query_start_loc,
            kv_start_loc=kv_start_loc,
            max_decode_seq_len=input_data.max_decode_seq_len,
            num_prefills=input_data.num_prefills,
            num_prefill_tokens=input_data.num_prefill_tokens,
            num_decode_tokens=input_data.num_decode_tokens,
            block_tables=block_tables,
            prefill_block_tables=prefill_block_tables,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            input_positions=torch.tensor([self.input_data.input_positions]))


class CPUMLAImpl(MLACommonImpl[CPUMLAMetadata]):

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
                "CPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "CPUMLAImpl")

        # states is implemented.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "CPUMLAImpl with FP8 KV cache not yet supported")

    def _forward_prefill(
            self,
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_and_k_pe_cache: torch.Tensor,
            attn_metadata: CPUMLAMetadata,  # type: ignore[override]
    ) -> torch.Tensor:

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        output = torch.empty_like(q)
        ipex_ops.varlen_attention(
            query=q,
            key=k,
            value=v_padded,
            out=output,
            seqlen_q=prefill_metadata.prefill_query_start_loc,
            seqlen_k=prefill_metadata.prefill_query_start_loc,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=prefill_metadata.max_query_len,
            pdropout=0.0,
            softmax_scale=self.scale,
            zero_tensors=False,
            is_causal=True,
            return_softmax=False,
            gen_=None,
            logits_soft_cap=0.0,
            window_size_left=-1,
            window_size_right=-1,
            alibi_slopes=None,
        )

        # remove padding
        output = output.view(-1, self.num_heads,
                             q.shape[-1])[..., :v.shape[-1]]
        return output.reshape(-1, self.num_heads * v.shape[-1])

    def _forward_decode(
            self,
            q_nope: torch.Tensor,
            q_pe: torch.Tensor,
            kv_c_and_k_pe_cache: torch.Tensor,
            attn_metadata: CPUMLAMetadata,  # type: ignore[override]
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = q.new_empty(q.shape[0], self.num_heads, self.kv_lora_rank)

        # Run MQA
        ops.mla_decode_kvcache_cpu(o, q, kv_c_and_k_pe_cache, self.scale,
                                   decode_meta.block_tables,
                                   decode_meta.seq_lens_tensor)
        return self._v_up_proj(o)
