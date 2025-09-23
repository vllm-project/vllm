# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer, AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class FlashMLASparseBackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return FlashMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        print("try running get_supported_dtypes")
        # TODO: verify this
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # TODO: verify this
        return [576]


class MLASparsePrefillMetadata:
    # NOTE(Chen): not call it "FlashMLASparsePrefillMetadata" because
    # the kernel is not from flashmla
    def __init__(self):
        pass


class FlashMLASparseDecodeMetadata(MLACommonDecodeMetadata):

    def __init__(self):
        pass


@dataclass
class FlashMLASparseMetadata(MLACommonMetadata[MLASparsePrefillMetadata]):
    # For now just create topk_indices that just attend to the first topk tokens
    # always to enable development
    debug_topk_indices: Optional[torch.Tensor] = None


@dataclass
class FlashMLASparseMetadataBuilder(
        MLACommonMetadataBuilder[FlashMLASparseMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         FlashMLASparseMetadata)
        self.topk_tokens = vllm_config.model_config.hf_config\
            .attn_module_list_cfg[0]["topk_tokens"]

    def _build_prefill(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> MLASparsePrefillMetadata:
        return MLASparsePrefillMetadata()

    def _build_decode(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashMLASparseDecodeMetadata:
        return FlashMLASparseDecodeMetadata()

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashMLASparseMetadata:
        logger.info("build FlashMLASparseMetadata")
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens =\
            split_decodes_and_prefills(common_attn_metadata,
                                       decode_threshold=self.reorder_batch_threshold)

        starts = np.asarray(common_attn_metadata.query_start_loc_cpu)
        pos = np.arange(starts[-1]) - np.repeat(starts[:-1], np.diff(starts))
        pos_gpu = torch.as_tensor(pos, device=self.device, dtype=torch.long)

        row = torch.arange(self.topk_tokens,
                           device=self.device,
                           dtype=torch.int64)
        debug_topk_indices = row.repeat(num_actual_tokens, 1)
        mask = debug_topk_indices < pos_gpu.unsqueeze(1)
        debug_topk_indices = debug_topk_indices.masked_fill(~mask, -1)

        return FlashMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            debug_topk_indices=debug_topk_indices,
            prefill=self._build_prefill(common_attn_metadata),
            decode=self._build_decode(common_attn_metadata),
        )


@dataclass
class FlashMLASparseImpl(MLACommonImpl[FlashMLASparseMetadata]):

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
        # self.sm_scale =
        self.topk_indices = None

    def set_topk_indices(self, topk_indices: torch.Tensor):
        self.topk_indices = topk_indices

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode (see:
        #  https://vllm-dev.slack.com/archives/C09GKA1D4LR/p1758506094148479)

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for MLACommonImpl")

        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert attn_metadata.num_decodes is not None and \
            attn_metadata.num_prefills is not None and \
            attn_metadata.num_decode_tokens is not None

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

        decode_ql_nope = ql_nope[:num_decode_tokens]
        decode_q_pe = q_pe[:num_decode_tokens]

        prefill_ql_nope = ql_nope[num_decode_tokens:]
        prefill_q_pe = q_pe[num_decode_tokens:]

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if has_prefill:
            attn_out = self._forward_prefill(prefill_ql_nope, prefill_q_pe,
                                             kv_cache, attn_metadata,
                                             layer._k_scale)
            # v_up projection
            output[num_decode_tokens:] = self._v_up_proj(attn_out)
        if has_decode:
            # call decode attn
            attn_out, lse = self._forward_decode(
                (decode_ql_nope, decode_q_pe), kv_cache, attn_metadata, layer)
            # v_up projection
            output[:num_decode_tokens] = self._v_up_proj(attn_out)
        return output_padded

    def _forward_prefill(self, ql_nope: torch.Tensor, q_pe: torch.Tensor,
                         kv_c_and_k_pe_cache: torch.Tensor,
                         attn_metadata: FlashMLASparseMetadata,
                         k_scale: torch.Tensor) -> torch.Tensor:
        # # assume indice of shape [num_prefill_tokens, topk]
        # block_id_in_req = topk_indices // self.block_size
        topk_indices = self.topk_indices[attn_metadata.num_decodes:]
        logger.info("called _forward_prefill with topk_indices shape %s",
                    topk_indices.shape)
        # NOTE(Chen): shape is unsure

        return torch.zeros((ql_nope.shape[0], ql_nope.shape[1], 512),
                           dtype=ql_nope.dtype,
                           device=ql_nope.device)

    def _forward_decode(
            self,
            q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
            kv_c_and_k_pe_cache: torch.Tensor,
            attn_metadata: FlashMLASparseMetadata,
            layer: AttentionLayer,
            topk_indices: Optional[torch.Tensor] = None,  # sparse attn
    ) -> torch.Tensor:

        topk_indices = self.topk_indices[:attn_metadata.num_decodes]

        # # assume indice of shape [num_decode_tokens, topk]
        # block_id_in_req = topk_indices // self.block_size

        logger.info("called _forward_decode with topk_indices shape %s",
                    topk_indices.shape)
        
        ql_nope, q_pe = q
        
        attn_out = torch.zeros((ql_nope.shape[0], ql_nope.shape[1], 512),
                           dtype=ql_nope.dtype,
                           device=ql_nope.device)
        lse = None #TODO
        
        # NOTE(Chen): shape is unsure
        return attn_out, lse
