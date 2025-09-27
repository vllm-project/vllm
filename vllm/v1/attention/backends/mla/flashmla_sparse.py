# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Union, ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer, AttentionMetadata
from vllm.attention.ops.flashmla import flash_mla_sparse_prefill
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec
import triton
import triton.language as tl
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

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
    def __init__(self, block_table: torch.Tensor,
                 req_id_per_token: torch.Tensor):
        pass


class FlashMLASparseDecodeMetadata(MLACommonDecodeMetadata):

    def __init__(self):
        pass


@dataclass
class FlashMLASparseMetadata:
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    # For now just create topk_indices that just attend to the first topk tokens
    # always to enable development
    debug_topk_indices: Optional[torch.Tensor] = None


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = block_id < max_num_blocks_per_req
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # If token == -1 OR block_id OOB, output -1; else base * BLOCK_SIZE + offset
    out_val = tl.where(is_invalid_tok | (~valid_block), -1,
                       base * BLOCK_SIZE + inblock_off)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table: torch.
    Tensor,  # int32 [num_requests, max_num_blocks_per_req]
        token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
        BLOCK_SIZE: int = 64,
        NUM_TOPK_TOKENS: int = 2048,
        BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id], token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, \
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out


@dataclass
class FlashMLASparseMetadataBuilder(
        MLACommonMetadataBuilder[FlashMLASparseMetadata]):

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         FlashMLASparseMetadata)
        self.topk_tokens = vllm_config.model_config.hf_config\
            .attn_module_list_cfg[0]["topk_tokens"]
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        self.reorder_batch_threshold += self.num_speculative_tokens

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
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        starts = np.asarray(common_attn_metadata.query_start_loc_cpu,
                            dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths)

        # pos = np.arange(starts[-1]) - np.repeat(starts[:-1], np.diff(starts))
        # seq_lengths = np.asarray(common_attn_metadata.seq_lens_cpu,
        #                          dtype=np.int32)
        # prefix_length = seq_lengths - seg_lengths
        # prefix_length_per_token = np.repeat(prefix_length, seg_lengths)
        # pos = pos + prefix_length_per_token
        # pos_gpu = torch.as_tensor(pos, device=self.device, dtype=torch.long)
        # row = torch.arange(self.topk_tokens,
        #                    device=self.device,
        #                    dtype=torch.int32)
        # debug_topk_indices = row.repeat(num_actual_tokens, 1)
        # mask = debug_topk_indices <= pos_gpu.unsqueeze(1)
        # debug_topk_indices = debug_topk_indices.masked_fill(~mask, -1)
        debug_topk_indices = None
        metadata = FlashMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=torch.from_numpy(req_id_per_token).to(
                device='cuda'),
            # num_decodes=num_decodes,
            # num_decode_tokens=num_decode_tokens,
            # num_prefills=num_prefills,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            debug_topk_indices=debug_topk_indices,
        )
        return metadata


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
            topk_indice_buffer: Optional[torch.Tensor] = None,
            indexer: Optional["Indexer"] = None,
            **mla_args) -> None:
        super().__init__(num_heads,
                         head_size,
                         scale,
                         num_kv_heads,
                         alibi_slopes,
                         sliding_window,
                         kv_cache_dtype,
                         logits_soft_cap,
                         attn_type,
                         kv_sharing_target_layer_name,
                         indexer=indexer,
                         **mla_args)
        self.topk_indice_buffer = indexer.topk_indices_buffer

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

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_tokens, ...]
        q = q[:num_actual_tokens, ...]
        k_c_normed = k_c_normed[:num_actual_tokens, ...]
        k_pe = k_pe[:num_actual_tokens, ...]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)

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

        attn_out = self._forward_bf16_kv(ql_nope, q_pe, kv_cache,
                                         attn_metadata, self.scale)

        output[:num_actual_tokens] = self._v_up_proj(
            attn_out[:num_actual_tokens])
        return output_padded

    def _forward_bf16_kv(self, ql_nope: torch.Tensor, q_pe: torch.Tensor,
                         kv_c_and_k_pe_cache: torch.Tensor,
                         attn_metadata: FlashMLASparseMetadata,
                         k_scale: torch.Tensor) -> torch.Tensor:
        topk_indices = self.topk_indice_buffer
        num_tokens = attn_metadata.num_actual_tokens
        q = torch.cat([ql_nope, q_pe], dim=-1)
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1])
        # NOTE(Chen): kernel requires num_local_head to be a multiple of 64.
        if self.num_heads % 64 != 0:
            assert 64 % self.num_heads == 0
            logger.warning_once(
                f"padding num_heads to 64 due to sparse attn kernel requirement"
            )
            q_padded = q.new_empty((q.shape[0], 64, q.shape[2]))
            q_padded[:, :self.num_heads, :] = q
            q = q_padded
        # TODO: handle index / kv_cache correctly
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices[:num_tokens],
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )
        topk_indices_global = topk_indices_global.view(num_tokens, 1, -1)
        output = flash_mla_sparse_prefill(q[:num_tokens], kv_c_and_k_pe_cache,
                                          topk_indices_global, k_scale)[0]
        output = output[:, :self.num_heads, :]
        return output

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
        lse = None  #TODO

        # NOTE(Chen): shape is unsure
        return attn_out, lse
