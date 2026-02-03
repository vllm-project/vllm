# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MLA Sparse Attention Backend.

This backend uses the FlashInfer TRT-LLM MLA kernel with sparse_mla_top_k
for models like DeepSeek-V3.2 that use index-based sparse attention.

For sparse MLA:
- block_tables shape changes from [batch_size, max_num_blocks] (dense)
  to [batch_size, q_len_per_request, sparse_mla_top_k] (sparse)
- The sparse indices represent physical cache slot positions to attend to
- sparse_mla_top_k parameter must be set to the topk value
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLASparseBackend(AttentionBackend):
    """FlashInfer MLA backend with sparse attention support.

    This backend uses the FlashInfer TRT-LLM MLA kernel with sparse_mla_top_k
    for models like DeepSeek-V3.2 that use index-based sparse attention.
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLASparseImpl"]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FlashInfer sparse MLA targets Blackwell (SM 10.x)
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # FlashInfer MLA sparse kernel requires qk_nope_head_dim == 128
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim != 128:
                return (
                    f"FlashInfer MLA Sparse kernel requires qk_nope_head_dim == 128, "
                    f"but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


@dataclass
class FlashInferMLASparseDecodeMetadata:
    """Decode-specific metadata for FlashInfer MLA Sparse."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    # Query start locations
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    # Sequence lengths for all requests (context + query)
    seq_lens: torch.Tensor

    # Sparse-specific
    block_size: int = 64
    topk_tokens: int = 2048

    # Decode metadata (None during prefill-only)
    decode: FlashInferMLASparseDecodeMetadata | None = None

    # For split batches
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0


class FlashInferMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        # Treat requests with query length <= 1 as decodes
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        # Build req_id_per_token mapping
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )

        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]

        # Split into decode and prefill
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
                require_uniform=True,
            )
        )

        # Build decode metadata if we have decode tokens
        decode_metadata = None
        if num_decodes > 0:
            decode_block_table = cm.block_table_tensor[:num_decodes]
            decode_seq_lens = cm.seq_lens[:num_decodes]
            decode_metadata = FlashInferMLASparseDecodeMetadata(
                block_table=decode_block_table,
                seq_lens=decode_seq_lens,
            )

        return FlashInferMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            seq_lens=cm.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            decode=decode_metadata,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None


def _get_workspace_buffer(device: torch.device) -> torch.Tensor:
    global _fi_sparse_workspace
    if _fi_sparse_workspace is None:
        _fi_sparse_workspace = torch.zeros(
            FLASHINFER_MLA_SPARSE_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA-specific dimensions
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]

        assert indexer is not None, "Indexer required for sparse MLA"
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

    def _call_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        topk_tokens: int,
    ) -> torch.Tensor:
        """Call the FlashInfer sparse MLA kernel for a batch with uniform q_len.

        Args:
            q: Query tensor of shape (batch_size, q_len, num_heads, head_dim)
            kv_c_and_k_pe_cache: KV cache tensor
            topk_indices: Physical indices of shape (batch_size, q_len, topk)
            seq_lens: Sequence lengths tensor of shape (batch_size,)
            max_seq_len: Maximum sequence length
            topk_tokens: Number of top-k tokens for sparse attention
        """
        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=topk_tokens,
        )
        return o

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for decode with sparse attention.

        For sparse mode:
        - topk_indices contains logical indices per token [num_tokens, topk]
        - These are converted to physical cache slots using the block table
        - The resulting indices are passed as block_tables with shape
          [batch_size, q_len_per_request, sparse_mla_top_k]

        Mixed batches (decode + prefill) are handled by processing each group
        separately, as the kernel requires uniform query lengths within a batch.
        """
        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Convert per-request topk indices to global cache slots
        topk_indices_physical = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        if self.bmm1_scale is None:
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Handle pure decode case (most common, optimized path)
        if num_prefills == 0 and num_decodes > 0:
            assert attn_metadata.decode is not None
            decode_q_len = num_decode_tokens // num_decodes

            q_reshaped = q.view(num_decodes, decode_q_len, q.shape[-2], q.shape[-1])
            indices_reshaped = topk_indices_physical.view(num_decodes, decode_q_len, -1)

            o = self._call_kernel(
                q_reshaped,
                kv_c_and_k_pe_cache,
                indices_reshaped,
                attn_metadata.decode.seq_lens,
                attn_metadata.max_seq_len,
                attn_metadata.topk_tokens,
            )
            return o.view(-1, o.shape[-2], o.shape[-1]), None

        # Handle pure prefill case
        if num_decodes == 0 and num_prefills > 0:
            prefill_q_len = num_prefill_tokens // num_prefills

            q_reshaped = q.view(num_prefills, prefill_q_len, q.shape[-2], q.shape[-1])
            indices_reshaped = topk_indices_physical.view(
                num_prefills, prefill_q_len, -1
            )

            o = self._call_kernel(
                q_reshaped,
                kv_c_and_k_pe_cache,
                indices_reshaped,
                attn_metadata.seq_lens,
                attn_metadata.max_seq_len,
                attn_metadata.topk_tokens,
            )
            return o.view(-1, o.shape[-2], o.shape[-1]), None

        # Handle mixed batch: process decode and prefill separately
        # Output tensor to hold results
        output = q.new_empty(num_actual_toks, self.num_heads, self.kv_lora_rank)

        # Process decode tokens
        if num_decode_tokens > 0:
            assert attn_metadata.decode is not None
            decode_q_len = num_decode_tokens // num_decodes

            decode_q = q[:num_decode_tokens].view(
                num_decodes, decode_q_len, q.shape[-2], q.shape[-1]
            )
            decode_indices = topk_indices_physical[:num_decode_tokens].view(
                num_decodes, decode_q_len, -1
            )

            decode_out = self._call_kernel(
                decode_q,
                kv_c_and_k_pe_cache,
                decode_indices,
                attn_metadata.decode.seq_lens,
                attn_metadata.max_seq_len,
                attn_metadata.topk_tokens,
            )
            output[:num_decode_tokens] = decode_out.view(
                num_decode_tokens, decode_out.shape[-2], decode_out.shape[-1]
            )

        # Process prefill tokens
        if num_prefill_tokens > 0:
            prefill_q_len = num_prefill_tokens // num_prefills

            prefill_q = q[num_decode_tokens:].view(
                num_prefills, prefill_q_len, q.shape[-2], q.shape[-1]
            )
            prefill_indices = topk_indices_physical[num_decode_tokens:].view(
                num_prefills, prefill_q_len, -1
            )

            # Get seq_lens for prefill requests (they come after decode requests)
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]

            prefill_out = self._call_kernel(
                prefill_q,
                kv_c_and_k_pe_cache,
                prefill_indices,
                prefill_seq_lens,
                attn_metadata.max_seq_len,
                attn_metadata.topk_tokens,
            )
            output[num_decode_tokens:] = prefill_out.view(
                num_prefill_tokens, prefill_out.shape[-2], prefill_out.shape[-1]
            )

        return output, None
