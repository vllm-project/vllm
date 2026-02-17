# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Common implementation for sparse MLA attention prefill (forward_mha).

Parallel to MLACommonImpl (in mla_attention.py) which provides forward_mha for
non-sparse MLA backends, this module provides SparseMLACommonImpl which gives
all sparse MLA backends a shared forward_mha implementation.

Sparse MLA prefill uses a two-phase approach:
  Phase 1: Dense causal self-attention on new prefill tokens (FA4).
  Phase 2: Sparse cross-attention on topk cached entries — gather compressed
           KV from the paged cache, decompress via kv_b_proj, attend with FA4.
  Phase 3: Merge the two outputs using LSE-based merging.

TODO: Replace the two-phase approach with a single-pass FA4 masked MHA kernel
once the upstream FlashAttention PR lands.

Decode (forward_mqa) is left abstract for each sparse backend to implement
with its own sparse decode kernel.
"""

import functools
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.v1.attention.backend import (
    AttentionMetadata,
    SparseMLAAttentionImpl,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import ColumnParallelLinear

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


class SparseMLACommonImpl(SparseMLAAttentionImpl[T], Generic[T]):
    """Common sparse MLA implementation providing forward_mha for prefill.

    Subclasses must implement forward_mqa() for decode.
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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = getattr(
            indexer, "topk_indices_buffer", None
        )

        # Set up flash_attn_varlen_func with the appropriate FA version.
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
        if self.vllm_flash_attn_version is not None:
            self._flash_attn = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )
        else:
            self._flash_attn = flash_attn_varlen_func

        # DCP (context parallelism) — lazily initialized by the caller.
        self.dcp_world_size: int = -1
        self.cp_kv_cache_interleave_size: int = (
            get_current_vllm_config().parallel_config.cp_kv_cache_interleave_size
        )

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate k_nope and k_pe along the last dimension."""
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _flash_attn_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        **kwargs,
    ):
        """Call flash_attn_varlen_func handling different Q/V head dimensions."""
        kwargs["return_softmax_lse"] = return_softmax_lse
        if vllm_is_batch_invariant():
            kwargs["num_splits"] = 1

        attn_out = self._flash_attn(
            q=q,
            k=k,
            v=v,
            softmax_scale=self.scale,
            **kwargs,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Sparse MLA forward_mha with FP8 KV cache not yet supported"
            )

        # --- Decompress new tokens' KV ---
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        has_context: bool = getattr(attn_metadata, "has_context", False)
        prefill_qsl = getattr(attn_metadata, "prefill_query_start_loc", None)
        prefill_max_ql: int = getattr(attn_metadata, "prefill_max_query_len", 0)

        assert prefill_qsl is not None, (
            "Metadata must provide prefill_query_start_loc for forward_mha"
        )

        # --- Phase 1: Self-attention on new tokens (causal) ---
        suffix_result = self._flash_attn_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill_qsl,
            cu_seqlens_k=prefill_qsl,
            max_seqlen_q=prefill_max_ql,
            max_seqlen_k=prefill_max_ql,
            causal=True,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = suffix_result

            # --- Phase 2: Sparse context attention on topk cached entries ---
            if self.dcp_world_size > 1:
                context_output, context_lse = self._dcp_compute_sparse_context(
                    q, kv_c_and_k_pe_cache, attn_metadata
                )
            else:
                context_output, context_lse = self._compute_sparse_context(
                    q, kv_c_and_k_pe_cache, attn_metadata
                )

            # Trim to v_head_dim if FA padded the output
            suffix_output = suffix_output[..., : self.v_head_dim]
            context_output = context_output[..., : self.v_head_dim]

            # --- Phase 3: Merge ---
            output_view = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output_view,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
        else:
            if isinstance(suffix_result, tuple):
                suffix_result = suffix_result[0]
            suffix_result = suffix_result[..., : self.v_head_dim]
            output.copy_(suffix_result.flatten(start_dim=-2))

    def _compute_sparse_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse context attention: gather topk entries, decompress, attend.

        Each prefill token attends to its topk cached KV entries. Indices come
        from topk_indices_buffer (populated by the Indexer).

        TODO: Replace with single-pass FA4 masked MHA kernel when available.
        """
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        num_decode_tokens: int = getattr(attn_metadata, "num_decode_tokens", 0)
        topk: int = getattr(attn_metadata, "topk_tokens", 2048)
        num_prefill_tokens = q.shape[0]

        # Get topk indices for prefill tokens
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + num_prefill_tokens
        ]

        # Convert per-request logical indices to global cache slot addresses
        req_id_per_token = attn_metadata.req_id_per_token[num_decode_tokens:]  # type: ignore[attr-defined]
        global_indices = triton_convert_req_index_to_global_index(
            req_id_per_token,
            attn_metadata.block_table,  # type: ignore[attr-defined]
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,  # type: ignore[attr-defined]
            NUM_TOPK_TOKENS=topk,
        )

        # Gather compressed KV from paged cache
        # kv_c_and_k_pe_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        cache_flat = kv_c_and_k_pe_cache.view(-1, kv_c_and_k_pe_cache.shape[-1])
        # global_indices: [num_prefill_tokens, topk]
        valid_mask = global_indices >= 0
        safe_indices = global_indices.clamp(min=0)
        # [num_prefill_tokens, topk, kv_lora_rank + qk_rope_head_dim]
        gathered_kv = cache_flat[safe_indices.long()]
        gathered_kv[~valid_mask] = 0

        # Split into kv_c and k_pe
        kv_c = gathered_kv[..., : self.kv_lora_rank]
        k_pe_ctx = gathered_kv[..., self.kv_lora_rank :]

        # Decompress via kv_b_proj: kv_c -> k_nope, v
        kv_c_flat = kv_c.reshape(-1, self.kv_lora_rank)
        kv_nope_flat = self.kv_b_proj(kv_c_flat)[0]
        kv_nope = kv_nope_flat.view(
            num_prefill_tokens,
            topk,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_ctx, v_ctx = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Build full K: concat k_nope with k_pe (broadcast over heads)
        # k_pe_ctx: [num_prefill_tokens, topk, qk_rope_head_dim]
        k_pe_ctx_expanded = k_pe_ctx.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_ctx = torch.cat([k_nope_ctx, k_pe_ctx_expanded], dim=-1)

        # Flatten for varlen: each prefill token is a separate "batch"
        # with seqlen_q=1 and seqlen_k=topk
        k_flat = k_ctx.reshape(
            num_prefill_tokens * topk, self.num_heads, self.qk_head_dim
        )
        v_flat = v_ctx.reshape(
            num_prefill_tokens * topk, self.num_heads, self.v_head_dim
        )

        cu_seqlens_q = torch.arange(
            0,
            num_prefill_tokens + 1,
            device=q.device,
            dtype=torch.int32,
        )
        cu_seqlens_k = torch.arange(
            0,
            num_prefill_tokens * topk + 1,
            step=topk,
            device=q.device,
            dtype=torch.int32,
        )

        context_output, context_lse = self._flash_attn_diff_headdims(
            q=q,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            causal=False,
            return_softmax_lse=True,
        )

        return context_output, context_lse

    def _dcp_compute_sparse_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse context attention with DCP (context parallelism).

        With DCP, the KV cache is distributed across ranks via interleaving.
        Each rank gathers its local subset of topk entries from its local
        cache, then all-reduces the gathered compressed KV to reconstruct
        the complete topk set. Since each cache entry lives on exactly one
        rank (non-zero there, zero on others), the sum yields the correct
        values. After reconstruction, decompression and attention proceed
        as in the non-DCP path.
        """
        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        dcp_group = get_dcp_group()
        dcp_rank = dcp_group.rank_in_group

        num_decode_tokens: int = getattr(attn_metadata, "num_decode_tokens", 0)
        topk: int = getattr(attn_metadata, "topk_tokens", 2048)
        num_prefill_tokens = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + num_prefill_tokens
        ]

        # Convert per-request logical indices to global cache slot addresses
        req_id_per_token = attn_metadata.req_id_per_token[num_decode_tokens:]  # type: ignore[attr-defined]
        global_indices = triton_convert_req_index_to_global_index(
            req_id_per_token,
            attn_metadata.block_table,  # type: ignore[attr-defined]
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,  # type: ignore[attr-defined]
            NUM_TOPK_TOKENS=topk,
        )

        # Mask out entries not on this DCP rank.
        # With interleaving, position p is on rank
        # (p // interleave_size) % world_size.
        owning_rank = (
            topk_indices // self.cp_kv_cache_interleave_size
        ) % self.dcp_world_size
        global_indices[owning_rank != dcp_rank] = -1

        # Gather from local cache (invalid entries → 0)
        cache_flat = kv_c_and_k_pe_cache.view(-1, kv_c_and_k_pe_cache.shape[-1])
        valid_mask = global_indices >= 0
        safe_indices = global_indices.clamp(min=0)
        gathered_kv = cache_flat[safe_indices.long()]
        gathered_kv[~valid_mask] = 0

        # All-reduce to reconstruct complete topk entries across DCP ranks.
        # Each entry is non-zero on exactly one rank, so sum is correct.
        gathered_kv = dcp_group.all_reduce(gathered_kv)

        # From here, identical to the non-DCP path: decompress and attend.
        kv_c = gathered_kv[..., : self.kv_lora_rank]
        k_pe_ctx = gathered_kv[..., self.kv_lora_rank :]

        kv_c_flat = kv_c.reshape(-1, self.kv_lora_rank)
        kv_nope_flat = self.kv_b_proj(kv_c_flat)[0]
        kv_nope = kv_nope_flat.view(
            num_prefill_tokens,
            topk,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_ctx, v_ctx = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_pe_ctx_expanded = k_pe_ctx.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_ctx = torch.cat([k_nope_ctx, k_pe_ctx_expanded], dim=-1)

        k_flat = k_ctx.reshape(
            num_prefill_tokens * topk, self.num_heads, self.qk_head_dim
        )
        v_flat = v_ctx.reshape(
            num_prefill_tokens * topk, self.num_heads, self.v_head_dim
        )

        cu_seqlens_q = torch.arange(
            0,
            num_prefill_tokens + 1,
            device=q.device,
            dtype=torch.int32,
        )
        cu_seqlens_k = torch.arange(
            0,
            num_prefill_tokens * topk + 1,
            step=topk,
            device=q.device,
            dtype=torch.int32,
        )

        context_output, context_lse = self._flash_attn_diff_headdims(
            q=q,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            causal=False,
            return_softmax_lse=True,
        )

        return context_output, context_lse
