# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Common implementation for sparse MLA attention prefill (forward_mha).

Parallel to MLACommonImpl (in mla_attention.py) which provides forward_mha for
non-sparse MLA backends, this module provides SparseMLACommonImpl which gives
all sparse MLA backends a shared forward_mha implementation.

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
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import ColumnParallelLinear

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)

# Kernel MMA tile size (fixed by hardware).
_M_BLOCK_SIZE = 128
_N_BLOCK_SIZE = 128


def _build_sparse_causal_mask(
    topk_indices_per_req: list[torch.Tensor],
    ctx_lens: list[int],
    q_lens: list[int],
    max_q_len: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense (B, max_Q, max_S) int32 mask combining topk + causal.

    For each request i, query token j (0-indexed within the request):
      - Context positions (kv_idx < ctx_len_i):
            1 if kv_idx is in topk_indices for that query token, else 0
      - New-token positions (kv_idx >= ctx_len_i):
            1 if (kv_idx - ctx_len_i) <= j (causal self-attention), else 0
    """
    B = len(ctx_lens)
    mask = torch.zeros(B, max_q_len, max_seq_len, dtype=torch.int32, device=device)

    for i in range(B):
        q_len = q_lens[i]
        ctx_len = ctx_lens[i]
        if q_len == 0:
            continue

        # Topk context bits
        if ctx_len > 0:
            req_topk = topk_indices_per_req[i]  # (q_len, topk)
            # Clamp to valid context range; invalid indices become sentinel
            valid = (req_topk >= 0) & (req_topk < ctx_len)
            safe_idx = torch.where(valid, req_topk, torch.zeros_like(req_topk))
            # scatter 1s at valid positions
            mask[i, :q_len].scatter_(
                1,
                safe_idx.long(),
                valid.to(torch.int32),
            )

        # Causal self-attention bits for new tokens
        # New token kv positions are [ctx_len, ctx_len + q_len)
        # Query j attends to kv positions ctx_len..ctx_len+j (inclusive)
        causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.int32, device=device))
        mask[i, :q_len, ctx_len : ctx_len + q_len] = causal

    return mask


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

        # Check if FA4 is available (required for mask_mod in forward_mha
        # with cached context).
        fa4_version = get_flash_attn_version(head_size=qk_head_dim)
        self._fa4_available = fa4_version is not None and fa4_version >= 4

        # Block-sparse tile sizes.  SM100 (Blackwell) has q_stage=2, so the
        # block-sparse tile_m must be >= 2 * m_block_size = 256.
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        q_stage = 2 if (cap is not None and cap.major >= 10) else 1
        self._bs_tile_m = q_stage * _M_BLOCK_SIZE
        self._bs_tile_n = _N_BLOCK_SIZE

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

    # ------------------------------------------------------------------
    # Context gathering helpers
    # ------------------------------------------------------------------

    def _gather_and_decompress_context(
        self,
        kv_c_and_k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        ctx_lens: list[int],
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather all context from paged cache, decompress via kv_b_proj.

        Returns (k_ctx, v_ctx, cu_ctx_lens) packed in varlen format.
          k_ctx: (total_ctx, num_heads, qk_head_dim)
          v_ctx: (total_ctx, num_heads, v_head_dim)
          cu_ctx_lens: (B+1,) int32
        """
        device = kv_c_and_k_pe_cache.device
        B = len(ctx_lens)

        # Build cumulative context lengths
        cu_ctx = torch.zeros(B + 1, dtype=torch.int32, device=device)
        ctx_lens_t = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
        torch.cumsum(ctx_lens_t, dim=0, out=cu_ctx[1:])
        total_ctx = cu_ctx[-1].item()

        if total_ctx == 0:
            empty_k = torch.empty(
                0,
                self.num_heads,
                self.qk_head_dim,
                dtype=kv_c_and_k_pe_cache.dtype,
                device=device,
            )
            empty_v = torch.empty(
                0,
                self.num_heads,
                self.v_head_dim,
                dtype=kv_c_and_k_pe_cache.dtype,
                device=device,
            )
            return empty_k, empty_v, cu_ctx

        # Compute physical slot addresses for all context tokens.
        # cache layout: (num_blocks, block_size, head_size)
        head_size = kv_c_and_k_pe_cache.shape[-1]
        cache_flat = kv_c_and_k_pe_cache.reshape(-1, head_size)

        # Build (total_ctx,) tensors of req_idx and logical position
        req_indices = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int32), ctx_lens_t
        )
        # Logical positions within each request's context
        offsets_within_req = torch.cat(
            [
                torch.arange(cl, device=device, dtype=torch.int32)
                for cl in ctx_lens
                if cl > 0
            ]
        )

        block_ids = offsets_within_req // block_size
        in_block_offsets = offsets_within_req % block_size
        physical_blocks = block_table[req_indices.long(), block_ids.long()]
        physical_slots = physical_blocks.long() * block_size + in_block_offsets.long()

        # Gather from cache
        context_packed = cache_flat[physical_slots]  # (total_ctx, head_size)

        # Split into kv_c and k_pe
        kv_c_ctx = context_packed[..., : self.kv_lora_rank]
        k_pe_ctx = context_packed[..., self.kv_lora_rank :]

        # Decompress via kv_b_proj
        kv_nope_ctx = self.kv_b_proj(kv_c_ctx)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_ctx, v_ctx = kv_nope_ctx.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Broadcast k_pe to all heads and concat with k_nope
        k_pe_ctx = k_pe_ctx.unsqueeze(1).expand(-1, self.num_heads, -1)
        k_ctx = self._concat_k_nope_k_pe(k_nope_ctx, k_pe_ctx)

        return k_ctx, v_ctx, cu_ctx

    # ------------------------------------------------------------------
    # forward_mha — single-pass FA4 masked MHA
    # ------------------------------------------------------------------

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

        # Decompress new tokens' KV
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_new, v_new = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_new = self._concat_k_nope_k_pe(k_nope_new, k_pe)

        prefill_qsl = getattr(attn_metadata, "prefill_query_start_loc", None)
        prefill_max_ql: int = getattr(attn_metadata, "prefill_max_query_len", 0)
        has_context: bool = getattr(attn_metadata, "has_context", False)

        assert prefill_qsl is not None, (
            "Metadata must provide prefill_query_start_loc for forward_mha"
        )

        if not has_context:
            # No cached context — causal self-attention on new tokens only.
            suffix_result = self._flash_attn_diff_headdims(
                q=q,
                k=k_new,
                v=v_new,
                cu_seqlens_q=prefill_qsl,
                cu_seqlens_k=prefill_qsl,
                max_seqlen_q=prefill_max_ql,
                max_seqlen_k=prefill_max_ql,
                causal=True,
            )
            suffix_result = suffix_result[..., : self.v_head_dim]
            output.copy_(suffix_result.flatten(start_dim=-2))
            return

        # ----- Has cached context: single-pass FA4 masked MHA -----
        if not self._fa4_available:
            raise NotImplementedError(
                "Sparse MLA forward_mha with cached context requires FA4 "
                "(SM100+). On SM90, all tokens are routed through forward_mqa."
            )

        device = q.device
        num_decodes: int = getattr(attn_metadata, "num_decodes", 0)
        num_prefills: int = getattr(attn_metadata, "num_prefills", 0)
        block_size: int = getattr(attn_metadata, "block_size", 64)
        num_decode_tokens: int = getattr(attn_metadata, "num_decode_tokens", 0)

        # seq_lens for ALL requests; prefill requests start at num_decodes.
        all_seq_lens = getattr(attn_metadata, "seq_lens", None)
        assert all_seq_lens is not None, (
            "Metadata must provide seq_lens for forward_mha with context"
        )
        prefill_seq_lens = all_seq_lens[num_decodes : num_decodes + num_prefills]

        # Compute per-request query and context lengths
        prefill_qsl_cpu = prefill_qsl.cpu()
        q_lens = [
            (prefill_qsl_cpu[i + 1] - prefill_qsl_cpu[i]).item()
            for i in range(num_prefills)
        ]
        prefill_seq_lens_cpu = prefill_seq_lens.cpu()
        ctx_lens = [
            prefill_seq_lens_cpu[i].item() - q_lens[i] for i in range(num_prefills)
        ]
        seq_lens_full = [ctx_lens[i] + q_lens[i] for i in range(num_prefills)]

        max_seq_len = max(seq_lens_full)
        max_q_len = max(q_lens) if q_lens else 0

        # --- Step 2: Gather + decompress ALL cached context ---
        block_table = getattr(attn_metadata, "block_table", None)
        assert block_table is not None
        prefill_block_table = block_table[num_decodes : num_decodes + num_prefills]

        k_ctx, v_ctx, cu_ctx = self._gather_and_decompress_context(
            kv_c_and_k_pe_cache,
            prefill_block_table,
            ctx_lens,
            block_size,
        )

        # --- Step 3: Build varlen K, V = [k_ctx; k_new] per request ---
        # Pack K and V: for each request, context tokens first, new tokens after.
        total_kv = sum(seq_lens_full)
        k_packed = torch.empty(
            total_kv, self.num_heads, self.qk_head_dim, dtype=q.dtype, device=device
        )
        v_packed = torch.empty(
            total_kv, self.num_heads, self.v_head_dim, dtype=q.dtype, device=device
        )

        cu_seqlens_k = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
        kv_offset = 0
        q_offset = 0
        ctx_offset = 0
        for i in range(num_prefills):
            cl = ctx_lens[i]
            ql = q_lens[i]
            sl = cl + ql
            # Context portion
            if cl > 0:
                k_packed[kv_offset : kv_offset + cl] = k_ctx[
                    ctx_offset : ctx_offset + cl
                ]
                v_packed[kv_offset : kv_offset + cl] = v_ctx[
                    ctx_offset : ctx_offset + cl
                ]
                ctx_offset += cl
            # New-token portion
            k_packed[kv_offset + cl : kv_offset + sl] = k_new[q_offset : q_offset + ql]
            v_packed[kv_offset + cl : kv_offset + sl] = v_new[q_offset : q_offset + ql]
            q_offset += ql
            kv_offset += sl
            cu_seqlens_k[i + 1] = kv_offset

        # --- Step 4: Build combined causal+topk mask ---
        # Get topk indices for prefill tokens
        assert self.topk_indices_buffer is not None
        num_prefill_tokens = q.shape[0]
        topk_all = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + num_prefill_tokens
        ]
        # Split per-request
        topk_per_req = []
        ti_offset = 0
        for i in range(num_prefills):
            ql = q_lens[i]
            topk_per_req.append(topk_all[ti_offset : ti_offset + ql])
            ti_offset += ql

        dense_mask = _build_sparse_causal_mask(
            topk_per_req,
            ctx_lens,
            q_lens,
            max_q_len,
            max_seq_len,
            device,
        )

        # Block sparsity for tile skipping
        from vllm.vllm_flash_attn.cute.topk_mask import (
            dense_mask_to_block_sparse,
            topk_mask_mod,
        )

        block_sparse = dense_mask_to_block_sparse(
            dense_mask,
            max_q_len,
            max_seq_len,
            self._bs_tile_m,
            self._bs_tile_n,
        )

        attn_out, _ = flash_attn_varlen_func(
            q=q,
            k=k_packed,
            v=v_packed,
            cu_seqlens_q=prefill_qsl,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_seq_len,
            softmax_scale=self.scale,
            causal=False,  # mask_mod handles both causal + topk
            return_softmax_lse=True,
            fa_version=4,
            mask_mod=topk_mask_mod,
            aux_tensors=[dense_mask],
            block_sparse_tensors=block_sparse,
            m_block_size=_M_BLOCK_SIZE,
            n_block_size=_N_BLOCK_SIZE,
        )

        # --- Step 6: Trim output ---
        attn_out = attn_out[..., : self.v_head_dim]
        output.copy_(attn_out.flatten(start_dim=-2))
