# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forward_mha implementation for sparse MLA backends."""

from typing import TYPE_CHECKING, Generic, TypeVar

import torch

import vllm._custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
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
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


def build_sparse_mla_prefill_fields(
    cm: "CommonAttentionMetadata",
    num_decodes: int,
    num_prefills: int,
) -> tuple[torch.Tensor | None, int, bool]:
    """Compute prefill fields for sparse MLA metadata builders.

    Mirrors how MLACommonMetadataBuilder.build() computes prefill_query_start_loc.

    Returns:
        (prefill_query_start_loc, prefill_max_query_len, has_context)
    """
    if num_prefills == 0:
        return None, 0, False

    offset = cm.query_start_loc[num_decodes]
    prefill_query_start_loc = cm.query_start_loc[num_decodes:] - offset

    qsl_cpu = cm.query_start_loc_cpu
    prefill_qlens = qsl_cpu[num_decodes + 1 :] - qsl_cpu[num_decodes:-1]
    prefill_max_query_len = int(prefill_qlens.max().item())

    has_context = False
    seq_lens_cpu = cm.seq_lens.cpu()
    for i in range(num_prefills):
        idx = num_decodes + i
        if seq_lens_cpu[idx] > qsl_cpu[idx + 1] - qsl_cpu[idx]:
            has_context = True
            break

    return prefill_query_start_loc, prefill_max_query_len, has_context


@triton.jit
def _scatter_topk_kernel(
    mask_ptr,
    topk_ptr,
    cu_q_lens_ptr,
    num_words: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    max_q_len: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    NUM_REQS: tl.constexpr,
):
    """Scatter bits at topk positions into a bit-packed int32 mask.

    Grid: (sum(q_lens),).
    Output shape: (B, max_Q, ceil(max_S / 32)) int32.
    """
    row_idx = tl.program_id(0)

    b: tl.int32 = 0
    for i in tl.static_range(NUM_REQS):
        next_start = tl.load(cu_q_lens_ptr + i + 1)
        b += tl.where(next_start <= row_idx, 1, 0)

    q_start = tl.load(cu_q_lens_ptr + b)
    q_local = row_idx - q_start

    topk_row_ptr = topk_ptr + row_idx * topk_stride
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(topk_row_ptr + offsets, mask=in_range, other=-1)

    valid = in_range & (indices >= 0)
    word_indices = indices >> 5  # index // 32
    bit_indices = indices & 31  # index % 32
    bits = (1 << bit_indices).to(tl.int32)

    mask_row_ptr = mask_ptr + (b * max_q_len + q_local) * num_words
    tl.atomic_or(mask_row_ptr + word_indices, bits, mask=valid)


def _build_topk_mask(
    topk_indices_per_req: list[torch.Tensor],
    q_lens: list[int],
    max_q_len: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a bit-packed (B, max_Q, ceil(max_S/32)) int32 mask from topk indices."""
    B = len(q_lens)
    num_words = (max_seq_len + 31) // 32
    mask = torch.zeros(B, max_q_len, num_words, dtype=torch.int32, device=device)

    total_q = sum(q_lens)
    if total_q == 0:
        return mask

    topk_packed = torch.cat(topk_indices_per_req, dim=0)
    num_topk = topk_packed.shape[1]

    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device=device)
    cu_q_lens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(q_lens_t, dim=0, out=cu_q_lens[1:])

    BLOCK_TOPK = triton.next_power_of_2(num_topk)
    _scatter_topk_kernel[(total_q,)](
        mask,
        topk_packed,
        cu_q_lens,
        num_words=num_words,
        num_topk=num_topk,
        topk_stride=topk_packed.stride(0),
        max_q_len=max_q_len,
        BLOCK_TOPK=BLOCK_TOPK,
        NUM_REQS=B,
    )

    return mask


class SparseMLACommonImpl(SparseMLAAttentionImpl[T], Generic[T]):
    """Shared forward_mha for sparse MLA. Subclasses implement forward_mqa."""

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
        # MLA-specific
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

        fa_version = get_flash_attn_version(head_size=qk_head_dim)
        self._fa4_available = fa_version is not None and fa_version >= 4

        self.dcp_world_size: int = -1
        self.cp_kv_cache_interleave_size: int = (
            get_current_vllm_config().parallel_config.cp_kv_cache_interleave_size
        )

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _gather_and_decompress_all(
        self,
        kv_c_and_k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens_t: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        total_tokens: int,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather all positions from paged cache and decompress to (k, v)."""
        device = kv_c_and_k_pe_cache.device
        head_size = kv_c_and_k_pe_cache.shape[-1]
        num_reqs = seq_lens_t.shape[0]

        workspace = torch.empty(
            total_tokens,
            head_size,
            dtype=kv_c_and_k_pe_cache.dtype,
            device=device,
        )
        token_to_seq = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=device),
            seq_lens_t,
        )

        ops.gather_and_maybe_dequant_cache(
            src_cache=kv_c_and_k_pe_cache,
            dst=workspace,
            block_table=block_table,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            num_tokens=total_tokens,
            kv_cache_dtype=self.kv_cache_dtype,
            scale=k_scale,
        )

        kv_c = workspace[..., : self.kv_lora_rank]
        k_pe = workspace[..., self.kv_lora_rank :]

        kv_nope = self.kv_b_proj(kv_c)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.unsqueeze(1).expand(-1, self.num_heads, -1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        return k, v

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
        if not self._fa4_available:
            raise NotImplementedError(
                "Sparse MLA forward_mha requires FA4 (SM100+). "
                "On SM90, all tokens are routed through forward_mqa."
            )

        device = q.device
        prefill_qsl = getattr(attn_metadata, "prefill_query_start_loc", None)
        prefill_max_ql: int = getattr(attn_metadata, "prefill_max_query_len", 0)
        has_context: bool = getattr(attn_metadata, "has_context", False)

        assert prefill_qsl is not None, (
            "Metadata must provide prefill_query_start_loc for forward_mha"
        )

        if not has_context:
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope_new, v_new = kv_nope.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k_new = self._concat_k_nope_k_pe(k_nope_new, k_pe)

            attn_out, _ = flash_attn_varlen_func(
                q=q,
                k=k_new,
                v=v_new,
                cu_seqlens_q=prefill_qsl,
                cu_seqlens_k=prefill_qsl,
                max_seqlen_q=prefill_max_ql,
                max_seqlen_k=prefill_max_ql,
                softmax_scale=self.scale,
                causal=True,
                return_softmax_lse=True,
                fa_version=4,
            )
            attn_out = attn_out[..., : self.v_head_dim]
            output.copy_(attn_out.flatten(start_dim=-2))
            return

        num_decodes: int = getattr(attn_metadata, "num_decodes", 0)
        num_prefills: int = getattr(attn_metadata, "num_prefills", 0)
        num_decode_tokens: int = getattr(attn_metadata, "num_decode_tokens", 0)

        all_seq_lens = getattr(attn_metadata, "seq_lens", None)
        assert all_seq_lens is not None
        prefill_seq_lens = all_seq_lens[num_decodes : num_decodes + num_prefills]

        prefill_qsl_cpu = prefill_qsl.cpu()
        q_lens = [
            (prefill_qsl_cpu[i + 1] - prefill_qsl_cpu[i]).item()
            for i in range(num_prefills)
        ]
        max_q_len = max(q_lens) if q_lens else 0

        seq_lens_t = prefill_seq_lens.to(torch.int32)
        cu_seqlens_k = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
        torch.cumsum(seq_lens_t, dim=0, out=cu_seqlens_k[1:])
        total_kv_tokens = cu_seqlens_k[-1].item()
        max_kv_len = seq_lens_t.max().item() if num_prefills > 0 else 0

        block_table = getattr(attn_metadata, "block_table", None)
        assert block_table is not None
        prefill_block_table = block_table[num_decodes : num_decodes + num_prefills]

        k, v = self._gather_and_decompress_all(
            kv_c_and_k_pe_cache,
            prefill_block_table,
            seq_lens_t,
            cu_seqlens_k,
            total_kv_tokens,
            k_scale,
        )

        assert self.topk_indices_buffer is not None
        topk = self.topk_indices_buffer.shape[1]
        all_within_topk = bool(seq_lens_t.max().item() <= topk)

        if all_within_topk:
            attn_out, _ = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_qsl,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=self.scale,
                causal=True,
                return_softmax_lse=True,
                fa_version=4,
            )
        else:
            from vllm.vllm_flash_attn.cute.dense_mask import (
                dense_mask_mod,
                dense_mask_to_block_sparse,
            )

            num_prefill_tokens = q.shape[0]
            topk_all = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
            topk_per_req: list[torch.Tensor] = []
            ti_offset = 0
            for i in range(num_prefills):
                ql = q_lens[i]
                topk_per_req.append(topk_all[ti_offset : ti_offset + ql])
                ti_offset += ql

            dense_mask = _build_topk_mask(
                topk_per_req,
                q_lens,
                max_q_len,
                max_kv_len,
                device,
            )

            block_sparse_tensors = dense_mask_to_block_sparse(
                dense_mask,
                max_q_len,
                max_kv_len,
                tile_m=256,
                tile_n=128,
            )

            attn_out, _ = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_qsl,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=self.scale,
                causal=False,
                return_softmax_lse=True,
                fa_version=4,
                mask_mod=dense_mask_mod,
                block_sparse_tensors=block_sparse_tensors,
                aux_tensors=[dense_mask],
            )

        attn_out = attn_out[..., : self.v_head_dim]
        output.copy_(attn_out.flatten(start_dim=-2))
