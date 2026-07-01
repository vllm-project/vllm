# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forward_mha implementation and metadata builder for sparse MLA backends."""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_down
from vllm.utils.torch_utils import is_quantized_kv_cache, np_to_pinned_tensor
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)
_PROFILE_COUNTS: defaultdict[str, int] = defaultdict(int)


def _profile_context_enabled() -> bool:
    return os.getenv("VLLM_SPARSE_MLA_PROFILE_CONTEXT", "0") == "1"


def _profile_context_max_samples() -> int:
    value = os.getenv("VLLM_SPARSE_MLA_PROFILE_CONTEXT_SAMPLES", "1")
    try:
        return max(int(value), 0)
    except ValueError:
        return 1


def _profile_context_time(label: str, fn):
    if not _profile_context_enabled() or not torch.cuda.is_available():
        return fn()
    if _PROFILE_COUNTS[label] >= _profile_context_max_samples():
        return fn()
    _PROFILE_COUNTS[label] += 1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.accelerator.synchronize()
    print(f"SPARSE_MLA_PROFILE {label} {start.elapsed_time(end):.3f} ms")
    return result


@dataclass
class SparseMLAChunkedContextMetadata:
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    starts_cpu: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens_cpu: torch.Tensor
    context_lens_cpu: torch.Tensor
    workspace: torch.Tensor
    token_to_seq: torch.Tensor
    chunk_total_token: list[int]
    prefill_tokens_with_context: int | None


class SparseMLACommonMetadataBuilder(AttentionMetadataBuilder[T]):
    metadata_cls: type[T]

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens: int = vllm_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )
        workspace_head_size = (
            self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
        )
        self.chunked_prefill_workspace = torch.empty(
            (self.chunked_prefill_workspace_size, workspace_head_size),
            dtype=self.model_config.dtype,
            device=device,
        )

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: "VllmConfig") -> int:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        workspace_size = min(
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            64 * 1024,
        )
        return max(
            workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            np_to_pinned_tensor(req_id_per_token), non_blocking=True
        )
        return self.req_id_per_token_buffer[:num_tokens]

    def _build_chunked_context_fields(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
        prefill_query_lens_cpu: torch.Tensor | None,
    ) -> SparseMLAChunkedContextMetadata | None:
        if num_prefills == 0 or prefill_query_lens_cpu is None:
            return None

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        context_lens_cpu = (
            seq_lens_cpu[num_decodes : num_decodes + num_prefills]
            - prefill_query_lens_cpu
        )
        max_context_len = context_lens_cpu.max().item()
        if max_context_len <= 0:
            return None

        num_prefills_with_context = (context_lens_cpu > 0).sum().item()
        assert num_prefills_with_context > 0

        max_context_chunk = (
            self.chunked_prefill_workspace_size // num_prefills_with_context
        )
        if current_platform.is_cuda():
            max_context_chunk = round_down(
                max_context_chunk, self.kv_cache_spec.block_size
            )
        assert max_context_chunk > 0

        num_chunks = cdiv(max_context_len, max_context_chunk)
        chunk_starts = torch.empty(
            num_chunks, num_prefills, dtype=torch.int32, pin_memory=True
        ).copy_(
            torch.arange(num_chunks, dtype=torch.int32)
            .multiply_(max_context_chunk)
            .unsqueeze(1)
        )
        chunk_ends = torch.min(
            context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
        )
        chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

        cu_seq_lens_cpu = torch.zeros(
            num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
        )
        torch.cumsum(
            chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
        )
        chunk_total_token = cu_seq_lens_cpu[:, -1]

        max_tokens_over_chunk = chunk_total_token.max().item()
        token_to_seq_cpu = torch.zeros(
            (num_chunks, max_tokens_over_chunk), dtype=torch.int32, pin_memory=True
        )
        req_indices = torch.arange(num_prefills, dtype=torch.int32)
        for i in range(num_chunks):
            token_to_seq = torch.repeat_interleave(req_indices, chunk_seq_lens[i])
            token_to_seq_cpu[i, : token_to_seq.shape[0]] = token_to_seq

        prefill_tokens_with_context = None
        if num_prefills_with_context > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
            prefill_tokens_with_context = prefill_qsl_cpu[
                num_prefills_with_context
            ].item()

        return SparseMLAChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(self.device, non_blocking=True),
            starts=chunk_starts.to(self.device, non_blocking=True),
            starts_cpu=chunk_starts,
            seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            seq_lens_cpu=chunk_seq_lens,
            context_lens_cpu=context_lens_cpu,
            workspace=self.chunked_prefill_workspace,
            token_to_seq=token_to_seq_cpu.to(self.device, non_blocking=True),
            chunk_total_token=chunk_total_token.tolist(),
            prefill_tokens_with_context=prefill_tokens_with_context,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> T:
        req_id_per_token = self._build_req_id_per_token(common_attn_metadata)

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
        )
        (
            prefill_query_start_loc,
            prefill_max_query_len,
            has_context,
            prefill_query_lens_cpu,
        ) = self._build_prefill_fields(common_attn_metadata, num_decodes, num_prefills)
        chunked_context = self._build_chunked_context_fields(
            common_attn_metadata,
            num_decodes,
            num_prefills,
            prefill_query_lens_cpu,
        )

        return self.metadata_cls(  # type: ignore[call-arg]
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_query_len=prefill_max_query_len,
            has_context=has_context,
            prefill_query_lens_cpu=prefill_query_lens_cpu,
            chunked_context=chunked_context,
        )

    @staticmethod
    def _build_prefill_fields(
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
    ) -> tuple[
        torch.Tensor | None,  # prefill_query_start_loc
        int,  # prefill_max_query_len
        bool,  # has_context
        torch.Tensor | None,  # prefill_query_lens_cpu
    ]:
        if num_prefills == 0:
            return None, 0, False, None

        offset = common_attn_metadata.query_start_loc[num_decodes]
        prefill_query_start_loc = (
            common_attn_metadata.query_start_loc[num_decodes:] - offset
        )

        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
        prefill_query_lens = prefill_qsl_cpu[1:] - prefill_qsl_cpu[:-1]
        prefill_max_query_len = int(prefill_query_lens.max().item())

        num_computed_tokens_cpu = (
            common_attn_metadata.compute_num_computed_tokens().cpu()
        )
        context_lens_cpu = num_computed_tokens_cpu[
            num_decodes : num_decodes + num_prefills
        ]
        has_context = bool(context_lens_cpu.max().item() > 0)

        return (
            prefill_query_start_loc,
            prefill_max_query_len,
            has_context,
            prefill_query_lens,
        )


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


@triton.jit
def _scatter_topk_single_req_kernel(
    mask_ptr,
    topk_ptr,
    num_words: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row_idx = tl.program_id(0)

    topk_row_ptr = topk_ptr + row_idx * topk_stride
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(topk_row_ptr + offsets, mask=in_range, other=-1)

    valid = in_range & (indices >= 0)
    word_indices = indices >> 5
    bit_indices = indices & 31
    bits = (1 << bit_indices).to(tl.int32)

    mask_row_ptr = mask_ptr + row_idx * num_words
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

    if B == 1:
        topk_packed = topk_indices_per_req[0]
        num_topk = topk_packed.shape[1]
        BLOCK_TOPK = triton.next_power_of_2(num_topk)
        _scatter_topk_single_req_kernel[(total_q,)](
            mask,
            topk_packed,
            num_words=num_words,
            num_topk=num_topk,
            topk_stride=topk_packed.stride(0),
            BLOCK_TOPK=BLOCK_TOPK,
        )
        return mask

    topk_packed = torch.cat(topk_indices_per_req, dim=0)
    num_topk = topk_packed.shape[1]

    q_lens_t = np_to_pinned_tensor(np.asarray(q_lens, dtype=np.int32)).to(
        device, non_blocking=True
    )
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
        topk_indices_buffer: torch.Tensor | None = None,
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

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer  # type: ignore[attr-defined]
            if indexer is not None
            else topk_indices_buffer
        )

        fa_version = get_flash_attn_version(head_size=qk_head_dim)
        self._fa4_available = fa_version is not None and fa_version >= 4

        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        if self._use_flashinfer_concat_mla_k:
            torch.ops.vllm.flashinfer_concat_mla_k(k, k_nope, k_pe)
        else:
            k[..., : k_nope.shape[-1]] = k_nope
            k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _project_kv(
        self, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return self._concat_k_nope_k_pe(k_nope, k_pe), v

    @staticmethod
    def _slice_topk_per_req(
        topk_all: torch.Tensor,
        q_lens: list[int],
    ) -> list[torch.Tensor]:
        topk_per_req: list[torch.Tensor] = []
        offset = 0
        for q_len in q_lens:
            topk_per_req.append(topk_all[offset : offset + q_len])
            offset += q_len
        return topk_per_req

    @staticmethod
    def _remap_topk_to_ranges(
        topk_per_req: list[torch.Tensor],
        range_starts: list[int],
        range_lens: list[int],
    ) -> list[torch.Tensor]:
        remapped: list[torch.Tensor] = []
        for topk, start, length in zip(topk_per_req, range_starts, range_lens):
            end = start + length
            valid = (topk >= start) & (topk < end)
            remapped.append(torch.where(valid, topk - start, -1))
        return remapped

    def _run_sparse_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_lens: list[int],
        k_lens: list[int],
        topk_per_req: list[torch.Tensor],
        causal: bool,
        allow_dense: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = topk_per_req[0].shape[1] if topk_per_req else 0
        force_dense_mha = getattr(self, "_sparse_mla_force_dense_mha", False)
        force_masked_mha = getattr(self, "_sparse_mla_force_masked_mha", False)
        use_dense = allow_dense and (
            force_dense_mha or (max_seqlen_k <= topk and not force_masked_mha)
        )
        if use_dense:
            attn_out, _ = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=causal,
                return_softmax_lse=True,
                fa_version=4,
            )
            return attn_out, _

        from vllm.model_executor.layers.attention.sparse_mla_mask import (
            dense_mask_mod,
            dense_mask_to_block_sparse,
        )

        use_block_sparse = envs.VLLM_SPARSE_MLA_USE_BLOCK_SPARSE
        dense_mask = _build_topk_mask(
            topk_per_req,
            q_lens,
            max_seqlen_q,
            max_seqlen_k,
            q.device,
        )
        if use_block_sparse:
            # FA4 varlen sparsity validates against q_stage * tile_m.
            sparse_tile_m = 128 if max_seqlen_q <= 128 else 256
            mask_words_per_tile = 128 // 32
            num_mask_n_blocks = triton.cdiv(max_seqlen_k, 128)
            padded_mask_words = num_mask_n_blocks * mask_words_per_tile
            if dense_mask.shape[2] < padded_mask_words:
                dense_mask = torch.nn.functional.pad(
                    dense_mask, (0, padded_mask_words - dense_mask.shape[2])
                )

            block_sparse_tensors = dense_mask_to_block_sparse(
                dense_mask,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seq_lens_q=q_lens,
                seq_lens_k=k_lens,
                tile_m=sparse_tile_m,
                tile_n=128,
            )

            attn_out, lse = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=False,
                return_softmax_lse=True,
                fa_version=4,
                mask_mod=dense_mask_mod,
                block_sparse_tensors=block_sparse_tensors,
                aux_tensors=[dense_mask],
                aux_tensor_leading_dims=[2],
            )
        else:
            attn_out, lse = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=causal,
                return_softmax_lse=True,
                fa_version=4,
                mask_mod=dense_mask_mod,
                aux_tensors=[dense_mask],
                aux_tensor_leading_dims=[2],
            )

        return attn_out, lse

    def _compute_context_mha(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        q_lens: list[int],
        topk_per_req: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunked_context = attn_metadata.chunked_context  # type: ignore[attr-defined]
        assert chunked_context is not None
        output = None
        output_lse = None
        merge_output = None
        merge_output_lse = None
        workspace = chunked_context.workspace

        for i, toks in enumerate(chunked_context.seq_tot):
            if toks == 0:
                continue
            _profile_context_time(
                "masked_context.dense_gather",
                lambda i=i: ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=attn_metadata.block_table,  # type: ignore[attr-defined]
                    cu_seq_lens=chunked_context.cu_seq_lens[i],
                    token_to_seq=chunked_context.token_to_seq[i],
                    num_tokens=chunked_context.chunk_total_token[i],
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=chunked_context.starts[i],
                ),
            )

            chunk_kv_c = workspace[:toks, : self.kv_lora_rank]
            chunk_k_pe = workspace[:toks, self.kv_lora_rank :].unsqueeze(1)
            k, v = _profile_context_time(
                "masked_context.project_kv",
                lambda chunk_kv_c=chunk_kv_c, chunk_k_pe=chunk_k_pe: self._project_kv(
                    chunk_kv_c, chunk_k_pe
                ),
            )
            chunk_lens = chunked_context.seq_lens_cpu[i].tolist()
            chunk_starts = chunked_context.starts_cpu[i].tolist()
            chunk_topk = self._remap_topk_to_ranges(
                topk_per_req, chunk_starts, chunk_lens
            )
            attn_out, lse = _profile_context_time(
                "masked_context.run_sparse_mha",
                lambda k=k,
                v=v,
                i=i,
                chunk_lens=chunk_lens,
                chunk_topk=chunk_topk: self._run_sparse_mha(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,  # type: ignore[attr-defined]
                    cu_seqlens_k=chunked_context.cu_seq_lens[i],
                    max_seqlen_q=attn_metadata.prefill_max_query_len,  # type: ignore[attr-defined]
                    max_seqlen_k=chunked_context.max_seq_lens[i],
                    q_lens=q_lens,
                    k_lens=chunk_lens,
                    topk_per_req=chunk_topk,
                    causal=False,
                ),
            )

            if output is None:
                output = attn_out
                output_lse = lse
            else:
                if merge_output is None:
                    assert output_lse is not None
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                assert output_lse is not None and merge_output_lse is not None
                _profile_context_time(
                    "masked_context.merge_chunks",
                    lambda merge_output=merge_output,
                    merge_output_lse=merge_output_lse,
                    output=output,
                    output_lse=output_lse,
                    attn_out=attn_out,
                    lse=lse: merge_attn_states(
                        output=merge_output,
                        output_lse=merge_output_lse,
                        prefix_output=output,
                        prefix_lse=output_lse,
                        suffix_output=attn_out,
                        suffix_lse=lse,
                    ),
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        assert output is not None and output_lse is not None
        return output, output_lse

    def _compute_context_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        q_lens: list[int],
        topk_per_req: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        mqa_q: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        del output_scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Sparse MLA forward_mha with FP8 KV cache not yet supported"
            )
        if not self._fa4_available:
            raise NotImplementedError(
                "Sparse MLA forward_mha requires FA4 (SM100+). "
                "On SM90, all tokens are routed through forward_mqa."
            )

        cu_seqlens = attn_metadata.prefill_query_start_loc  # type: ignore[attr-defined]
        max_seq_len = attn_metadata.prefill_max_query_len  # type: ignore[attr-defined]
        num_decode_tokens = attn_metadata.num_decode_tokens  # type: ignore[attr-defined]
        prefill_query_lens_cpu = attn_metadata.prefill_query_lens_cpu  # type: ignore[attr-defined]
        assert prefill_query_lens_cpu is not None
        q_lens = prefill_query_lens_cpu.tolist()

        k, v = self._project_kv(kv_c_normed, k_pe)

        assert self.topk_indices_buffer is not None
        topk_all = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + q.shape[0]
        ]
        topk_per_req = self._slice_topk_per_req(topk_all, q_lens)

        has_context = attn_metadata.has_context  # type: ignore[attr-defined]
        if not has_context:
            attn_out, _ = self._run_sparse_mha(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                q_lens=q_lens,
                k_lens=q_lens,
                topk_per_req=topk_per_req,
                causal=True,
                allow_dense=True,
            )
            attn_out = attn_out[..., : self.v_head_dim]
            output.copy_(attn_out.flatten(start_dim=-2))
            return

        chunked_context = attn_metadata.chunked_context  # type: ignore[attr-defined]
        assert chunked_context is not None
        context_lens = chunked_context.context_lens_cpu.tolist()
        suffix_topk = self._remap_topk_to_ranges(topk_per_req, context_lens, q_lens)
        suffix_output, suffix_lse = _profile_context_time(
            "suffix.run_sparse_mha",
            lambda: self._run_sparse_mha(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                q_lens=q_lens,
                k_lens=q_lens,
                topk_per_req=suffix_topk,
                causal=True,
            ),
        )
        context_result = None
        if mqa_q is not None:
            context_result = self._compute_context_mqa(
                q=mqa_q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                attn_metadata=attn_metadata,
                q_lens=q_lens,
                topk_per_req=topk_per_req,
            )
        if context_result is None:
            context_result = self._compute_context_mha(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                attn_metadata=attn_metadata,
                k_scale=k_scale,
                q_lens=q_lens,
                topk_per_req=topk_per_req,
            )
        context_output, context_lse = context_result

        merged_output = output.view(-1, self.num_heads, self.v_head_dim)
        _profile_context_time(
            "forward.merge_context_suffix",
            lambda: merge_attn_states(
                output=merged_output,
                prefix_output=context_output[..., : self.v_head_dim],
                prefix_lse=context_lse,
                suffix_output=suffix_output[..., : self.v_head_dim],
                suffix_lse=suffix_lse,
                prefill_tokens_with_context=chunked_context.prefill_tokens_with_context,
            ),
        )
