# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared MHA implementation and metadata builder for sparse MLA backends."""

from shutil import which
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonMetadata,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import is_quantized_kv_cache, np_to_pinned_tensor
from vllm.v1.attention.backend import AttentionMetadata, AttentionMetadataBuilder
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


class SparseMLACommonMetadataBuilder(AttentionMetadataBuilder[T]):
    metadata_cls: type[T]
    require_uniform_decodes: ClassVar[bool] = False

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
        parallel_config = vllm_config.parallel_config
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_local_block_size = self.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )
        workspace_head_size = (
            self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
        )
        workspace_rows = self.chunked_prefill_workspace_size
        if self.dcp_world_size > 1:
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            workspace_rows += self.chunked_prefill_workspace_size // self.dcp_world_size
        self.chunked_prefill_workspace = torch.empty(
            (workspace_rows, workspace_head_size),
            dtype=self.model_config.dtype,
            device=device,
        )
        layer_prefill_backend = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend
        self._prefill_backend = (
            layer_prefill_backend.clone() if layer_prefill_backend is not None else None
        )

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: "VllmConfig") -> int:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        topk_tokens = model_config.hf_config.index_topk

        workspace_size = min(
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            64 * 1024,
            scheduler_config.max_num_seqs * topk_tokens,
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
    ) -> "MLACommonPrefillMetadata.ChunkedContextMetadata | None":
        if num_prefills == 0 or prefill_query_lens_cpu is None:
            return None

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        context_lens_cpu = (
            seq_lens_cpu[num_decodes : num_decodes + num_prefills]
            - prefill_query_lens_cpu
        )
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_query_start_loc_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]

        return build_mla_chunked_context_metadata(
            context_lens_cpu=context_lens_cpu,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            num_prefills=num_prefills,
            chunked_prefill_workspace=self.chunked_prefill_workspace,
            chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
            block_size=self.kv_cache_spec.block_size,
            align_chunk_to_block=current_platform.is_cuda(),
            device=self.device,
            dcp_world_size=self.dcp_world_size,
            dcp_local_block_size=self.dcp_local_block_size,
            dcp_virtual_block_size=self.dcp_virtual_block_size,
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
            require_uniform=self.require_uniform_decodes,
        )
        (
            prefill_query_start_loc,
            prefill_max_query_len,
            prefill_query_lens_cpu,
        ) = self._build_prefill_fields(common_attn_metadata, num_decodes, num_prefills)

        prefill_max_seq_len = 0
        prefill: MLACommonPrefillMetadata | None = None
        if num_prefills > 0 and self._prefill_backend is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_max_seq_len = int(
                seq_lens_cpu[num_decodes : num_decodes + num_prefills].max().item()
            )
            prefill = MLACommonPrefillMetadata(
                block_table=common_attn_metadata.block_table_tensor[num_decodes:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=prefill_max_query_len,
                query_lens_cpu=prefill_query_lens_cpu,
                chunked_context=self._build_chunked_context_fields(
                    common_attn_metadata,
                    num_decodes,
                    num_prefills,
                    prefill_query_lens_cpu,
                ),
                q_data_type=self.model_config.dtype,
                output_dtype=self.model_config.dtype,
                prefill_backend=self._prefill_backend,
            )
            self._prefill_backend.prepare_metadata(prefill)

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
            prefill_max_seq_len=prefill_max_seq_len,
            prefill=prefill,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        )

    @staticmethod
    def _build_prefill_fields(
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
    ) -> tuple[torch.Tensor | None, int, torch.Tensor | None]:
        if num_prefills == 0:
            return None, 0, None

        offset = common_attn_metadata.query_start_loc[num_decodes]
        prefill_query_start_loc = (
            common_attn_metadata.query_start_loc[num_decodes:] - offset
        )

        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
        prefill_query_lens = prefill_qsl_cpu[1:] - prefill_qsl_cpu[:-1]
        prefill_max_query_len = int(prefill_query_lens.max().item())
        return prefill_query_start_loc, prefill_max_query_len, prefill_query_lens


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
    row_idx = tl.program_id(0)
    req_idx: tl.int32 = 0
    for i in tl.static_range(NUM_REQS):
        next_start = tl.load(cu_q_lens_ptr + i + 1)
        req_idx += tl.where(next_start <= row_idx, 1, 0)

    q_start = tl.load(cu_q_lens_ptr + req_idx)
    q_local = row_idx - q_start
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(
        topk_ptr + row_idx * topk_stride + offsets,
        mask=in_range,
        other=-1,
    )
    valid = in_range & (indices >= 0)
    word_indices = indices >> 5
    bits = (1 << (indices & 31)).to(tl.int32)
    mask_row_ptr = mask_ptr + (req_idx * max_q_len + q_local) * num_words
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
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(
        topk_ptr + row_idx * topk_stride + offsets,
        mask=in_range,
        other=-1,
    )
    valid = in_range & (indices >= 0)
    word_indices = indices >> 5
    bits = (1 << (indices & 31)).to(tl.int32)
    tl.atomic_or(mask_ptr + row_idx * num_words + word_indices, bits, mask=valid)


def _build_topk_mask(
    topk_indices_per_req: list[torch.Tensor],
    q_lens: list[int],
    max_q_len: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a bit-packed ``[B, max_Q, ceil(max_S / 32)]`` top-k mask."""
    batch_size = len(q_lens)
    num_words = (max_seq_len + 31) // 32
    mask = torch.zeros(
        batch_size,
        max_q_len,
        num_words,
        dtype=torch.int32,
        device=device,
    )
    total_q = sum(q_lens)
    if total_q == 0:
        return mask

    if batch_size == 1:
        topk_packed = topk_indices_per_req[0]
        num_topk = topk_packed.shape[1]
        _scatter_topk_single_req_kernel[(total_q,)](
            mask,
            topk_packed,
            num_words=num_words,
            num_topk=num_topk,
            topk_stride=topk_packed.stride(0),
            BLOCK_TOPK=triton.next_power_of_2(num_topk),
        )
        return mask

    topk_packed = torch.cat(topk_indices_per_req, dim=0)
    num_topk = topk_packed.shape[1]
    q_lens_tensor = np_to_pinned_tensor(np.asarray(q_lens, dtype=np.int32)).to(
        device, non_blocking=True
    )
    cu_q_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    torch.cumsum(q_lens_tensor, dim=0, out=cu_q_lens[1:])
    _scatter_topk_kernel[(total_q,)](
        mask,
        topk_packed,
        cu_q_lens,
        num_words=num_words,
        num_topk=num_topk,
        topk_stride=topk_packed.stride(0),
        max_q_len=max_q_len,
        BLOCK_TOPK=triton.next_power_of_2(num_topk),
        NUM_REQS=batch_size,
    )
    return mask


class SparseMLACommonImpl(MLACommonBaseImpl[T], Generic[T]):
    """Sparse MLA base with dense and masked-MHA prefill paths."""

    is_sparse = True

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
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer  # type: ignore[attr-defined]
            if indexer is not None
            else topk_indices_buffer
        )
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and which("ninja") is not None
            and self.num_heads == 128
            and self.qk_nope_head_dim == 128
            and self.qk_rope_head_dim == 64
        )
        fa_version = get_flash_attn_version(
            head_size=qk_head_dim,
            head_size_v=v_head_dim,
        )
        self.masked_mha_available = fa_version == 4 and not is_quantized_kv_cache(
            kv_cache_dtype
        )

    @staticmethod
    def _slice_topk_per_req(
        topk_all: torch.Tensor,
        q_lens: list[int],
    ) -> list[torch.Tensor]:
        topk_per_req = []
        offset = 0
        for q_len in q_lens:
            topk_per_req.append(topk_all[offset : offset + q_len])
            offset += q_len
        return topk_per_req

    @staticmethod
    def _remap_topk_to_ranges(
        topk_per_req: list[torch.Tensor],
        range_starts: list[int] | torch.Tensor,
        range_lens: list[int],
    ) -> list[torch.Tensor]:
        remapped = []
        for topk, start, length in zip(topk_per_req, range_starts, range_lens):
            valid = (topk >= start) & (topk < start + length)
            remapped.append(torch.where(valid, topk - start, -1))
        return remapped

    def _project_kv(
        self, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return self._concat_k_nope_k_pe(k_nope, k_pe), v

    def _run_masked_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        topk_per_req: list[torch.Tensor],
        q_lens: list[int],
        causal: bool,
        return_softmax_lse: bool = False,
        dense_mask: torch.Tensor | None = None,
        key_starts: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.attention.sparse_mla_mask import (
            dense_mask_mod,
            offset_dense_mask_mod,
        )
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        tile_m = 128 if max_seqlen_q <= 128 else 256
        padded_q_len = triton.cdiv(max_seqlen_q, tile_m) * tile_m
        if dense_mask is None:
            dense_mask = _build_topk_mask(
                topk_per_req,
                q_lens,
                padded_q_len,
                max_seqlen_k,
                q.device,
            )
        if key_starts is not None:
            dense_mask[:, 0, -1].copy_(key_starts)
        kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.scale,
            "return_softmax_lse": return_softmax_lse,
            "fa_version": 4,
            "mask_mod": dense_mask_mod if key_starts is None else offset_dense_mask_mod,
            "aux_tensors": [dense_mask],
            "aux_tensor_leading_dims": [2],
            "causal": causal,
        }

        return flash_attn_varlen_func(**kwargs)

    def _compute_context_mha(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        prefill_metadata: MLACommonPrefillMetadata,
        k_scale: torch.Tensor,
        q_lens: list[int],
        topk_per_req: list[torch.Tensor],
        dense_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "Masked MHA with context does not yet support decode context "
                "parallelism"
            )

        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None
        output = None
        output_lse = None
        merge_output = None
        merge_output_lse = None
        workspace = chunked_context.workspace

        for i, toks in enumerate(chunked_context.seq_tot):
            if toks == 0:
                continue
            ops.gather_and_maybe_dequant_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=chunked_context.cu_seq_lens[i],
                token_to_seq=chunked_context.token_to_seq[i],
                num_tokens=chunked_context.chunk_total_token[i],
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=chunked_context.starts[i],
            )

            chunk_kv_c = workspace[:toks, : self.kv_lora_rank]
            chunk_k_pe = workspace[:toks, self.kv_lora_rank :].unsqueeze(1)
            k, v = self._project_kv(chunk_kv_c, chunk_k_pe)
            chunk_lens = chunked_context.seq_lens[i].tolist()
            chunk_topk = (
                topk_per_req
                if dense_mask is not None
                else self._remap_topk_to_ranges(
                    topk_per_req,
                    chunked_context.starts[i],
                    chunk_lens,
                )
            )
            attn_out, lse = self._run_masked_mha(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=chunked_context.cu_seq_lens[i],
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=chunked_context.max_seq_lens[i],
                topk_per_req=chunk_topk,
                q_lens=q_lens,
                causal=False,
                return_softmax_lse=True,
                dense_mask=dense_mask,
                key_starts=(
                    chunked_context.starts[i] if dense_mask is not None else None
                ),
            )

            if output is None:
                output = attn_out
                output_lse = lse
            else:
                assert output_lse is not None
                if merge_output is None:
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                assert merge_output_lse is not None
                merge_attn_states(
                    output=merge_output,
                    output_lse=merge_output_lse,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_out,
                    suffix_lse=lse,
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        assert output is not None and output_lse is not None
        return output, output_lse

    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        prefill_max_seq_len = attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
        topk_tokens = attn_metadata.topk_tokens  # type: ignore[attr-defined]
        force_dense = getattr(self, "_sparse_mla_force_dense_mha", False)
        force_masked = getattr(self, "_sparse_mla_force_masked_mha", False)
        if force_dense or (prefill_max_seq_len <= topk_tokens and not force_masked):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                cast(MLACommonMetadata, attn_metadata),
                k_scale,
                output,
                output_scale,
            )

        assert output_scale is None
        assert self.masked_mha_available
        prefill_metadata = attn_metadata.prefill  # type: ignore[attr-defined]
        assert prefill_metadata is not None
        assert prefill_metadata.query_lens_cpu is not None
        assert self.topk_indices_buffer is not None

        q_lens = prefill_metadata.query_lens_cpu.tolist()
        num_decode_tokens = attn_metadata.num_decode_tokens  # type: ignore[attr-defined]
        topk_all = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + q.shape[0]
        ]
        topk_per_req = self._slice_topk_per_req(topk_all, q_lens)

        k, v = self._project_kv(kv_c_normed, k_pe)
        chunked_context = prefill_metadata.chunked_context
        if chunked_context is None:
            attn_out = self._run_masked_mha(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.max_query_len,
                topk_per_req=topk_per_req,
                q_lens=q_lens,
                causal=True,
            )
            assert isinstance(attn_out, torch.Tensor)
            output.copy_(attn_out[..., : self.v_head_dim].flatten(start_dim=-2))
            return

        context_lens = chunked_context.seq_lens.sum(dim=0).tolist()
        dense_mask = None
        tile_m = 128 if prefill_metadata.max_query_len <= 128 else 256
        padded_q_len = triton.cdiv(prefill_metadata.max_query_len, tile_m) * tile_m
        global_mask_bytes = (
            len(q_lens)
            * padded_q_len
            * (triton.cdiv(prefill_max_seq_len, 32) + 1)
            * torch.int32.itemsize
        )
        workspace_bytes = (
            chunked_context.workspace.numel() * chunked_context.workspace.element_size()
        )
        if global_mask_bytes <= workspace_bytes:
            dense_mask = _build_topk_mask(
                topk_per_req,
                q_lens,
                padded_q_len,
                prefill_max_seq_len,
                q.device,
            )
            dense_mask = torch.nn.functional.pad(dense_mask, (0, 1))
            suffix_topk = topk_per_req
        if dense_mask is None:
            suffix_topk = self._remap_topk_to_ranges(topk_per_req, context_lens, q_lens)
        suffix_output, suffix_lse = self._run_masked_mha(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill_metadata.query_start_loc,
            cu_seqlens_k=prefill_metadata.query_start_loc,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=prefill_metadata.max_query_len,
            topk_per_req=suffix_topk,
            q_lens=q_lens,
            causal=True,
            return_softmax_lse=True,
            dense_mask=dense_mask,
            key_starts=(
                chunked_context.context_lens if dense_mask is not None else None
            ),
        )
        context_output, context_lse = self._compute_context_mha(
            q=q,
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            prefill_metadata=prefill_metadata,
            k_scale=k_scale,
            q_lens=q_lens,
            topk_per_req=topk_per_req,
            dense_mask=dense_mask,
        )
        merge_attn_states(
            output=output.view(-1, self.num_heads, self.v_head_dim),
            prefix_output=context_output[..., : self.v_head_dim],
            prefix_lse=context_lse,
            suffix_output=suffix_output[..., : self.v_head_dim],
            suffix_lse=suffix_lse,
            prefill_tokens_with_context=chunked_context.prefill_tokens_with_context,
        )
