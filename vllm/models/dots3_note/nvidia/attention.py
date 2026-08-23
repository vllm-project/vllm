# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3 NOTE sliding-window MLA attention backends for Hopper.

Prefill and mixed batches expand the latent cache and use FlashAttention-3
varlen MHA. Decode-only batches use the Triton absorbed-MQA kernel.
"""

from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseBackend,
    FlashAttnMLASparseImpl,
    FlashAttnMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheLayout
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


@triton.jit
def _gather_swa_kv_kernel(
    kv_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    kv_out_ptr,
    valid_out_ptr,
    k_scale_ptr,
    kv_cache_stride_block,
    kv_cache_stride_token,
    kv_cache_stride_dim,
    block_table_stride_req,
    kv_out_stride_req,
    kv_out_stride_token,
    kv_out_stride_dim,
    valid_out_stride_req,
    valid_out_stride_token,
    GATHER_LEN: tl.constexpr,
    KV_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    token_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    gather_start = tl.maximum(seq_len - GATHER_LEN, 0)
    token_offsets = token_block_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    logical_tokens = gather_start + token_offsets
    logical_valid = (token_offsets < GATHER_LEN) & (logical_tokens < seq_len)

    logical_pages = logical_tokens // PAGE_SIZE
    page_offsets = logical_tokens - logical_pages * PAGE_SIZE
    physical_pages = tl.load(
        block_table_ptr + req_idx * block_table_stride_req + logical_pages,
        mask=logical_valid,
        other=-1,
    ).to(tl.int64)
    physical_valid = logical_valid & (physical_pages >= 0)

    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    values = tl.load(
        kv_cache_ptr
        + physical_pages[:, None] * kv_cache_stride_block
        + page_offsets[:, None] * kv_cache_stride_token
        + dim_offsets[None, :] * kv_cache_stride_dim,
        mask=physical_valid[:, None] & (dim_offsets[None, :] < KV_DIM),
        other=0.0,
    )
    if values.dtype.is_fp8():
        values = values.to(tl.float32) * tl.load(k_scale_ptr)
    tl.store(
        kv_out_ptr
        + req_idx * kv_out_stride_req
        + token_offsets[:, None] * kv_out_stride_token
        + dim_offsets[None, :] * kv_out_stride_dim,
        values,
        mask=(token_offsets[:, None] < GATHER_LEN) & (dim_offsets[None, :] < KV_DIM),
    )
    tl.store(
        valid_out_ptr
        + req_idx * valid_out_stride_req
        + token_offsets * valid_out_stride_token,
        physical_valid,
        mask=(dim_block_idx == 0) & (token_offsets < GATHER_LEN),
    )


@triton.jit
def _apply_swa_score_mask_kernel(
    scores_ptr,
    seq_lens_ptr,
    valid_ptr,
    scores_stride_req,
    scores_stride_head,
    scores_stride_query,
    scores_stride_token,
    valid_stride_req,
    valid_stride_token,
    GATHER_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_idx = tl.program_id(2)

    token_offsets = tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    gather_start = tl.maximum(seq_len - GATHER_LEN, 0)
    kv_positions = gather_start + token_offsets
    query_position = seq_len - QUERY_LEN + query_idx
    page_valid = tl.load(
        valid_ptr + req_idx * valid_stride_req + token_offsets * valid_stride_token,
        mask=token_offsets < GATHER_LEN,
        other=0,
    ).to(tl.int1)
    valid = (
        (token_offsets < GATHER_LEN)
        & page_valid
        & (kv_positions <= query_position)
        & (kv_positions >= query_position - WINDOW_SIZE + 1)
        & (query_position >= 0)
    )
    tl.store(
        scores_ptr
        + req_idx * scores_stride_req
        + head_idx * scores_stride_head
        + query_idx * scores_stride_query
        + token_offsets * scores_stride_token,
        tl.full((BLOCK_T,), -3.4028234663852886e38, tl.float32),
        mask=(token_offsets < GATHER_LEN) & ~valid,
    )


@dataclass
class Dots3NoteDecodeMetadata(MLACommonDecodeMetadata):
    query_len: int


@dataclass
class _SlidingWindowChunk:
    req_start: int
    req_end: int
    query_start: int
    query_end: int
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    starts: torch.Tensor
    token_to_seq: torch.Tensor
    num_kv_tokens: int
    max_seq_len_q: int
    max_seq_len_k: int


@dataclass
class _SlidingWindowMetadata:
    chunks: list[_SlidingWindowChunk]
    workspace: torch.Tensor


def _build_sliding_window_metadata(
    *,
    seq_lens_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    sliding_window: int,
    workspace: torch.Tensor,
    workspace_size: int,
    device: torch.device,
) -> _SlidingWindowMetadata:
    """Plan per-request latent-cache gathers for SWA varlen attention."""
    query_lens_cpu = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).to(
        dtype=torch.int32
    )
    seq_lens_cpu = seq_lens_cpu.to(dtype=torch.int32)
    kv_lens_cpu = torch.minimum(seq_lens_cpu, query_lens_cpu + sliding_window - 1)
    starts_cpu = seq_lens_cpu - kv_lens_cpu

    chunks: list[_SlidingWindowChunk] = []
    req_start = 0
    while req_start < query_lens_cpu.numel():
        req_end = req_start
        num_kv_tokens = 0
        while req_end < query_lens_cpu.numel():
            next_len = int(kv_lens_cpu[req_end].item())
            if num_kv_tokens and num_kv_tokens + next_len > workspace_size:
                break
            if next_len > workspace_size:
                raise ValueError(
                    "Dots3 NOTE SWA prefill window exceeds the MLA workspace: "
                    f"{next_len} > {workspace_size}"
                )
            num_kv_tokens += next_len
            req_end += 1

        query_lens = query_lens_cpu[req_start:req_end]
        kv_lens = kv_lens_cpu[req_start:req_end]
        num_reqs = req_end - req_start
        cu_seq_lens_q_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        cu_seq_lens_k_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        torch.cumsum(query_lens, 0, out=cu_seq_lens_q_cpu[1:])
        torch.cumsum(kv_lens, 0, out=cu_seq_lens_k_cpu[1:])
        token_to_seq_cpu = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32), kv_lens
        )
        query_start = int(query_start_loc_cpu[req_start].item())
        query_end = int(query_start_loc_cpu[req_end].item())
        chunks.append(
            _SlidingWindowChunk(
                req_start=req_start,
                req_end=req_end,
                query_start=query_start,
                query_end=query_end,
                cu_seq_lens_q=cu_seq_lens_q_cpu.to(device, non_blocking=True),
                cu_seq_lens_k=cu_seq_lens_k_cpu.to(device, non_blocking=True),
                starts=starts_cpu[req_start:req_end].to(device, non_blocking=True),
                token_to_seq=token_to_seq_cpu.to(device, non_blocking=True),
                num_kv_tokens=num_kv_tokens,
                max_seq_len_q=int(query_lens.max().item()),
                max_seq_len_k=int(kv_lens.max().item()),
            )
        )
        req_start = req_end

    return _SlidingWindowMetadata(chunks=chunks, workspace=workspace)


class Dots3NoteFlashAttnPrefillBackend(FlashAttnPrefillBackend):
    """FA3 varlen prefill for the NOTE SWA MLA dimensions."""

    @classmethod
    def supports_mla_dimensions(cls, dims: MLADimensions) -> bool:
        return dims == MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # FA3 on SM90 does not support NOTE's Q/K=256, V=128 combination via
        # the different-head-dimension path. Padding V selects its supported
        # equal-dimension varlen kernel.
        self.requires_v_padding = True

    def supports_quant_output(self, quant_key) -> bool:
        return False

    def run_sliding_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        sliding_window: int,
    ) -> torch.Tensor:
        output = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=(sliding_window - 1, 0),
            return_softmax_lse=False,
        )
        assert isinstance(output, torch.Tensor)
        return output


class Dots3NoteMLAMetadataBuilder(TritonMLAMetadataBuilder):
    """Keep decode on MQA and route prefill/mixed batches through FA3."""

    query_len_support = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.sliding_window = kv_cache_spec.sliding_window
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def _reserve_attn_logits_workspace(self) -> None:
        if not is_workspace_manager_initialized():
            return
        batch = self.vllm_config.scheduler_config.max_num_seqs
        max_query_len = self.reorder_batch_threshold or 1
        gather_len = (self.sliding_window + max_query_len - 1 + 7) // 8 * 8
        current_workspace_manager().get_simultaneous(
            (
                (
                    batch,
                    gather_len,
                    self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim,
                ),
                self.model_config.dtype,
            ),
            ((batch, gather_len), torch.bool),
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> Dots3NoteDecodeMetadata:
        del max_seq_len, query_start_loc_device, num_decode_tokens
        query_len = int(query_start_loc_cpu[1] - query_start_loc_cpu[0])
        return Dots3NoteDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            query_len=query_len,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MLACommonMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        if metadata.prefill is None:
            return metadata
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "Dots3 NOTE SWA prefill does not support decode context parallelism"
            )

        reqs_start = metadata.num_decodes
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[reqs_start:]
        query_start_loc_cpu = query_start_loc_cpu - query_start_loc_cpu[0]
        sliding_metadata = _build_sliding_window_metadata(
            seq_lens_cpu=seq_lens_cpu[reqs_start:],
            query_start_loc_cpu=query_start_loc_cpu,
            sliding_window=self.sliding_window,
            workspace=self.chunked_prefill_workspace,
            workspace_size=self.chunked_prefill_workspace_size,
            device=self.device,
        )
        metadata.prefill.chunked_context = None
        metadata.prefill.sliding_window = sliding_metadata  # type: ignore[attr-defined]
        if metadata.num_decodes > 0 and metadata.num_prefills > 0:
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            for chunk in sliding_metadata.chunks:
                req_start, req_end = chunk.req_start, chunk.req_end
                seq_lens = common_attn_metadata.seq_lens[
                    reqs_start + req_start : reqs_start + req_end
                ]
                query_lens = query_lens_cpu[req_start:req_end].to(
                    self.device, non_blocking=True
                )
                kv_lens = torch.minimum(seq_lens, query_lens + self.sliding_window - 1)
                chunk.cu_seq_lens_k.zero_()
                torch.cumsum(
                    kv_lens,
                    dim=0,
                    out=chunk.cu_seq_lens_k[1:],
                    dtype=torch.int32,
                )
                chunk.starts = seq_lens - kv_lens
                token_ids = torch.arange(
                    chunk.num_kv_tokens,
                    dtype=torch.int32,
                    device=self.device,
                )
                chunk.token_to_seq = torch.searchsorted(
                    chunk.cu_seq_lens_k[1:], token_ids, right=True
                ).to(torch.int32)
                chunk.token_to_seq.clamp_max_(req_end - req_start - 1)
        assert metadata.prefill.prefill_backend is not None
        metadata.prefill.prefill_backend.prepare_metadata(metadata.prefill)
        return metadata


class Dots3NoteTritonMLABackend(TritonMLABackend):
    """Internal NOTE SWA specialization; not a user-selectable backend."""

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [1088]

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["Dots3NoteTritonMLAImpl"]:
        return Dots3NoteTritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type[Dots3NoteMLAMetadataBuilder]:
        return Dots3NoteMLAMetadataBuilder


class Dots3NoteTritonMLAImpl(TritonMLAImpl):
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
        **mla_args,
    ) -> None:
        assert sliding_window is not None
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **mla_args,
        )
        self.sliding_window = sliding_window

    def _forward_swa_mqa(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        decode: Dots3NoteDecodeMetadata,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        query_len = decode.query_len
        num_tokens, num_heads, head_size = q.shape
        assert num_tokens % query_len == 0
        assert head_size == self.head_size
        num_reqs = num_tokens // query_len
        block_table = decode.block_table[:num_reqs]
        seq_lens = decode.seq_lens[:num_reqs]
        gather_len = (self.sliding_window + query_len - 1 + 7) // 8 * 8

        workspace_specs = (
            ((num_reqs, gather_len, head_size), q.dtype),
            ((num_reqs, gather_len), torch.bool),
        )
        if is_workspace_manager_initialized():
            kv_latent, kv_valid = current_workspace_manager().get_simultaneous(
                *workspace_specs
            )
        else:
            kv_latent = torch.empty(
                workspace_specs[0][0], dtype=q.dtype, device=q.device
            )
            kv_valid = torch.empty(
                workspace_specs[1][0], dtype=torch.bool, device=q.device
            )

        block_t = 8
        block_d = 128
        _gather_swa_kv_kernel[
            (
                num_reqs,
                triton.cdiv(gather_len, block_t),
                triton.cdiv(head_size, block_d),
            )
        ](
            kv_cache,
            block_table,
            seq_lens,
            kv_latent,
            kv_valid,
            k_scale,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            kv_latent.stride(0),
            kv_latent.stride(1),
            kv_latent.stride(2),
            kv_valid.stride(0),
            kv_valid.stride(1),
            GATHER_LEN=gather_len,
            KV_DIM=head_size,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            PAGE_SIZE=kv_cache.shape[1],
            num_warps=4,
        )

        q = q.view(num_reqs, query_len, num_heads, head_size)
        scores = torch.bmm(
            q.reshape(num_reqs, query_len * num_heads, head_size),
            kv_latent.transpose(1, 2),
        ).view(num_reqs, query_len, num_heads, gather_len)
        scores = scores.float()
        scores.mul_(self.scale)
        scores_by_head = scores.transpose(1, 2)
        _apply_swa_score_mask_kernel[(num_reqs, num_heads, query_len)](
            scores_by_head,
            seq_lens,
            kv_valid,
            scores_by_head.stride(0),
            scores_by_head.stride(1),
            scores_by_head.stride(2),
            scores_by_head.stride(3),
            kv_valid.stride(0),
            kv_valid.stride(1),
            GATHER_LEN=gather_len,
            WINDOW_SIZE=self.sliding_window,
            QUERY_LEN=query_len,
            BLOCK_T=triton.next_power_of_2(gather_len),
            num_warps=4,
        )
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        output = torch.bmm(
            probs.reshape(num_reqs, query_len * num_heads, gather_len),
            kv_latent[..., : self.kv_lora_rank],
        )
        return output.view(num_tokens, num_heads, self.kv_lora_rank), None

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        prefill = attn_metadata.prefill
        sliding = getattr(prefill, "sliding_window", None)
        if prefill is None or sliding is None:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        assert output_scale is None
        assert isinstance(prefill.prefill_backend, Dots3NoteFlashAttnPrefillBackend)
        use_fp8_prefill = prefill.q_data_type == current_platform.fp8_dtype()
        output = output.view(-1, self.num_heads, self.v_head_dim)

        for chunk in sliding.chunks:
            toks = chunk.num_kv_tokens
            workspace = sliding.workspace
            block_table = prefill.block_table[chunk.req_start : chunk.req_end]
            if is_quantized_kv_cache(self.kv_cache_dtype) and not use_fp8_prefill:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens_k,
                    token_to_seq=chunk.token_to_seq,
                    num_tokens=toks,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=chunk.starts,
                )
            else:
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens_k,
                    batch_size=chunk.req_end - chunk.req_start,
                    seq_starts=chunk.starts,
                )

            # Slicing the latent part from the packed latent+RoPE cache keeps
            # the full cache-entry row stride. FP8 linear kernels require
            # contiguous rows and would otherwise read them at wrong offsets.
            kv_c = workspace[:toks, : self.kv_lora_rank].contiguous()
            weight_dtype = (
                self.kv_b_proj.weight.dtype
                if hasattr(self.kv_b_proj, "weight")
                else self.kv_b_proj.params_dtype
            )
            if (
                use_fp8_prefill or weight_dtype != current_platform.fp8_dtype()
            ) and weight_dtype != torch.uint8:
                kv_c = kv_c.to(weight_dtype)
            k_pe_chunk = workspace[:toks, self.kv_lora_rank :].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill.q_data_type)
                k_pe_chunk = k_pe_chunk.to(prefill.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe_chunk)
            chunk_output = prefill.prefill_backend.run_sliding_window(
                q=q[chunk.query_start : chunk.query_end],
                k=k,
                v=v,
                cu_seq_lens_q=chunk.cu_seq_lens_q,
                cu_seq_lens_k=chunk.cu_seq_lens_k,
                max_seq_len_q=chunk.max_seq_len_q,
                max_seq_len_k=chunk.max_seq_len_k,
                sliding_window=self.sliding_window,
            )
            output[chunk.query_start : chunk.query_end].copy_(
                chunk_output[..., : self.v_head_dim]
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        decode = attn_metadata.decode
        assert isinstance(decode, Dots3NoteDecodeMetadata)
        return self._forward_swa_mqa(
            q,
            kv_c_and_k_pe_cache,
            decode,
            layer._k_scale,
        )


class Dots3NotePaddedSparseBackend(FlashAttnMLASparseBackend):
    """NOTE DSA backend for cache rows padded to the SWA latent width."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # NOTE packs the smaller DSA index pages beside the padded MLA/SWA pages.
        # Keep the layer dimension inside the block dimension so mixed page sizes
        # remain representable by the shared KV-cache allocation.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_name() -> str:
        return "DOTS3_NOTE_PADDED_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["Dots3NotePaddedSparseImpl"]:
        return Dots3NotePaddedSparseImpl


class Dots3NotePaddedSparseImpl(FlashAttnMLASparseImpl):
    """Read top-k KV directly from uniformly padded cache rows."""

    def _logical_cache(self, kv_cache: torch.Tensor) -> torch.Tensor:
        assert kv_cache.shape[-1] >= self.head_size
        return kv_cache[..., : self.head_size]

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            self._logical_cache(kv_cache),
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        super().forward_mha(
            q,
            kv_c_normed,
            k_pe,
            self._logical_cache(kv_c_and_k_pe_cache),
            attn_metadata,
            k_scale,
            output,
            output_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "Dots3NotePaddedSparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        physical_head_size = kv_c_and_k_pe_cache.shape[-1]
        block_stride, token_stride, head_stride = kv_c_and_k_pe_cache.stride()
        if (
            token_stride != physical_head_size
            or head_stride != 1
            or block_stride % physical_head_size != 0
        ):
            raise RuntimeError(
                "Dots3 NOTE padded DSA cache must contain contiguous token rows; "
                f"got shape={tuple(kv_c_and_k_pe_cache.shape)} and "
                f"stride={kv_c_and_k_pe_cache.stride()}"
            )
        block_row_stride = block_stride // physical_head_size
        block_indices = torch.div(
            topk_indices,
            attn_metadata.block_size,
            rounding_mode="floor",
        )
        block_indices.clamp_min_(0).mul_(block_row_stride - attn_metadata.block_size)
        topk_indices.add_(block_indices)

        num_cache_rows = (
            kv_c_and_k_pe_cache.shape[0] - 1
        ) * block_row_stride + attn_metadata.block_size
        kv_cache = kv_c_and_k_pe_cache.as_strided(
            (num_cache_rows, physical_head_size),
            (physical_head_size, 1),
        )
        cu_seqlens_q = torch.arange(
            num_actual_toks + 1,
            dtype=torch.int32,
            device=q_rope.device,
        )
        output = flash_attn_varlen_func(
            q=q_rope,
            k=kv_cache[:, self.kv_lora_rank : self.head_size].unsqueeze(1).unsqueeze(1),
            v=kv_cache[:, : self.kv_lora_rank].unsqueeze(1).unsqueeze(1),
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,
        )
        return output, None
