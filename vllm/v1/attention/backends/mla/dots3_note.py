# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3 NOTE sliding-window MLA execution on Hopper.

Prefill and mixed batches expand the latent cache and use FlashAttention-3
varlen MHA. Decode-only batches use the Triton absorbed-MQA kernel.
"""

from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
    _compute_num_kv_splits,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


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

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.sliding_window = kv_cache_spec.sliding_window
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def _reserve_attn_logits_workspace(self) -> None:
        if not is_workspace_manager_initialized():
            return
        batch = self.vllm_config.scheduler_config.max_num_seqs
        q_num_heads = self.num_heads * self.dcp_world_size
        max_splits = _compute_num_kv_splits(
            self.sliding_window, current_platform.num_compute_units()
        )
        current_workspace_manager().get_simultaneous(
            (
                (
                    batch,
                    q_num_heads,
                    max_splits,
                    self.mla_dims.kv_lora_rank + 1,
                ),
                torch.float32,
            ),
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
        metadata.prefill.sliding_window = sliding_metadata
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
                kv_lens = torch.minimum(
                    seq_lens, query_lens + self.sliding_window - 1
                )
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

        batch, q_num_heads = q.shape[:2]
        output = torch.zeros(
            batch,
            q_num_heads,
            self.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.zeros(batch, q_num_heads, dtype=q.dtype, device=q.device)
        num_kv_splits = (
            1
            if envs.VLLM_BATCH_INVARIANT
            else _compute_num_kv_splits(
                min(attn_metadata.max_seq_len, self.sliding_window), self._sm_count
            )
        )
        logits_shape = (
            batch,
            q_num_heads,
            num_kv_splits,
            self.kv_lora_rank + 1,
        )
        if is_workspace_manager_initialized():
            (attn_logits,) = current_workspace_manager().get_simultaneous(
                (logits_shape, torch.float32),
            )
        else:
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=q.device
            )

        kv_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_cache[..., : self.kv_lora_rank]
        decode_attention_fwd(
            q,
            kv_cache,
            kv_c_cache,
            output,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            kv_cache.size(1),
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
            sliding_window=self.sliding_window,
        )
        return output, lse
