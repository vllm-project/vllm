# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.nvidia.ops.o_proj import (
    dsv41_o_proj,
    register_dsv41_o_proj_warmup,
)
from vllm.models.deepseek_v41.nvidia.ops.packed_sparse_decode import (
    merge_slot_indices,
    pack_slot_rows,
    unpack_slot_rows,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport, max_decode_query_len
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta


class DeepseekSparseSWAFlashMLAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """SWA metadata for the FlashMLA decode path, which allows varlen decode."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashMLABackend(DeepseekSparseSWABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAFlashMLAMetadataBuilder]:
        return DeepseekSparseSWAFlashMLAMetadataBuilder


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4.1 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend
    swa_backend_cls = DeepseekSparseSWAFlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        register_dsv41_o_proj_warmup(self)
        # Spec-decode packing: each request's 1 + K positions share one head
        # tile, n_local_heads rows apiece.
        self._pack_slots = max_decode_query_len(get_current_vllm_config())
        self._pack_spec_decode = (
            envs.VLLM_DSV41_PACK_SPEC_DECODE
            and current_platform.is_device_capability_family(90)
            and self.padded_heads == 64
            and self._pack_slots > 1
            and self._pack_slots * self.n_local_heads <= self.padded_heads
        )
        self._packed_attn_sink: torch.Tensor | None = None

    def _get_packed_attn_sink(self) -> torch.Tensor:
        # Built on first use so it sees the loaded sink. Tile row r holds head
        # r % n_local_heads; rows past the packed slots stay -inf.
        if self._packed_attn_sink is None:
            sink = torch.full_like(self.attn_sink, float("-inf"))
            n = self._pack_slots * self.n_local_heads
            sink[:n] = self.attn_sink[: self.n_local_heads].repeat(self._pack_slots)
            self._packed_attn_sink = sink
        return self._packed_attn_sink

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio == 0
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            if swa_only:
                top_k = 0
            else:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        if self.use_pcp:
            num_tokens = (
                swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens
            )
            output[num_tokens:].zero_()
            if num_tokens == 0:
                return

        swa_only = self.compress_ratio == 0
        # SWA-only layers (compress_ratio == 0) don't have their own KV cache
        # allocation; consumers read the kv source's cache, which is bound on
        # the source layer.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            global_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            topk_indices = global_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        tile_metadata = {
            0: swa_metadata.tile_sched_swaonly,
            1: swa_metadata.tile_sched_c1a,
            2: swa_metadata.tile_sched_c2a,
        }[self.compress_ratio]
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        # Pack only when every decode request is exactly 1 + K tokens wide.
        if (
            self._pack_spec_decode
            and num_decodes > 0
            and num_decode_tokens == num_decodes * self._pack_slots
        ):
            self._forward_decode_packed(
                q.squeeze(1),
                swa_cache,
                kv_cache if not swa_only else None,
                swa_metadata,
                topk_indices,
                topk_lens,
                tile_metadata,
                output,
            )
            return

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_decode_packed(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        topk_indices: torch.Tensor | None,
        topk_lens: torch.Tensor | None,
        tile_metadata: "FlashMLASchedMeta",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        rows = self.n_local_heads
        slots = self._pack_slots

        # The SWA union depends on the layer's KV-cache group block table, so
        # it lives on that group's metadata. The top-k union and its split plan
        # depend only on the index source and the compressed cache, so every
        # consumer in any group shares them for the whole step.
        swa_cache_union = swa_metadata.packed_decode_cache
        swa_key = "swa"
        if swa_key not in swa_cache_union:
            assert swa_metadata.decode_swa_indices is not None
            assert swa_metadata.decode_swa_lens is not None
            swa_cache_union[swa_key] = merge_slot_indices(
                swa_metadata.decode_swa_indices.view(num_decode_tokens, -1),
                swa_metadata.decode_swa_lens,
                slots,
            )
        swa_idx, swa_mask, swa_len = swa_cache_union[swa_key]
        extra_idx = extra_mask = extra_len = None
        if topk_indices is not None:
            assert topk_lens is not None
            cache = get_forward_context().additional_kwargs.setdefault(
                "dsv41_packed_decode", {}
            )
            topk_key = (
                "topk",
                self.kv_source_layer_id,
                self.index_source_layer_id,
                self.compress_ratio,
            )
            if topk_key not in cache:
                cache[topk_key] = merge_slot_indices(
                    topk_indices.view(num_decode_tokens, -1),
                    topk_lens,
                    slots,
                )
            extra_idx, extra_mask, extra_len = cache[topk_key]
            # Union lengths differ between index sources, while a split plan
            # is only valid for the lengths it was built from.
            sched_key = ("sched", *topk_key)
            if sched_key not in cache:
                cache[sched_key] = (get_mla_metadata()[0],)
            tile_metadata = cache[sched_key][0]

        q_packed = pack_slot_rows(q, num_decodes, slots, rows, self.padded_heads)
        out_packed, _ = flash_mla_with_kvcache(
            q=q_packed,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_idx,
            topk_length=swa_len,
            softmax_scale=self.scale,
            attn_sink=self._get_packed_attn_sink(),
            extra_k_cache=kv_cache,
            extra_indices_in_kvcache=extra_idx,
            extra_topk_length=extra_len,
            indices_mask=swa_mask,
            extra_indices_mask=extra_mask,
            rows_per_slot=rows,
        )
        unpack_slot_rows(out_packed, output, slots, rows)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            # v4.1: every cr>0 layer (including cr==1) gathers a full-length
            # compressed region; only cr==0 layers are SWA-only.
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            workspace = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            kv, combined_indices_out, combined_lens_out = workspace
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_out = combined_indices_out[: query_end - query_start]
            combined_lens_out = combined_lens_out[: query_end - query_start]

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out, combined_lens_out),
            )
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
