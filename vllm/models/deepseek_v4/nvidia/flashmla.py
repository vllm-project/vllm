# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _batch_invariant_prefill_chunk_plan(
    metadata: "DeepseekSparseSWAMetadata",
    compress_ratio: int,
    window_size: int,
) -> list[tuple[int, int, int, int]]:
    if metadata.num_prefills == 0:
        return []
    assert metadata.prefill_seq_lens_cpu is not None
    assert metadata.prefill_query_lens_cpu is not None
    prefix_lens = metadata.prefill_seq_lens_cpu - metadata.prefill_query_lens_cpu
    gather_lens = metadata.prefill_query_lens_cpu + torch.clamp(
        prefix_lens, min=0, max=window_size - 1
    )
    compressed_lens = (
        torch.zeros_like(metadata.prefill_seq_lens_cpu)
        if compress_ratio <= 1
        else torch.div(
            metadata.prefill_seq_lens_cpu,
            compress_ratio,
            rounding_mode="floor",
        )
    )
    query_lens = metadata.prefill_query_lens_cpu
    compressed_values = compressed_lens.numpy()
    gather_values = gather_lens.numpy()
    query_values = query_lens.numpy()
    plan: list[tuple[int, int, int, int]] = []
    chunk_start = 0
    while chunk_start < metadata.num_prefills:
        chunk_n = int(compressed_values[chunk_start])
        chunk_gather = int(gather_values[chunk_start])
        chunk_query = int(query_values[chunk_start])
        chunk_end = chunk_start + 1
        while chunk_end < metadata.num_prefills and (
            int(compressed_values[chunk_end]),
            int(gather_values[chunk_end]),
            int(query_values[chunk_end]),
        ) == (chunk_n, chunk_gather, chunk_query):
            chunk_end += 1
        plan.append((chunk_start, chunk_end, chunk_n, chunk_n + chunk_gather))
        chunk_start = chunk_end
    return plan


def _batch_invariant_decode_request_ranges(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_decodes: int,
    compress_ratio: int,
    window_size: int,
) -> list[tuple[int, int, int, int]]:
    query_offsets = query_start_loc_cpu.numpy()
    seq_values = seq_lens_cpu.numpy()
    ranges: list[tuple[int, int, int, int]] = []
    request_start = 0
    while request_start < num_decodes:
        token_start = int(query_offsets[request_start])
        query_len = int(query_offsets[request_start + 1]) - token_start
        seq_len = int(seq_values[request_start])
        compressed_len = 0 if compress_ratio <= 1 else seq_len // compress_ratio
        gather_len = query_len + min(max(seq_len - query_len, 0), window_size - 1)
        request_end = request_start + 1
        while request_end < num_decodes:
            next_token_start = int(query_offsets[request_end])
            next_query_len = int(query_offsets[request_end + 1]) - next_token_start
            next_seq_len = int(seq_values[request_end])
            next_shape = (
                next_query_len,
                0 if compress_ratio <= 1 else next_seq_len // compress_ratio,
                next_query_len
                + min(max(next_seq_len - next_query_len, 0), window_size - 1),
            )
            if next_shape != (query_len, compressed_len, gather_len):
                break
            request_end += 1
        ranges.append(
            (
                request_start,
                token_start,
                request_end - request_start,
                int(query_offsets[request_end]) - token_start,
            )
        )
        request_start = request_end
    return ranges


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

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
            swa_only = self.compress_ratio <= 1
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
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
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
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        decode_kernel = envs.VLLM_DS4_DECODE_KERNEL.lower()
        if decode_kernel == "sparse":
            self._forward_decode_sparse(
                q=q,
                compressed_k_cache=kv_cache,
                swa_k_cache=self.swa_cache_layer.kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=attn_metadata,
                swa_only=swa_only,
                output=output,
            )
            return
        if decode_kernel != "paged":
            # envs validates this eagerly; keep the dispatch fail-closed if a
            # test or embedding overrides the value after initialization.
            raise ValueError(
                "VLLM_DS4_DECODE_KERNEL must be 'paged' or 'sparse', "
                f"got {decode_kernel!r}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                    output_buffers=self._global_topk_output_buffers(
                        self.topk_indices_buffer[:num_decode_tokens]
                    ),
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

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
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

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

    def _forward_decode_sparse(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        *,
        request_start: int = 0,
        token_start: int = 0,
        request_count: int | None = None,
    ) -> None:
        """Run real decode requests through the prefill sparse FlashMLA kernel."""
        num_decodes = (
            swa_metadata.num_decodes if request_count is None else request_count
        )
        num_decode_tokens = q.shape[0]
        if num_decodes <= 0 or num_decode_tokens <= 0:
            raise RuntimeError("decode-sparse requires at least one decode request")
        if swa_metadata.decode_swa_indices is None:
            raise RuntimeError("decode-sparse requires decode SWA indices")
        if swa_metadata.decode_swa_indices.shape[-1] != self.window_size:
            raise RuntimeError(
                "decode-sparse only supports causal SWA metadata with width "
                f"{self.window_size}, got {swa_metadata.decode_swa_indices.shape[-1]}"
            )
        if (
            swa_metadata.seq_lens is None
            or swa_metadata.seq_lens_cpu is None
            or swa_metadata.query_start_loc is None
            or swa_metadata.query_start_loc_cpu is None
            or swa_metadata.is_valid_token is None
        ):
            raise RuntimeError("decode-sparse requires finalized scheduler metadata")

        if (
            envs.VLLM_BATCH_INVARIANT
            and request_count is None
            and num_decodes > 1
        ):
            request_ranges = _batch_invariant_decode_request_ranges(
                swa_metadata.query_start_loc_cpu,
                swa_metadata.seq_lens_cpu,
                num_decodes,
                self.compress_ratio,
                self.window_size,
            )
            if len(request_ranges) > num_decodes:
                raise RuntimeError("decode-sparse BI launch count exceeded request count")
            for request_index, begin, request_count_, token_count in request_ranges:
                self._forward_decode_sparse(
                    q[begin : begin + token_count],
                    compressed_k_cache,
                    swa_k_cache,
                    swa_metadata,
                    attn_metadata,
                    swa_only,
                    output[begin : begin + token_count],
                    request_start=request_index,
                    token_start=begin,
                    request_count=request_count_,
                )
            return

        request_end = request_start + num_decodes
        token_end = token_start + num_decode_tokens
        seq_lens = swa_metadata.seq_lens[request_start:request_end]
        seq_lens_cpu = swa_metadata.seq_lens_cpu[request_start:request_end]
        query_start_loc = (
            swa_metadata.query_start_loc[request_start : request_end + 1]
            - token_start
        )
        query_start_loc_cpu = (
            swa_metadata.query_start_loc_cpu[request_start : request_end + 1]
            - token_start
        )
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        prefix_lens_cpu = seq_lens_cpu - query_lens_cpu
        gather_lens_cpu = query_lens_cpu + torch.clamp(
            prefix_lens_cpu, min=0, max=self.window_size - 1
        )
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        prefix_lens = seq_lens - query_lens
        gather_lens = query_lens + torch.clamp(
            prefix_lens, min=0, max=self.window_size - 1
        )

        query_lens_values = query_lens_cpu.numpy()
        if int(query_lens_values.sum()) != num_decode_tokens:
            raise RuntimeError(
                "decode-sparse query metadata does not match decode token count"
            )

        if swa_only:
            assert self.topk_indices_buffer is not None
            local_topk = self.topk_indices_buffer[token_start:token_end]
            top_k = 0
            max_compressed = 0
        else:
            if attn_metadata is None or compressed_k_cache is None:
                raise RuntimeError(
                    "compressed decode-sparse requires attention metadata and KV cache"
                )
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                local_topk = self.topk_indices_buffer[token_start:token_end]
                top_k = local_topk.shape[-1]
            elif self.compress_ratio == 128:
                if attn_metadata.c128a_global_decode_topk_indices is None:
                    raise RuntimeError(
                        "C128 decode-sparse requires finalized top-k metadata"
                    )
                top_k = attn_metadata.c128a_global_decode_topk_indices.shape[-1]
                local_topk = None
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )
            max_compressed = int(
                (seq_lens_cpu.numpy() // self.compress_ratio).max()
            )

        max_gather = int(gather_lens_cpu.numpy().max())
        workspace_width = max_compressed + max_gather
        combined_topk = round_up(top_k + self.window_size, 128)
        specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((num_decodes, workspace_width, q.shape[-1]), torch.bfloat16),
            ((num_decode_tokens, combined_topk), torch.int32),
            ((num_decode_tokens,), torch.int32),
        ]
        if not swa_only and self.compress_ratio == 128:
            specs.append(((num_decode_tokens, top_k), torch.int32))
        workspace = current_workspace_manager().get_simultaneous(*specs)
        kv, combined_indices_out, combined_lens_out = workspace[:3]

        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            dequantize_and_gather_k_cache(
                kv,
                compressed_k_cache,
                seq_lens=torch.div(
                    seq_lens, self.compress_ratio, rounding_mode="floor"
                ),
                gather_lens=None,
                block_table=attn_metadata.block_table[request_start:request_end],
                block_size=attn_metadata.block_size // self.compress_ratio,
                offset=0,
            )
        dequantize_and_gather_k_cache(
            kv,
            swa_k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=swa_metadata.block_table[request_start:request_end],
            block_size=swa_metadata.block_size,
            offset=max_compressed,
        )

        if not swa_only and self.compress_ratio == 128:
            local_topk = workspace[3]
            torch.arange(
                top_k,
                dtype=torch.int32,
                device=q.device,
                out=local_topk[0],
            )
            if num_decode_tokens > 1:
                local_topk[1:].copy_(local_topk[0])
            # ``top_k`` is the graph-stable padded C128 width, not the number
            # of compressed rows currently visible to each token.  Leaving
            # the padded tail as arange values makes BI canonicalization sort
            # an invalid high index into the valid prefix at the first C128
            # boundary (for example 127 instead of 0 when the true length is
            # one).  Prefill metadata already represents this tail as -1.
            assert attn_metadata.c128a_decode_topk_lens is not None
            local_topk.masked_fill_(
                local_topk
                >= attn_metadata.c128a_decode_topk_lens[token_start:token_end, None],
                -1,
            )
        assert local_topk is not None
        combined_indices, combined_lens = combine_topk_swa_indices(
            local_topk,
            query_start_loc,
            seq_lens,
            gather_lens,
            self.window_size,
            self.compress_ratio,
            top_k,
            workspace_width,
            max_compressed,
            out=(combined_indices_out, combined_lens_out),
        )
        combined_lens.masked_fill_(
            ~swa_metadata.is_valid_token[token_start:token_end], 0
        )
        flash_mla_sparse_fwd(
            q=q,
            kv=kv.view(-1, 1, q.shape[-1]),
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            topk_length=combined_lens,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

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

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
        if envs.VLLM_BATCH_INVARIANT:
            # The grouped prefill path chooses shared workspace dimensions and
            # offsets from all requests in a chunk. Its end-to-end attention
            # output is therefore not bitwise invariant even though the sparse
            # FlashMLA kernel itself is invariant for identical q/kv/indices.
            # Keep each request in its own chunk in BI mode so N, M, gathers,
            # indices and kernel launch geometry depend only on that request.
            chunk_plan = _batch_invariant_prefill_chunk_plan(
                swa_metadata,
                self.compress_ratio,
                self.window_size,
            )
            num_prefills = swa_metadata.num_prefills
            if len(chunk_plan) > num_prefills:
                raise RuntimeError("prefill BI launch count exceeded request count")
        else:
            chunk_plan = swa_metadata.get_prefill_chunk_plan(
                compress_ratio=self.compress_ratio,
                prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
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
