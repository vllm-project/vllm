# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
    sparse_prefill_combined_topk_size,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_triton_sparse_mla_enabled,
    is_triton_sparse_mla_enabled_for_platform,
    triton_sparse_mla_matmul_decode_enabled,
    triton_sparse_mla_query_chunk_size,
    triton_sparse_mla_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_sparse_mla_attention_chunk,
    build_combined_sparse_mla_decode_valid_mask,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    sparse_mla_decode_head_block_size,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.nvidia.ops.attention import (
        DeepseekV4MLAAttention,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _sparse_mla_prefill_workspace_bounds(
    seq_lens_cpu: torch.Tensor,
    gather_lens_cpu: torch.Tensor,
    compress_ratio: int,
    swa_only: bool,
) -> tuple[int, int]:
    if seq_lens_cpu.numel() == 0:
        return 0, 0

    max_gather_len = int(gather_lens_cpu.max().item())
    if swa_only:
        return 0, max_gather_len

    compressed_region_size = int((seq_lens_cpu // compress_ratio).max().item())
    return compressed_region_size, compressed_region_size + max_gather_len


def _sparse_mla_prefill_gather_len_upper_bound(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    window_size: int,
) -> tuple[int, int]:
    max_query_chunk_tokens = max(1, min(max_model_len, max_num_batched_tokens))
    max_prefix_len = max(max_model_len - max_query_chunk_tokens, 0)
    max_gather_len = max_query_chunk_tokens + min(
        max_prefix_len,
        max(window_size - 1, 0),
    )
    return max_query_chunk_tokens, max_gather_len


class DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """Abstract parent for DeepseekV4 sparse MLA impls.

    V4 sparse MLA is driven by the layer (``DeepseekV4MLAAttention.forward``)
    rather than the v1 framework, so ``forward_mqa`` is overridden with a
    classmethod that takes the layer as its first argument. This Liskov-broken
    override is intentional: the grandparent's instance-method ``forward_mqa``
    is never called on V4 layers.
    """

    backend_cls: ClassVar[type[AttentionBackend]]

    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the V4 layer's
    # dummy-run path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        kv_workspace: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError


class DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "V4_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SparseMLAAttentionImpl"]:
        return DeepseekV4FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512 (overrides the
        # V3.2 default of 576 from FlashMLASparseBackend).
        return [512]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)


class DeepseekV4FlashMLASparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """FlashMLA sparse MLA implementation for DeepSeek V4's custom MLA layer."""

    backend_cls = DeepseekV4FlashMLASparseBackend

    @classmethod
    def _prefill_workspace_topk_bound(
        cls,
        layer: "DeepseekV4MLAAttention",
    ) -> int:
        if layer.compress_ratio <= 1:
            return 0
        if (
            layer.topk_indices_buffer is not None
            and layer.topk_indices_buffer.ndim > 0
            and layer.topk_indices_buffer.shape[-1] > 0
        ):
            return int(layer.topk_indices_buffer.shape[-1])
        indexer_topk = getattr(layer.indexer, "topk_tokens", None)
        if indexer_topk is not None:
            return int(indexer_topk)
        return 2048

    @classmethod
    def _prefill_workspace_reservation_specs(
        cls,
        layer: "DeepseekV4MLAAttention",
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        max_model_len = max(1, int(layer.max_model_len))
        max_num_batched_tokens = max(1, int(layer.max_num_batched_tokens))
        window_size = max(1, int(layer.window_size))
        compress_ratio = max(1, int(layer.compress_ratio))
        head_dim = int(layer.head_dim)
        num_heads = int(layer.num_heads)

        max_query_chunk_tokens, max_gather_len = (
            _sparse_mla_prefill_gather_len_upper_bound(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                window_size=window_size,
            )
        )
        if compress_ratio <= 1:
            m_bound = max_gather_len
        else:
            compressed_region_size = max_model_len // compress_ratio
            m_bound = compressed_region_size + max_gather_len

        combined_topk = sparse_prefill_combined_topk_size(
            cls._prefill_workspace_topk_bound(layer),
            window_size,
        )
        specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((cls.PREFILL_CHUNK_SIZE, m_bound, head_dim), torch.bfloat16),
            ((max_query_chunk_tokens, combined_topk), torch.int32),
            ((max_query_chunk_tokens,), torch.int32),
        ]
        if is_triton_sparse_mla_enabled_for_platform():
            query_chunk_size = min(
                max_query_chunk_tokens,
                triton_sparse_mla_query_chunk_size(),
            )
            specs.extend(
                [
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads, head_dim), torch.float32),
                ]
            )
        return tuple(specs)

    @classmethod
    def _reserve_prefill_workspace(
        cls,
        layer: "DeepseekV4MLAAttention",
    ) -> None:
        try:
            workspace_manager = current_workspace_manager()
        except AssertionError:
            return
        workspace_manager.get_simultaneous(
            *cls._prefill_workspace_reservation_specs(layer)
        )

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        kv_workspace: torch.Tensor | None = None,
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
            # Warmup dummy run: no real metadata. Reserve the same workspace
            # shapes _forward_prefill would use so CUDA graph replay sees
            # stable addresses, but skip the real sparse/dequant kernels.
            cls._reserve_prefill_workspace(layer)
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so layer.kv_cache may be empty after profiling cleanup.
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_kv_cache = layer.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            cls._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                kv_workspace=kv_workspace,
            )
        if num_decodes > 0:
            cls._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_sparse_mla_swa_decode_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if not mtp_decode:
            fp8ds_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.num_heads,
            )
            if output.shape[1] > layer.num_heads:
                output[:, layer.num_heads :].zero_()
            return

        (
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
        )
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=layer.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_sparse_mla_attention_with_sink(
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.num_heads:
            output[:, layer.num_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_compressed_decode_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata,
        output: torch.Tensor,
    ) -> None:
        if layer.compress_ratio not in (4, 128):
            raise NotImplementedError(
                "Triton sparse MLA compressed decode currently supports "
                f"compress_ratio=4 or 128, got {layer.compress_ratio}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        compressed_block_size = attn_metadata.block_size // layer.compress_ratio
        compressed_topk = topk_indices.shape[-1]
        topk_chunk_size = min(
            compressed_topk,
            triton_sparse_mla_topk_chunk_size(),
        )
        compressed_slot_ids = topk_indices[:, 0, :]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if (
            compressed_topk <= topk_chunk_size
            and triton_sparse_mla_matmul_decode_enabled()
        ):
            total_candidates = compressed_topk + max_swa_len
            (
                combined_kv,
                valid_tokens,
                score_buffer,
            ) = current_workspace_manager().get_simultaneous(
                ((num_decode_tokens, total_candidates, q.shape[-1]), torch.bfloat16),
                ((num_decode_tokens, total_candidates), torch.bool),
                (
                    (num_decode_tokens, layer.num_heads, total_candidates),
                    torch.bfloat16,
                ),
            )
            if mtp_decode:
                dequantize_global_slots_k_cache(
                    combined_kv[:, :compressed_topk],
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                )
                dequantize_global_slots_k_cache(
                    combined_kv[:, compressed_topk:],
                    swa_k_cache,
                    swa_indices,
                    swa_metadata.block_size,
                )
            else:
                dequantize_combined_sparse_mla_decode_kv(
                    combined_kv,
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                    swa_k_cache,
                    swa_metadata.seq_lens[:num_decodes],
                    swa_lens,
                    swa_metadata.block_table[:num_decodes],
                    swa_metadata.block_size,
                )

            build_combined_sparse_mla_decode_valid_mask(
                valid_tokens,
                compressed_slot_ids,
                topk_lens,
                swa_lens,
            )
            use_dot_finish = num_decode_tokens <= 16
            matmul_sparse_mla_attention_with_sink(
                q=q,
                kv=combined_kv,
                valid_tokens=valid_tokens,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                num_heads=layer.num_heads,
                score_buffer=score_buffer,
                value_block_size=512 if use_dot_finish else 256,
                candidate_block_size=128 if use_dot_finish else None,
            )
            return

        if not mtp_decode and compressed_topk <= topk_chunk_size:
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids,
                topk_lens=topk_lens,
                compressed_block_size=compressed_block_size,
                swa_k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                swa_block_size=swa_metadata.block_size,
                num_compressed_candidates=compressed_topk,
                num_swa_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.num_heads,
            )
            if output.shape[1] > layer.num_heads:
                output[:, layer.num_heads :].zero_()
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
        )
        comp_max_score.fill_(float("-inf"))
        comp_denom.zero_()
        comp_acc.zero_()
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()

        for chunk_start in range(0, compressed_topk, topk_chunk_size):
            chunk_end = min(chunk_start + topk_chunk_size, compressed_topk)
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids[:, chunk_start:chunk_end],
                lens=topk_lens,
                block_size=compressed_block_size,
                candidate_offset=chunk_start,
                scale=layer.scale,
                max_score=comp_max_score,
                denom=comp_denom,
                acc=comp_acc,
                head_block_size=head_block_size,
            )
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=layer.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.num_heads:
            output[:, layer.num_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_prefill_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
        state_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        kv_flat = kv.reshape(-1, q.shape[-1])
        topk_chunk_size = min(
            combined_indices.shape[-1],
            triton_sparse_mla_topk_chunk_size(),
        )
        query_chunk_size = min(
            q.shape[0],
            triton_sparse_mla_query_chunk_size(),
        )
        if state_buffers is None:
            (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = current_workspace_manager().get_simultaneous(
                ((query_chunk_size, layer.num_heads), torch.float32),
                ((query_chunk_size, layer.num_heads), torch.float32),
                ((query_chunk_size, layer.num_heads, q.shape[-1]), torch.float32),
            )
        else:
            max_score_buffer, denom_buffer, output_buffer = state_buffers

        for token_start in range(0, q.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q.shape[0])
            q_chunk = q[token_start:token_end]
            indices_chunk_full = combined_indices[token_start:token_end]
            lens_chunk = combined_lens[token_start:token_end]
            num_tokens = token_end - token_start
            max_score = max_score_buffer[:num_tokens]
            denom = denom_buffer[:num_tokens]
            subset_acc = output_buffer[:num_tokens]
            max_score.fill_(float("-inf"))
            denom.zero_()
            subset_acc.zero_()

            for index_start in range(0, combined_indices.shape[-1], topk_chunk_size):
                index_end = min(
                    index_start + topk_chunk_size,
                    combined_indices.shape[-1],
                )
                accumulate_indexed_sparse_mla_attention_chunk(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full[:, index_start:index_end],
                    lens=lens_chunk,
                    candidate_offset=index_start,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                )

            finish_sparse_mla_attention_with_sink(
                max_score,
                denom,
                subset_acc,
                layer.attn_sink,
                output=output[token_start:token_end],
            )
            if output.shape[1] > layer.num_heads:
                output[token_start:token_end, layer.num_heads :].zero_()

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // layer.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert layer.topk_indices_buffer is not None
                local_topk_indices = layer.topk_indices_buffer[:num_decode_tokens]
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                    global_topk_indices=local_topk_indices,
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
        # q arrives pre-padded to layer.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        if is_triton_sparse_mla_enabled(q.device):
            if swa_only:
                cls._forward_sparse_mla_swa_decode_triton(
                    layer=layer,
                    q=q,
                    swa_k_cache=layer.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                return
            if layer.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                cls._forward_sparse_mla_compressed_decode_triton(
                    layer=layer,
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=layer.swa_cache_layer.kv_cache,
                    topk_indices=topk_indices,
                    topk_lens=topk_lens,
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                return

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if layer.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif layer.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif layer.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={layer.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={layer.compress_ratio}; "
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
            softmax_scale=layer.scale,
            attn_sink=layer.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        kv_workspace: torch.Tensor | None = None,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        gather_lens_cpu = swa_metadata.prefill_gather_lens_cpu
        assert seq_lens is not None
        assert gather_lens is not None
        assert seq_lens_cpu is not None
        assert gather_lens_cpu is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        N, M = _sparse_mla_prefill_workspace_bounds(
            seq_lens_cpu=seq_lens_cpu,
            gather_lens_cpu=gather_lens_cpu,
            compress_ratio=layer.compress_ratio,
            swa_only=swa_only,
        )
        chunk_size_const = cls.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const
        max_query_chunk_tokens = 0
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            max_query_chunk_tokens = max(
                max_query_chunk_tokens, int(query_end - query_start)
            )
        combined_topk = sparse_prefill_combined_topk_size(top_k, layer.window_size)

        workspace_manager = current_workspace_manager()
        triton_sparse_mla_enabled = is_triton_sparse_mla_enabled(q.device)
        if triton_sparse_mla_enabled:
            query_chunk_size = min(q.shape[0], triton_sparse_mla_query_chunk_size())
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = workspace_manager.get_simultaneous(
                ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
                ((query_chunk_size, layer.num_heads), torch.float32),
                ((query_chunk_size, layer.num_heads), torch.float32),
                ((query_chunk_size, layer.num_heads, q.shape[-1]), torch.float32),
            )
            prefill_state_buffers = (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            )
        else:
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
            ) = workspace_manager.get_simultaneous(
                ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
            )
            prefill_state_buffers = None
        # When the wrapper's attention_impl has pre-gathered KV into
        # kv_workspace on an aux stream (overlapped with the indexer), use
        # that buffer in place of the per-chunk gather below. The workspace
        # allocation in attention_impl aliases offset 0 of the same per-ubatch
        # workspace buffer as ``kv`` here, but we route through the explicit
        # parameter so the contract stays visible at the call site.
        _kv = kv_workspace if kv_workspace is not None else kv
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if kv_workspace is None:
                if not swa_only:
                    # Gather compressed KV
                    assert attn_metadata is not None
                    block_table = attn_metadata.block_table[num_decodes:]
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end]
                        // layer.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // layer.compress_ratio,
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
                    offset=N,
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            query_tokens = query_end - query_start
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                layer.window_size,
                layer.compress_ratio,
                top_k,
                M,
                N,
                combined_indices=combined_indices_buffer[:query_tokens],
                combined_lens=combined_lens_buffer[:query_tokens],
            )

            if triton_sparse_mla_enabled:
                cls._forward_sparse_mla_prefill_triton(
                    layer=layer,
                    q=q[query_start:query_end],
                    kv=_kv[:chunk_size],
                    combined_indices=combined_indices,
                    combined_lens=combined_lens,
                    output=output[query_start:query_end],
                    state_buffers=prefill_state_buffers,
                )
                continue

            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=_kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=layer.scale,
                attn_sink=layer.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
