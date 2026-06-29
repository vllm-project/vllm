# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
    sparse_prefill_combined_topk_size,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    _C128A_TOPK_ALIGNMENT,
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_triton_sparse_mla_enabled,
    is_triton_sparse_mla_enabled_for_platform,
    triton_sparse_mla_matmul_decode_enabled,
    triton_sparse_mla_prefill_topk_chunk_size,
    triton_sparse_mla_query_chunk_size,
    triton_sparse_mla_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_d512_chunked_sparse_mla_attention,
    accumulate_indexed_d512_split_sparse_mla_attention,
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
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


_INDEXED_D512_SPLIT_PREFILL_MIN_TOPK = 256
_INDEXED_D512_SPLIT_PREFILL_MAX_TOPK = 1152


def _use_indexed_d512_split_prefill(
    *,
    compress_ratio: int,
    head_dim: int,
    num_prefills: int,
    combined_topk: int,
    max_prefill_seq_len: int,
    swa_only: bool,
) -> bool:
    return (
        envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
        and not swa_only
        and compress_ratio in (4, 128)
        and head_dim == 512
        and num_prefills == 1
        and _is_indexed_d512_split_topk(combined_topk)
        and max_prefill_seq_len
        >= envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS
    )


def _is_indexed_d512_split_topk(combined_topk: int) -> bool:
    return (
        _INDEXED_D512_SPLIT_PREFILL_MIN_TOPK
        <= combined_topk
        <= _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
    )


def _use_indexed_d512_chunked_prefill(
    *,
    compress_ratio: int,
    head_dim: int,
    num_prefills: int,
    combined_topk: int,
    max_prefill_seq_len: int,
    swa_only: bool,
) -> bool:
    return (
        envs.VLLM_DEEPSEEK_V4_INDEXED_D512_CHUNKED_PREFILL
        and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
        and not swa_only
        and compress_ratio in (4, 128)
        and head_dim == 512
        and num_prefills == 1
        and combined_topk > _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
        and max_prefill_seq_len
        >= envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS
    )


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

    @classmethod
    def _prefill_workspace_topk_bound(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> int:
        if layer.compress_ratio <= 1:
            return 0
        if (
            layer.topk_indices_buffer is not None
            and layer.topk_indices_buffer.ndim > 0
            and layer.topk_indices_buffer.shape[-1] > 0
        ):
            bound = int(layer.topk_indices_buffer.shape[-1])
        else:
            indexer_topk = getattr(layer.indexer, "topk_tokens", None)
            bound = int(indexer_topk) if indexer_topk is not None else 2048
        # C128A prefill builds raw candidates over the full compressed region,
        # so its top-k width grows with max_model_len (independent of
        # index_topk) up to the c128a_max_compressed that the metadata builder
        # allocates c128a_prefill_buffer with. Reserve that worst case here so
        # the locked prefill workspace is sized self-consistently, instead of
        # depending on the lightning indexer's (much larger, incidental)
        # reservation to absorb the gap at long context.
        if layer.compress_ratio == 128:
            compressed = cdiv(int(layer.max_model_len), layer.compress_ratio)
            c128a_bound = (
                cdiv(compressed, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT
            )
            bound = max(bound, c128a_bound)
        return bound

    @classmethod
    def _prefill_workspace_reservation_specs(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        max_model_len = max(1, int(layer.max_model_len))
        max_num_batched_tokens = max(1, int(layer.max_num_batched_tokens))
        window_size = max(1, int(layer.window_size))
        compress_ratio = max(1, int(layer.compress_ratio))
        head_dim = int(layer.head_dim)
        num_heads = int(layer.n_local_heads)

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
            ((layer.PREFILL_CHUNK_SIZE, m_bound, head_dim), torch.bfloat16),
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
            if _use_indexed_d512_split_prefill(
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                num_prefills=1,
                combined_topk=combined_topk,
                max_prefill_seq_len=max_model_len,
                swa_only=False,
            ):
                specs.append(
                    ((query_chunk_size, num_heads, combined_topk), torch.float32)
                )
            elif _use_indexed_d512_chunked_prefill(
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                num_prefills=1,
                combined_topk=combined_topk,
                max_prefill_seq_len=max_model_len,
                swa_only=False,
            ):
                chunked_score_width = min(
                    combined_topk,
                    _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
                )
                specs.extend(
                    (
                        (
                            (query_chunk_size, num_heads, chunked_score_width),
                            torch.float32,
                        ),
                        ((query_chunk_size, num_heads), torch.float32),
                        ((query_chunk_size, num_heads), torch.float32),
                        ((query_chunk_size, num_heads, head_dim), torch.float32),
                    )
                )
        return tuple(specs)

    @classmethod
    def _reserve_prefill_workspace(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> None:
        try:
            workspace_manager = current_workspace_manager()
        except AssertionError:
            return
        workspace_manager.get_simultaneous(
            *cls._prefill_workspace_reservation_specs(layer)
        )

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
            # Warmup dummy run: no real metadata. Reserve the same graph-stable
            # workspace shapes _forward_prefill can use, but skip real kernels.
            self._reserve_prefill_workspace(self)
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

    @classmethod
    def _forward_sparse_mla_swa_decode_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        # Decode metadata is unconditionally populated when num_decode_tokens > 0,
        # which is the only path that reaches the decode kernels.
        assert swa_metadata.decode_swa_lens is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.seq_lens is not None
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
                num_heads=layer.n_local_heads,
            )
            if output.shape[1] > layer.n_local_heads:
                output[:, layer.n_local_heads :].zero_()
            return

        (
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
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
        if output.shape[1] > layer.n_local_heads:
            output[:, layer.n_local_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_compressed_decode_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata,
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

        # Decode metadata is unconditionally populated when num_decode_tokens > 0,
        # which is the only path that reaches the decode kernels.
        assert swa_metadata.decode_swa_lens is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.seq_lens is not None
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
                    (num_decode_tokens, layer.n_local_heads, total_candidates),
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
                num_heads=layer.n_local_heads,
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
                num_heads=layer.n_local_heads,
            )
            if output.shape[1] > layer.n_local_heads:
                output[:, layer.n_local_heads :].zero_()
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
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
        if output.shape[1] > layer.n_local_heads:
            output[:, layer.n_local_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_prefill_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
        state_buffers: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        kv_flat = kv.reshape(-1, q.shape[-1])
        topk_chunk_size = triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=combined_indices.shape[-1],
            compress_ratio=int(layer.compress_ratio),
            request_count=kv.shape[0],
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
                ((query_chunk_size, layer.n_local_heads), torch.float32),
                ((query_chunk_size, layer.n_local_heads), torch.float32),
                ((query_chunk_size, layer.n_local_heads, q.shape[-1]), torch.float32),
            )
        else:
            max_score_buffer, denom_buffer, output_buffer = state_buffers[:3]
        indexed_d512_scores = None
        indexed_d512_chunked_buffers = None
        if (
            state_buffers is not None
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
            and layer.compress_ratio in (4, 128)
            and q.shape[-1] == 512
            and kv.shape[0] == 1
            and _is_indexed_d512_split_topk(combined_indices.shape[-1])
            and len(state_buffers) == 4
        ):
            indexed_d512_scores = state_buffers[3]
        elif (
            state_buffers is not None
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_CHUNKED_PREFILL
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
            and layer.compress_ratio in (4, 128)
            and q.shape[-1] == 512
            and kv.shape[0] == 1
            and combined_indices.shape[-1] > _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
            and len(state_buffers) == 7
        ):
            indexed_d512_chunked_buffers = state_buffers[3:7]

        for token_start in range(0, q.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q.shape[0])
            q_chunk = q[token_start:token_end]
            indices_chunk_full = combined_indices[token_start:token_end]
            lens_chunk = combined_lens[token_start:token_end]
            num_tokens = token_end - token_start
            max_score = max_score_buffer[:num_tokens]
            denom = denom_buffer[:num_tokens]
            subset_acc = output_buffer[:num_tokens]
            can_use_indexed_d512_scores = (
                indexed_d512_scores is not None
                and indexed_d512_scores.shape[0] >= num_tokens
                and indexed_d512_scores.shape[2] >= combined_indices.shape[-1]
            )
            can_use_indexed_d512_chunked = (
                indexed_d512_chunked_buffers is not None
                and indexed_d512_chunked_buffers[0].shape[0] >= num_tokens
            )
            if can_use_indexed_d512_scores:
                assert indexed_d512_scores is not None
                accumulate_indexed_d512_split_sparse_mla_attention(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full,
                    lens=lens_chunk,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                    scores=indexed_d512_scores[
                        :num_tokens, :, : combined_indices.shape[-1]
                    ],
                )
            elif can_use_indexed_d512_chunked:
                assert indexed_d512_chunked_buffers is not None
                (
                    indexed_d512_scores,
                    chunk_max_score,
                    chunk_denom,
                    chunk_acc,
                ) = indexed_d512_chunked_buffers
                accumulate_indexed_d512_chunked_sparse_mla_attention(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full,
                    lens=lens_chunk,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                    scores=indexed_d512_scores[:num_tokens],
                    chunk_max_score=chunk_max_score[:num_tokens],
                    chunk_denom=chunk_denom[:num_tokens],
                    chunk_acc=chunk_acc[:num_tokens],
                )
            else:
                max_score.fill_(float("-inf"))
                denom.zero_()
                subset_acc.zero_()

                for index_start in range(
                    0, combined_indices.shape[-1], topk_chunk_size
                ):
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
            if output.shape[1] > layer.n_local_heads:
                output[token_start:token_end, layer.n_local_heads :].zero_()

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
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
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        if is_triton_sparse_mla_enabled(q.device):
            if swa_only:
                self._forward_sparse_mla_swa_decode_triton(
                    layer=self,
                    q=q,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                return
            if self.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                self._forward_sparse_mla_compressed_decode_triton(
                    layer=self,
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
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

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        assert seq_lens is not None
        assert gather_lens is not None
        assert seq_lens_cpu is not None

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
                # Rebase the indexer's BATCH-GLOBAL compressed top-k positions to
                # per-request-local. combine_topk_swa_indices maps a position p of
                # the k-th in-chunk request to gathered slot p + M*k, which is only
                # correct when p is request-local; the indexer writes cu_seqlen_ks-
                # cumulative (batch-global) positions, so without this rebase non-
                # first prefill requests index past their gathered slot and read
                # stale workspace (latent C4A multi-request prefill bug). torch.where
                # preserves -1 sentinels; no-op at num_prefills==1 (cu_base[0]==0).
                assert swa_metadata.token_to_req_indices is not None
                _comp_lens = seq_lens // self.compress_ratio
                _cu_base = (torch.cumsum(_comp_lens, dim=0) - _comp_lens).to(
                    torch.int32
                )
                _req_local = (
                    swa_metadata.token_to_req_indices[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ]
                    - num_decodes
                ).long()
                _base = _cu_base[_req_local].unsqueeze(1)
                topk_indices = torch.where(
                    topk_indices >= 0, topk_indices - _base, topk_indices
                )
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

        # Adaptive prefill chunk plan (#45061): pack as many requests as fit the
        # workspace-area bound into each chunk, with per-chunk compressed (chunk_N)
        # and total (chunk_M) widths. Replaces the fixed PREFILL_CHUNK_SIZE
        # chunking with batch-wide M/N.
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=int(self.compress_ratio),
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"

        max_query_chunk_tokens = 0
        for chunk_start, chunk_end, _chunk_n, _chunk_m in chunk_plan:
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            max_query_chunk_tokens = max(
                max_query_chunk_tokens, int(query_end - query_start)
            )
        combined_topk = sparse_prefill_combined_topk_size(top_k, self.window_size)

        workspace_manager = current_workspace_manager()
        triton_sparse_mla_enabled = is_triton_sparse_mla_enabled(q.device)
        indexed_d512_split_prefill = False
        indexed_d512_chunked_prefill = False
        extra_specs: list[tuple[tuple[int, ...], torch.dtype]] = []
        if triton_sparse_mla_enabled:
            query_chunk_size = min(
                max_query_chunk_tokens,
                triton_sparse_mla_query_chunk_size(),
            )
            indexed_d512_split_prefill = _use_indexed_d512_split_prefill(
                compress_ratio=int(self.compress_ratio),
                head_dim=int(self.head_dim),
                num_prefills=int(num_prefills),
                combined_topk=int(combined_topk),
                max_prefill_seq_len=int(seq_lens_cpu.max().item()),
                swa_only=swa_only,
            )
            if not indexed_d512_split_prefill:
                indexed_d512_chunked_prefill = _use_indexed_d512_chunked_prefill(
                    compress_ratio=int(self.compress_ratio),
                    head_dim=int(self.head_dim),
                    num_prefills=int(num_prefills),
                    combined_topk=int(combined_topk),
                    max_prefill_seq_len=int(seq_lens_cpu.max().item()),
                    swa_only=swa_only,
                )
            if indexed_d512_split_prefill:
                extra_specs.append(
                    (
                        (query_chunk_size, self.n_local_heads, combined_topk),
                        torch.float32,
                    )
                )
            elif indexed_d512_chunked_prefill:
                chunked_score_width = min(
                    combined_topk,
                    _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
                )
                extra_specs.extend(
                    (
                        (
                            (query_chunk_size, self.n_local_heads, chunked_score_width),
                            torch.float32,
                        ),
                        ((query_chunk_size, self.n_local_heads), torch.float32),
                        ((query_chunk_size, self.n_local_heads), torch.float32),
                        (
                            (query_chunk_size, self.n_local_heads, q.shape[-1]),
                            torch.float32,
                        ),
                    )
                )

        # Per-chunk workspace allocation (#45061): the kv buffer width is this
        # chunk's compressed+gather width (chunk_m), keeping the area bounded by
        # the planner's max_workspace_area instead of the batch-wide worst case.
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            if triton_sparse_mla_enabled:
                (
                    kv,
                    combined_indices_buffer,
                    combined_lens_buffer,
                    max_score_buffer,
                    denom_buffer,
                    output_buffer,
                    *extra_state_buffers,
                ) = workspace_manager.get_simultaneous(
                    ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
                    ((max_query_chunk_tokens, combined_topk), torch.int32),
                    ((max_query_chunk_tokens,), torch.int32),
                    ((query_chunk_size, self.n_local_heads), torch.float32),
                    ((query_chunk_size, self.n_local_heads), torch.float32),
                    (
                        (query_chunk_size, self.n_local_heads, q.shape[-1]),
                        torch.float32,
                    ),
                    *extra_specs,
                )
                prefill_state_buffers = (
                    max_score_buffer,
                    denom_buffer,
                    output_buffer,
                    *extra_state_buffers,
                )
            else:
                (
                    kv,
                    combined_indices_buffer,
                    combined_lens_buffer,
                ) = workspace_manager.get_simultaneous(
                    ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
                    ((max_query_chunk_tokens, combined_topk), torch.int32),
                    ((max_query_chunk_tokens,), torch.int32),
                )
                prefill_state_buffers = None

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
                offset=chunk_n,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

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
                chunk_m,
                chunk_n,
                combined_indices=combined_indices_buffer,
                combined_lens=combined_lens_buffer,
            )
            if triton_sparse_mla_enabled:
                self._forward_sparse_mla_prefill_triton(
                    self,
                    q=q[query_start:query_end],
                    kv=kv[:chunk_size],
                    combined_indices=combined_indices,
                    combined_lens=combined_lens,
                    output=output[query_start:query_end],
                    state_buffers=prefill_state_buffers,
                )
            else:
                flash_mla_sparse_fwd(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output[query_start:query_end],
                )
