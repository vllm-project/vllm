# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse Attention Indexer — Triton + PyTorch.

Registered as ``torch.ops.vllm.dsv4_sparse_attn_indexer`` so the agnostic
op coexists with the HW-specific ``sparse_attn_indexer`` registration of
the same op family.
"""

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.hw_agnostic.attention.indexer import (
    DeepseekV4IndexerMetadata,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.ops.fp8_logits_torch import (
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits_torch,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.ops.indexer_quant_cache import (
    cp_gather_indexer_k_quant_cache_triton,
    indexer_k_quant_and_cache_triton,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.ops.seq_packing import (
    pack_seq_triton,
    unpack_seq_triton,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.ops.topk_torch import (
    topk_indices_torch,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace: ``(T, head_dim)`` fp8 + ``(T, 4)`` uint8 fp32
    scales."""
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view from the 3D
    indexer kv-cache allocation."""
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def dsv4_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
) -> torch.Tensor:
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if not isinstance(attn_metadata, dict):
        # Profile run: attn_metadata is not a dict. Reserve the indexer
        # K-gather workspace and the peak logits allocation so memory
        # planning sees real high-water marks, then return the fake op.
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
        )

        # FP8 elements so elements == bytes.
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return dsv4_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV4IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        indexer_k_quant_and_cache_triton(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first
        # use). Layout: ``head_dim`` fp8 bytes + 4-byte fp32 scale per token.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                cp_gather_indexer_k_quant_cache_triton(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    chunk.token_to_seq,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = fp8_mqa_logits_torch(
                q_slice,
                (k_quant, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            topk_indices.copy_(
                topk_indices_torch(logits, topk_tokens, row_starts=chunk.cu_seqlen_ks)
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Pad short chunked-prefill rows whose query_len < decode_threshold
            # (= 1 + speculative tokens). FP8 Q (float8_e4m3fn) tolerates the
            # default fp32 ``-inf`` pad — context_lens masks stale slots
            # downstream.
            padded_q_quant_decode_tokens = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]

        logits = fp8_paged_mqa_logits_torch(
            padded_q_quant_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            max_model_len,
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        topk_indices.copy_(topk_indices_torch(logits, topk_tokens))

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def dsv4_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="dsv4_sparse_attn_indexer",
    op_func=dsv4_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=dsv4_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class SparseAttnIndexer(nn.Module):
    """Sparse Attention Indexer layer (Triton + PyTorch)."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # Per-token Q scale is folded into ``weights`` earlier in the
        # pipeline (``fused_indexer_q_rope_quant``); the kernel sees a
        # single quantized tensor.
        return torch.ops.vllm.dsv4_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
        )
