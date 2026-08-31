# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp weight-free QSA indexer."""

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    canonical_qsa_rope_positions,
)
from .ops.qsa_pre_indexer import qsa_pre_indexer


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    num_tokens, _, head_dim = tensor.shape
    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        shape = tensor.shape
        tensor, _ = triton_mrope(
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.is_neox_style,
        )
        return tensor.reshape(shape)

    rotated = rotary_emb.apply_rotary_emb.forward_cuda(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


def _supports_fused_pre_indexer(
    rotary_emb: nn.Module,
    head_dim: int,
    num_kv_heads: int,
    compress_ratio: int,
) -> bool:
    rotary_dim = int(rotary_emb.rotary_dim)
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    return (
        bool(getattr(rotary_emb, "is_neox_style", False))
        and (
            not mrope_section
            or (
                len(mrope_section) == 3
                and sum(mrope_section) == rotary_dim // 2
                and bool(getattr(rotary_emb, "mrope_interleaved", False))
            )
        )
        and head_dim == 128
        and rotary_dim == 64
        and num_kv_heads == 1
        and compress_ratio > 1
        and compress_ratio & (compress_ratio - 1) == 0
    )


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.use_fused_pre_indexer = _supports_fused_pre_indexer(
            rotary_emb,
            self.index_head_dim,
            self.index_kv_heads,
            self.compress_ratio,
        )
        self.prefix = prefix
        # MTP step 0 selects the target-aligned rows; later steps reuse them
        # while continuing to update the QSA side cache.
        self.skip_topk = False

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        raw_split = (
            raw.num_decodes,
            raw.num_decode_tokens,
            raw.num_prefills,
            raw.num_prefill_tokens,
            raw.decode_query_len,
        )
        compressed_split = (
            compressed.num_decodes,
            compressed.num_decode_tokens,
            compressed.num_prefills,
            compressed.num_prefill_tokens,
            compressed.decode_query_len,
        )
        if raw_split != compressed_split:
            raise RuntimeError("QSA side-cache metadata batch splits disagree")
        if not raw.logical_positions.is_cuda and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        from .ops.qsa import qsa_compress_groups_with_ratio, qsa_store_cache_rows
        from .ops.qsa_indexer import (
            expand_qsa_block_indices,
            qsa_select_paged_decode,
            qsa_select_paged_prefill,
        )

        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]

        # Q/K projection
        projected_qk, _ = self.index_qk_proj(hidden_states)
        projected_q, raw_keys = projected_qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        raw_key_state_cache = self.raw_key_cache
        compressed_key_cache = self.compressed_key_cache.kv_cache

        if self.use_fused_pre_indexer:
            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
            )
            qsa_pre_indexer(
                projected_q,
                raw_keys,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                self.q_layernorm.variance_epsilon,
                q,
                raw_key_state_cache.kv_cache,
                raw_metadata.slot_mapping,
                raw_metadata.block_table,
                raw_metadata.query_start_loc,
                raw_metadata.logical_positions,
                compressed_key_cache,
                compressed_metadata.slot_mapping,
                compressed_metadata.k_work_metadata,
                compress_ratio=self.compress_ratio,
                mrope_section=getattr(self.rotary_emb, "mrope_section", None),
                rope_pos_offset=(
                    raw_key_state_cache.rope_position_offset
                    if raw_key_state_cache.rope_position_cache is not None
                    else None
                ),
            )
        else:
            # Unfused reference path
            from flashinfer.norm import gemma_rmsnorm

            q = projected_q.reshape(-1, self.index_n_heads, self.index_head_dim)
            q = gemma_rmsnorm(
                q.reshape(-1, self.index_head_dim),
                self.q_layernorm.weight,
                self.q_layernorm.variance_epsilon,
            ).reshape_as(q)
            q = apply_qsa_rope(self.rotary_emb, positions, q)

            raw_key_cache = raw_key_state_cache.key_cache
            rope_position_cache = raw_key_state_cache.rope_position_cache
            if rope_position_cache is None:
                position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(
                    -1, 1, 3
                )
            else:
                position_rows = canonical_qsa_rope_positions(positions).to(
                    device=raw_key_cache.device
                )
            pooled, first_positions = qsa_compress_groups_with_ratio(
                raw_keys.reshape(-1, 1, self.index_head_dim),
                position_rows,
                raw_key_cache,
                raw_metadata.block_table,
                raw_metadata.token_to_req,
                raw_metadata.query_start_loc,
                raw_metadata.logical_positions,
                compressed_metadata.slot_mapping,
                self.compress_ratio,
                rope_position_cache,
            )
            compressed_keys = gemma_rmsnorm(
                pooled.reshape(-1, self.index_head_dim),
                self.k_layernorm.weight,
                self.k_layernorm.variance_epsilon,
            ).reshape(-1, 1, self.index_head_dim)
            if getattr(self.rotary_emb, "mrope_section", None):
                first_positions = first_positions.transpose(0, 1)
            else:
                first_positions = first_positions[:, 0]
            compressed_keys = apply_qsa_rope(
                self.rotary_emb,
                first_positions,
                compressed_keys,
            )
            qsa_store_cache_rows(
                compressed_key_cache,
                compressed_metadata.slot_mapping,
                compressed_keys,
            )
            qsa_store_cache_rows(
                raw_key_cache,
                raw_metadata.slot_mapping,
                raw_keys,
            )
            if rope_position_cache is not None:
                qsa_store_cache_rows(
                    rope_position_cache,
                    raw_metadata.slot_mapping,
                    position_rows,
                )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out

        if out is None:
            out = torch.empty(
                num_tokens,
                self.output_width,
                dtype=torch.int32,
                device=q.device,
            )
        elif out.shape != (num_tokens, self.output_width):
            raise ValueError("QSA selection output has an invalid shape")

        num_decode_tokens = compressed_metadata.num_decode_tokens
        decode_query_len = compressed_metadata.decode_query_len
        visible_blocks = compressed_metadata.visible_blocks[:num_tokens]
        block_indices = torch.empty(
            num_tokens,
            self.token_topk // self.compress_ratio,
            dtype=torch.int32,
            device=q.device,
        )

        # Decode requests occupy the leading rows and share one query length.
        if num_decode_tokens:
            num_decodes = compressed_metadata.num_decodes
            if num_decodes * decode_query_len != num_decode_tokens:
                raise ValueError("QSA decode rows must form a uniform request batch")
            decode_slice = slice(0, num_decode_tokens)
            qsa_select_paged_decode(
                q[decode_slice],
                compressed_key_cache,
                compressed_metadata.block_table[:num_decodes],
                visible_blocks[decode_slice],
                self.token_topk,
                self.compress_ratio,
                decode_query_len,
                block_indices[decode_slice],
            )

        # Prefill requests follow the leading decode rows in the reordered batch.
        if num_decode_tokens < num_tokens:
            num_decodes = compressed_metadata.num_decodes
            prefill_slice = slice(num_decode_tokens, num_tokens)
            qsa_select_paged_prefill(
                q[prefill_slice],
                compressed_key_cache,
                compressed_metadata.block_table[num_decodes:],
                compressed_metadata.query_start_loc[num_decodes:],
                visible_blocks[prefill_slice],
                self.token_topk,
                self.compress_ratio,
                compressed_metadata.max_query_len,
                block_indices[prefill_slice],
            )
        return expand_qsa_block_indices(
            block_indices,
            compressed_metadata.logical_positions[:num_tokens],
            visible_blocks,
            self.compress_ratio,
            self.token_topk,
            out,
        )


__all__ = ["QSAIndexer", "apply_qsa_rope"]
