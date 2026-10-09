# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's indexer on aiter's paged MXFP4 cache, for ROCm gfx950.

The dense path of the ROCm paged MXFP4 indexer: K is written in the order
aiter's MQA-logits kernel reads and scored in place, prefill included. V4's
own MXFP4 Q/K Triton kernels pack FP4 with CUDA PTX, so ROCm quantizes both
with aiter's ops instead.
"""

import functools
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (
    RocmSparseAttnIndexer,
)
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    DeepseekV4RocmMxfp4IndexerBackend,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_mxfp4_indexer_k_store,
    rocm_mxfp4_indexer_q_quant,
)


@triton.jit
def _compress_pool_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    kv_slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    out_ptr,
    out_stride,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    """The softmax-gated pooling of `_fused_kv_compress_norm_rope_insert_*`,
    stopping before the norm: one bf16 row per boundary token."""
    token_idx = tl.program_id(0)
    # Pool exactly the rows aiter's writer stores; it skips the same tokens.
    if tl.load(kv_slot_mapping_ptr + token_idx) < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=mask_pos,
        other=0,
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + (pos % block_size) * state_cache_stride1
        + head_offset
    )

    block = tl.arange(0, HEAD_SIZE)
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=mask_pos[:, None],
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + block[None, :], mask=mask_pos[:, None], other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)
    tl.store(
        out_ptr + token_idx.to(tl.int64) * out_stride + block,
        compressed_kv.to(tl.bfloat16),
    )


@functools.cache
def _pooled_k_scratch(num_tokens: int, head_dim: int, device: str) -> torch.Tensor:
    # One buffer for all indexer layers: they run one after another and each
    # consumes its rows before the next layer writes.
    return torch.empty(num_tokens, head_dim, dtype=torch.bfloat16, device=device)


class DeepseekV4RocmMxfp4Compressor(DeepseekCompressor):
    """The indexer's compressor writing aiter's preshuffled MXFP4 pages: a
    pooling kernel, then aiter's norm + RoPE + MXFP4 cache op, which owns the
    page layout its MQA-logits kernel reads."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        super().__init__(vllm_config, **kwargs)
        assert self.head_dim == 128 and self.use_fp4_cache
        self.num_index_heads = vllm_config.model_config.hf_config.index_n_heads
        self._pooled_k = _pooled_k_scratch(
            self.max_num_batched_tokens, self.head_dim, self.device
        )

    def _register_store_warmup(self, vllm_config: VllmConfig) -> None:
        # The fused store kernel packs FP4 with CUDA PTX and never runs here.
        pass

    def _compress_norm_rope_store(
        self,
        state_metadata: CompressorMetadata,
        state_cache: torch.Tensor,
        state_width: int,
        positions: torch.Tensor,
        rotary_emb,
        attn_metadata: dict[str, Any],
        pdl_kwargs: dict,
    ) -> None:
        num_tokens = state_metadata.slot_mapping.shape[0]
        if num_tokens == 0:
            return
        assert num_tokens <= self._pooled_k.shape[0]
        k_cache_metadata = attn_metadata[self.k_cache_prefix]
        kv_slot_mapping = k_cache_metadata.slot_mapping[:num_tokens]
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        pooled_k = self._pooled_k[:num_tokens]
        _compress_pool_kernel[(num_tokens,)](
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            state_metadata.token_to_req_indices,
            positions,
            kv_slot_mapping,
            state_metadata.block_table,
            state_metadata.block_table.stride(0),
            state_metadata.block_size,
            pooled_k,
            pooled_k.stride(0),
            HEAD_SIZE=self.head_dim,
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            num_warps=1,
        )
        rocm_mxfp4_indexer_k_store(
            pooled_k,
            positions[:num_tokens],
            rotary_emb.cos_sin_cache,
            self.norm.weight,
            self.rms_norm_eps,
            kv_cache,
            kv_slot_mapping,
            self.compress_ratio,
            self.use_fp4_cache,
            num_heads=self.num_index_heads,
        )


class DeepseekV4RocmMxfp4IndexerCache(DeepseekV4IndexerCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4RocmMxfp4IndexerBackend


class DeepseekV4RocmMxfp4Indexer(DeepseekV4Indexer):
    """`DeepseekV4Indexer` on aiter's paged MXFP4 cache, scoring the whole
    context in place (V4 has no candidate pool)."""

    cache_cls = DeepseekV4RocmMxfp4IndexerCache
    compressor_cls = DeepseekV4RocmMxfp4Compressor
    attn_cls = RocmSparseAttnIndexer

    def _register_q_quant_warmup(self) -> None:
        # The shared MXFP4 Q kernel packs FP4 with CUDA PTX; aiter's op runs here.
        pass

    def _q_rope_quant(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        rotary_emb: nn.Module,
        indexer_weights: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # aiter's MQA-logits kernel takes fp32 weights with both scales folded.
        return rocm_mxfp4_indexer_q_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
        )
