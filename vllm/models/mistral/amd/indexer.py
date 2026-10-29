# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA lightning indexer and its FP8 key cache."""

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    fused_indexer_q_rope_quant,
)
from vllm.models.deepseek_v32.common.kernels import _INDEXER_CACHE_BLOCK_TILE
from vllm.models.mistral.amd.ops import fusion_enabled
from vllm.models.mistral.amd.ops.indexer import (
    fused_indexer_k_norm_rope,
    fused_indexer_k_norm_rope_cache_write,
    fused_indexer_q_ssmax_rope_quant,
)
from vllm.models.mistral.amd.ops.layernorm import layer_norm
from vllm.models.mistral.amd.ssmax import SSMaxScaling
from vllm.models.mistral.common.layer_spec import MistralLayerSpec
from vllm.models.mistral.common.rope import MistralRotaryEmbedding
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_cache_is_shuffled
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SparseCacheRole


class MistralIndexerCache(nn.Module, AttentionLayerBase):
    """FP8 key cache of the DSA indexer (fp8 values + fp32 per-block scales)."""

    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C] as expected by the indexer kernels.
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        # The ROCm insert, gather and paged-logits readers tile a block of more
        # than one token as 16x16 SHUFFLE (indexer_k_cache_is_shuffled); a
        # partial tile would overwrite the block's scales.
        if (
            block_size is not None
            and indexer_k_cache_is_shuffled(block_size)
            and block_size % _INDEXER_CACHE_BLOCK_TILE != 0
        ):
            raise ValueError(
                f"{self.prefix}: indexer K cache block size {block_size} must be 1 "
                f"or a multiple of {_INDEXER_CACHE_BLOCK_TILE} on ROCm"
            )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
        )

    def forward(self): ...

    @property
    def uses_shuffled_layout(self) -> bool:
        """Whether the bound cache's readers expect the SHUFFLE value layout;
        fused writers take the layout from here."""
        return self.kv_cache.ndim == 3 and indexer_k_cache_is_shuffled(
            self.kv_cache.shape[1]
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV32IndexerBackend


class MistralIndexer(nn.Module):
    """DSA lightning indexer; writes the top-k key indices of each query token
    into the model-wide ``topk_indices_buffer``."""

    quant_block_size = 128

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        spec: MistralLayerSpec,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        eps = config.indexer_norm_eps

        self.q_input_norm = LayerNorm(hidden_size, eps=eps)
        self.wq_b = ReplicatedLinear(
            hidden_size,
            self.n_head * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=eps, dtype=torch.float32)
        self.ssmax = (
            SSMaxScaling(self.n_head, tp_sharded=False)
            if config.indexer_ssmax_scaling
            else None
        )
        self.fuse_q = self.ssmax is not None and fusion_enabled("indexer_q")
        self.fuse_k = fusion_enabled("indexer_k")
        self.fuse_q_norm = fusion_enabled("indexer_q_norm")
        # K goes straight to the FP8 cache and the Q launch clears the top-k
        # rows, so the indexer op skips both. Under PCP/DP it gathers K first.
        parallel_config = vllm_config.parallel_config
        self.fuse_k_cache = (
            self.fuse_q
            and self.fuse_k
            and fusion_enabled("indexer_k_cache")
            and parallel_config.prefill_context_parallel_size == 1
            and parallel_config.data_parallel_size == 1
        )
        # Folded into the per-head weights: softmax scale, head averaging.
        self.weights_scale = self.head_dim**-0.5 * self.n_head**-0.5
        # Read by the sparse MLA impl (resolved per read, see SparseMLACommonImpl).
        self.topk_indices_buffer = topk_indices_buffer

        self.k_cache = MistralIndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=vllm_config.cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            "ue8m0",
            config.index_topk,
            self.head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
            skip_k_cache_insert=self.fuse_k_cache,
            skip_topk_buffer_clear=self.fuse_k_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding | None,
    ) -> torch.Tensor:
        if self.fuse_q_norm:
            q_in = layer_norm(
                hidden_states,
                self.q_input_norm.weight,
                self.q_input_norm.bias,
                self.q_input_norm.eps,
            )
        else:
            q_in = self.q_input_norm(hidden_states)
        q = self.wq_b(q_in)[0]
        q = q.view(-1, self.n_head, self.head_dim)
        kw = self.wk_weights_proj(hidden_states)[0]
        weights = kw[:, self.head_dim :]
        # MistralRotaryEmbedding is interleaved, hence is_neox=False below.
        cos_sin_cache = None if rotary_emb is None else rotary_emb.cos_sin_cache

        k: torch.Tensor | None
        if self.fuse_k_cache:
            k = None
            slot_mapping = get_forward_context().slot_mapping
            assert isinstance(slot_mapping, dict)
            # Empty during profiling, before the caches are bound.
            if self.k_cache.prefix in slot_mapping:
                fused_indexer_k_norm_rope_cache_write(
                    positions,
                    kw[:, : self.head_dim],
                    self.k_norm.weight,
                    self.k_norm.bias,
                    self.k_norm.eps,
                    cos_sin_cache,
                    is_neox=False,
                    kv_cache=self.k_cache.kv_cache,
                    slot_mapping=slot_mapping[self.k_cache.prefix],
                    shuffled=self.k_cache.uses_shuffled_layout,
                )
        elif self.fuse_k:
            k = fused_indexer_k_norm_rope(
                positions,
                kw[:, : self.head_dim],
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.eps,
                cos_sin_cache,
                is_neox=False,
            )
        else:
            k = self.k_norm(kw[:, : self.head_dim])
            if rotary_emb is not None:
                k_pe = k[:, None, : self.rope_dim]
                _, k_pe = rotary_emb(positions, torch.empty_like(k_pe), k_pe)
                k = torch.cat(
                    [k_pe.reshape(-1, self.rope_dim), k[:, self.rope_dim :]], -1
                )

        if self.fuse_q:
            assert self.ssmax is not None
            # Same scale factors (and fp32 rounding) as the unfused paths.
            softmax_scale, head_scale = (
                (self.weights_scale, 1.0)
                if rotary_emb is None
                else (self.head_dim**-0.5, self.n_head**-0.5)
            )
            q_fp8, weights = fused_indexer_q_ssmax_rope_quant(
                positions,
                q,
                cos_sin_cache,
                weights,
                log_positions,
                self.ssmax.s,
                self.ssmax.b,
                softmax_scale,
                head_scale,
                is_neox=False,
                topk_indices=self.topk_indices_buffer if self.fuse_k_cache else None,
            )
            return self.indexer_op(hidden_states, q_fp8, k, weights)

        if self.ssmax is not None:
            q = self.ssmax(q, log_positions)
        if rotary_emb is None:
            q_fp8, q_scale = per_token_group_quant_fp8(
                q.reshape(-1, self.head_dim),
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=True,
            )
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            weights = weights * q_scale.view(-1, self.n_head) * self.weights_scale
        else:
            # Indexer layout is [rope | nope], unlike MLA's [nope | rope].
            q_fp8, weights = fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                weights,
                self.head_dim**-0.5,
                self.n_head**-0.5,
                rotary_emb.is_neox_style,
            )
        return self.indexer_op(hidden_states, q_fp8, k, weights)
