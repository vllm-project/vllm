# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic MLA attention for DeepSeek V3.2 on SM100 (Blackwell).

MLA forward fully inlined:
  KV cache update -> W_UK_T absorption -> sparse attn kernel -> W_UV up-proj
MLAAttention kept only as a registration stub for KV cache / backend.
"""

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size


class MonolithicMLAAttention(nn.Module):
    """
    Monolithic MLA attention for DeepSeek V3.2 targeting SM100.
    MLA forward fully inlined. MLAAttention kept only for KV cache
    registration and backend/impl initialization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_local_heads = num_heads // get_tensor_model_parallel_world_size()
        self.scaling = self.qk_head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # Q path
        self.q_a_layernorm_weight = nn.Parameter(
            torch.ones(q_lora_rank, dtype=torch.get_default_dtype())
        )
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        # KV path
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=torch.get_default_dtype())
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection (TP sync point)
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # V3.2 Sparse Indexer (inlined)
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )
        self.topk_tokens = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.indexer_softmax_scale = config.index_head_dim**-0.5
        self.indexer_quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer_wq_b = ReplicatedLinear(
            q_lora_rank,
            config.index_head_dim * config.index_n_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wq_b",
        )
        self.indexer_wk = ReplicatedLinear(
            hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wk",
        )
        self.indexer_k_norm = LayerNorm(config.index_head_dim, eps=1e-6)
        self.indexer_weights_proj = ReplicatedLinear(
            hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.indexer.weights_proj",
        )

        idx_dim = config.index_head_dim
        indexer_cache_head_dim = idx_dim + idx_dim // 128 * 4
        self.indexer_k_cache = DeepseekV32IndexerCache(
            head_dim=indexer_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.indexer.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.indexer_k_cache,
            self.indexer_quant_block_size,
            "ue8m0",
            self.topk_tokens,
            config.index_head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            self.topk_indices_buffer,
        )

        # MLAAttention stub: only for KV cache registration + backend init.
        # We never call its forward(); we inline everything below.
        class _IndexerProxy:
            def __init__(proxy_self):
                proxy_self.topk_indices_buffer = topk_indices_buffer
                proxy_self.indexer_op = self.indexer_op

        self._indexer_proxy = _IndexerProxy()
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mla_attn",
            use_sparse=True,
            indexer=self._indexer_proxy,
        )

    def _run_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """V3.2 sparse indexer: compute top-K token indices."""
        q, _ = self.indexer_wq_b(q_c)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = q.split(
            [
                self.qk_rope_head_dim,
                self.index_head_dim - self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        k, _ = self.indexer_wk(hidden_states)
        k = self.indexer_k_norm(k)
        k_pe, k_nope = k.split(
            [
                self.qk_rope_head_dim,
                self.index_head_dim - self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        q_pe, k_pe = self.indexer_rope_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.index_n_heads, self.qk_rope_head_dim)
        k_pe = k_pe.reshape(-1, 1, self.qk_rope_head_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        q_fp8, q_scale = per_token_group_quant_fp8(
            q.view(-1, self.index_head_dim),
            self.indexer_quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
        q_scale = q_scale.view(-1, self.index_n_heads, 1)

        weights, _ = self.indexer_weights_proj(hidden_states)
        weights = (
            weights.unsqueeze(-1)
            * q_scale
            * self.indexer_softmax_scale
            * self.index_n_heads**-0.5
        ).squeeze(-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        kv_lora: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fully inlined MLA forward. q_c is already layernormed.

        Steps:
          1. Q B-projection + split q_nope/q_pe
          2. KV layernorm + RoPE
          3. Sparse indexer (fills topk_indices_buffer)
          4. KV cache update (concat_and_cache_mla)
          5. W_UK_T absorption: q_nope @ W_UK_T -> q_latent
          6. Sparse attention kernel (impl.forward_mqa)
          7. W_UV up-projection: attn_out @ W_UV -> v
          8. Output projection (o_proj, TP all-reduce)
        """
        mla = self.mla_attn

        # 1. Q B-projection
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        # 2. KV layernorm + RoPE
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = torch.empty_like(kv_c)
        ops.rms_norm(
            kv_c_normed,
            kv_c,
            self.kv_a_layernorm_weight,
            self.rms_norm_eps,
        )
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        # 3. Sparse indexer
        self._run_indexer(hidden_states, q_c, positions)

        # 4-7. KV cache update + W_UK_T absorption + sparse attn + W_UV
        attn_out = mla(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.num_local_heads * self.v_head_dim,
            ),
        )

        # 8. Output projection (TP all-reduce)
        result, _ = self.o_proj(attn_out)
        return result
