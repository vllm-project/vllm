# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic decoder layer for DeepSeek V3.2 on SM100 (Blackwell).
Direct kernel calls, no module wrappers for norms.
Gate weight inlined, FusedMoE kept for quantized expert kernels.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

from .attention import MonolithicMLAAttention
from .ops import (
    fused_add_rms_norm,
    fused_norm_rope,
    fused_q,
    rms_norm,
)


class MonolithicDecoderLayer(nn.Module):
    """
    Single decoder layer: norm -> attn -> norm -> MoE/MLP.
    Norms are raw weight + direct kernel call.
    Gate inlined as raw weight, experts kept as FusedMoE for quantization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        # LayerNorm weights (raw)
        dtype = torch.get_default_dtype()
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )

        # Fused QKV A-projection lives inside self_attn namespace
        # for weight loading compatibility with original checkpoint paths
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear,
        )

        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA Attention
        self.attn = MonolithicMLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )

        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input norm + residual
        if residual is None:
            residual = hidden_states
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm_weight, self.rms_norm_eps
            )
        else:
            hidden_states, residual = fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm_weight,
                self.rms_norm_eps,
            )

        # Step 1. hidden_states -> q_c, kv_c, k_pe
        #                          index_k
        #                          index_weights
        out: torch.Tensor | tuple = self.self_attn.fused_qkv_a_proj(hidden_states)
        if isinstance(out, tuple):
            out: torch.Tensor = out[0]
        q_c, kv_c, k_pe = out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        index_k, _ = self.attn.indexer_wk(hidden_states)
        index_weights, _ = self.attn.indexer_weights_proj(hidden_states)

        # Step 2. Q RMS norm + KV RMS norm + KV RoPE + Index K layer norm + RoPE
        q_c, kv_c = fused_norm_rope(
            positions,
            # Q RMS norm
            q_c,
            self.attn.q_a_layernorm_weight,
            self.rms_norm_eps,
            # KV RMS norm
            kv_c,
            self.attn.kv_a_layernorm_weight,
            self.attn.rms_norm_eps,
            # KV RoPE
            k_pe,
            self.attn.rotary_emb.cos_sin_cache,
            # Index K layer norm + RoPE
            index_k,
            self.attn.indexer_k_norm.weight,
            self.attn.indexer_k_norm.bias,
            self.attn.rms_norm_eps,
            self.attn.indexer_rope_emb.cos_sin_cache,
        )

        # Step 3. q_c -> q
        #                index_q
        q = self.attn.q_b_proj(q_c)[0].view(
            -1, self.attn.num_local_heads, self.attn.qk_head_dim
        )
        index_q, _ = self.attn.indexer_wq_b(q_c)
        index_q = index_q.view(-1, self.attn.index_n_heads, self.attn.index_head_dim)

        # Step 4. Q RoPE + Index Q RoPE + Quantize + Index weights
        fused_q(
            positions,
            # Q RoPE
            q,
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.qk_nope_head_dim,
            # Index Q RoPE
            index_q,
            self.attn.indexer_rope_emb.cos_sin_cache,
        )

        index_q_fp8, index_q_scale = per_token_group_quant_fp8(
            index_q.view(-1, self.attn.index_head_dim),
            self.attn.indexer_quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        index_q_fp8 = index_q_fp8.view(
            -1, self.attn.index_n_heads, self.attn.index_head_dim
        )
        index_q_scale = index_q_scale.view(-1, self.attn.index_n_heads, 1)

        index_weights = (
            index_weights.unsqueeze(-1)
            * index_q_scale
            * self.attn.indexer_softmax_scale
            * self.attn.index_n_heads**-0.5
        ).squeeze(-1)

        self.attn.indexer_op(hidden_states, index_q_fp8, index_k, index_weights)

        # 4-7. KV cache update + W_UK_T absorption + sparse attn + W_UV
        attn_out = self.attn.mla_attn(
            q,
            kv_c,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.attn.num_local_heads * self.attn.v_head_dim,
            ),
        )

        # 8. Output projection (TP all-reduce)
        hidden_states, _ = self.attn.o_proj(attn_out)

        # Post-attn norm + residual
        hidden_states, residual = fused_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
        )

        # MLP / MoE
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
