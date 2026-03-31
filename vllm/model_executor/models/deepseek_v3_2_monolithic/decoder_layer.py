# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic decoder layer for DeepSeek V3.2 on SM100 (Blackwell).
Direct kernel calls, no module wrappers for norms.
Gate weight inlined, FusedMoE kept for quantized expert kernels.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from .allreduce_rms import AllReduceRMSParams, allreduce_add_rms_norm
from .attention import MonolithicMLAAttention
from .ops import concat_quant_fp8, fused_norm_rope, fused_q, rms_norm
from .sparse_indexer import sparse_attn_indexer


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
        fi_params: AllReduceRMSParams | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()
        self._fi_params = fi_params

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.indexer_workspace_size = get_max_prefill_buffer_size(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len

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

        # MLA Attention — disable AllReduce in o_proj when using fused path
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
        self.attn.o_proj.reduce_results = False

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
            self.mlp.skip_final_allreduce = True
            self.mlp.skip_scale_and_add = True
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input norm + residual
        # When fused_allreduce_rms is enabled, hidden_states arriving from
        # the previous layer is the *unreduced* MLP/MoE output. We fuse
        # AllReduce + residual-add + RMSNorm into a single kernel.
        if residual is None:
            # First layer: hidden_states is from embed_tokens (already
            # fully materialised), no allreduce needed.
            residual = hidden_states
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm_weight, self.rms_norm_eps
            )
        else:
            hidden_states, residual = allreduce_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm_weight,
                self.rms_norm_eps,
                self._fi_params,
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

        # Fetch slot_mapping early so fused_norm_rope can write FP8 data
        # directly into the indexer KV cache (saves a separate kernel).
        from vllm.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        if isinstance(attn_metadata, dict):
            idx_meta = attn_metadata[self.attn.indexer_k_cache.prefix]
            # Indexer and MLA caches share the same block_size and track
            # the same requests, so their slot_mappings are identical.
            slot_mapping = idx_meta.slot_mapping
        else:
            slot_mapping = None
        if slot_mapping is not None:
            indexer_k_cache = self.attn.indexer_k_cache.kv_cache
            mla_kv_cache = self.attn.mla_attn.kv_cache
            mla_k_scale = self.attn.mla_attn._k_scale
        else:
            indexer_k_cache = None
            mla_kv_cache = None
            mla_k_scale = None

        # Step 2. Q RMS norm
        #         + KV RMS norm + KV RoPE + MLA cache write
        #         + Index K layer norm + RoPE + FP8 quant + cache write
        #         + Init topk indices
        q_c = fused_norm_rope(
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
            # Top k indices
            self.attn.topk_indices_buffer,
            # Fused cache writes (single slot_mapping for both caches)
            slot_mapping=slot_mapping,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.attn.mla_attn.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
        )

        # Step 3. q_c -> q
        #                index_q
        q = self.attn.q_b_proj(q_c)[0].view(
            -1, self.attn.num_local_heads, self.attn.qk_head_dim
        )
        index_q, _ = self.attn.indexer_wq_b(q_c)
        index_q = index_q.view(-1, self.attn.index_n_heads, self.attn.index_head_dim)

        # Step 4. Q RoPE + Index Q RoPE + Quantize + Index weights
        index_q_fp8, index_weights = fused_q(
            positions,
            # Q RoPE
            q,
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.qk_nope_head_dim,
            # Index Q RoPE
            index_q,
            self.attn.indexer_rope_emb.cos_sin_cache,
            # Index weights
            index_weights,
            self.attn.indexer_softmax_scale,
            self.attn.index_n_heads**-0.5,
        )

        # Step 5. Sparse indexer.
        # The FP8 quant + cache write for index_k is already done in
        # fused_norm_rope (step 2) when slot_mapping is available.
        sparse_attn_indexer(
            self.attn.indexer_k_cache.prefix,
            self.attn.indexer_k_cache.kv_cache,
            index_q_fp8,
            index_weights,
            self.attn.topk_tokens,
            self.attn.index_head_dim,
            self.max_model_len,
            self.indexer_workspace_size,
            self.attn.topk_indices_buffer,
        )

        # Step 6. MLA sparse decode attention (inlined).
        # The KV cache update was already done in fused_norm_rope (step 2).
        attn_out = self._mla_sparse_decode(q, slot_mapping, hidden_states.shape[0])

        # Step 7. Output projection (AllReduce disabled when fused).
        hidden_states, _ = self.attn.o_proj(attn_out)

        # Post-attn norm + residual
        # Fuse the o_proj AllReduce with post-attention RMSNorm.
        hidden_states, residual = allreduce_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
            self._fi_params,
        )

        # MLP / MoE
        # When fused_allreduce_rms is enabled, the MLP/MoE AllReduce is
        # deferred — it will be fused with the next layer's input norm.
        if self.is_moe:
            # MoE returns raw (shared_output, routed_output) without
            # applying scale + add. torch.compile fuses the elementwise ops.
            shared_output, routed_output = self.mlp(hidden_states)
            hidden_states = scale_and_add(
                routed_output, self.routed_scaling_factor, shared_output
            )
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def _mla_sparse_decode(
        self,
        q: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        num_padded_tokens: int,
    ) -> torch.Tensor:
        mla = self.attn.mla_attn
        output_shape = (num_padded_tokens, mla.num_heads * mla.v_head_dim)

        from vllm.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[mla.layer_name]
        if attn_metadata is None or slot_mapping is None:
            return torch.zeros(output_shape, dtype=q.dtype, device=q.device)

        num_actual_toks = attn_metadata.num_actual_tokens
        q = q[:num_actual_toks]
        kv_cache = mla.kv_cache

        fp8_attention = mla.kv_cache_dtype.startswith("fp8")
        if fp8_attention and mla.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        impl = mla.impl

        # 1. Q absorption: q_nope @ W_UK^T → ql_nope
        q_nope, q_pe = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)  # (B, N, P) → (N, B, P)
        ql_nope = q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], mla.W_UK_T.shape[2]
        )
        torch.bmm(q_nope, mla.W_UK_T, out=ql_nope)
        ql_nope = ql_nope.transpose(0, 1)  # (N, B, L) → (B, N, L)

        # 2. FP8 query quantization (if needed)
        if fp8_attention and impl.supports_quant_query_input:
            mqa_q = concat_quant_fp8(ql_nope, q_pe, mla._q_scale)
        else:
            mqa_q = (ql_nope, q_pe)

        # 3. Forward MQA (topk conversion + FlashInfer kernel)
        attn_out, _ = impl.forward_mqa(mqa_q, kv_cache, attn_metadata, mla)

        # 4. V up-projection: attn_out @ W_UV → output
        #    (N, B, L) x (N, L, V) → (N, B, V) → (B, N*V)
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
        x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)
        out = output[:num_actual_toks].view(-1, mla.num_heads, mla.v_head_dim)
        out = out.transpose(0, 1)
        torch.bmm(x, mla.W_UV, out=out)
        return output


@torch.compile
def scale_and_add(x: torch.Tensor, scale: float, y: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * scale
    z = x + y.to(torch.float32)
    return z.to(orig_dtype)
