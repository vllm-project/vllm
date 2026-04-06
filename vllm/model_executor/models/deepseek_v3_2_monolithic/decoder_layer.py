# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic decoder layer for DeepSeek V3.2 on SM100 (Blackwell).
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from .attention import MonolithicMLAAttention
from .ops import fused_norm_rope, fused_q
from .sparse_indexer import sparse_attn_indexer


def monolithic_attn(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    attn = layer.attn
    mla = attn.mla_attn

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        output.zero_()
        return output

    mla_attn_metadata = attn_metadata.get(mla.layer_name)
    if mla_attn_metadata is None:
        output.zero_()
        return output

    num_actual_toks = mla_attn_metadata.num_actual_tokens
    if num_actual_toks == 0:
        output.zero_()
        return output

    # Step 2. fused norm + rope + cache writes
    slot_mapping = None
    indexer_k_cache = None
    mla_kv_cache = None
    mla_k_scale = None
    idx_meta = attn_metadata.get(attn.indexer_k_cache.prefix)
    if idx_meta is not None:
        slot_mapping = idx_meta.slot_mapping
        indexer_k_cache = attn.indexer_k_cache.kv_cache
        mla_kv_cache = attn.mla_attn.kv_cache
        mla_k_scale = attn.mla_attn._k_scale

    q_c = fused_norm_rope(
        positions,
        q_c,
        attn.q_a_layernorm_weight,
        layer.rms_norm_eps,
        kv_c,
        attn.kv_a_layernorm_weight,
        attn.rms_norm_eps,
        k_pe,
        attn.rotary_emb.cos_sin_cache,
        index_k,
        attn.indexer_k_norm.weight,
        attn.indexer_k_norm.bias,
        attn.rms_norm_eps,
        attn.indexer_rope_emb.cos_sin_cache,
        attn.topk_indices_buffer,
        slot_mapping=slot_mapping,
        indexer_k_cache=indexer_k_cache,
        mla_kv_cache=mla_kv_cache,
        mla_kv_cache_dtype=attn.mla_attn.kv_cache_dtype,
        mla_k_scale=mla_k_scale,
    )

    # Step 3. q_c -> index_q, q
    step3_out = torch.mm(q_c, layer._fused_step3_q_w.T)
    index_q, q = step3_out.split(
        [layer._step3_index_q_dim, step3_out.shape[-1] - layer._step3_index_q_dim],
        dim=-1,
    )
    index_q = index_q.view(-1, attn.index_n_heads, attn.index_head_dim)
    q = q.view(-1, attn.num_local_heads, attn.qk_head_dim)

    # Step 4. Q RoPE + W_UK_T absorption + FP8 packing
    q_nope, q_pe = q.split(
        [mla.qk_nope_head_dim, mla.qk_rope_head_dim],
        dim=-1,
    )
    q_nope = q_nope.transpose(0, 1)
    ql_nope = torch.bmm(q_nope, mla.W_UK_T)
    ql_nope = ql_nope.transpose(0, 1)

    index_q_fp8, index_weights, mqa_q = fused_q(
        positions,
        q_pe,
        attn.rotary_emb.cos_sin_cache,
        index_q,
        attn.indexer_rope_emb.cos_sin_cache,
        ql_nope,
        mla._q_scale,
        index_weights,
        attn.indexer_softmax_scale,
        attn.index_n_heads**-0.5,
    )

    # Steps 5-6. Sparse indexer + MLA sparse decode attention
    sparse_attn_indexer(
        attn.indexer_k_cache.prefix,
        attn.indexer_k_cache.kv_cache,
        index_q_fp8,
        index_weights,
        attn.topk_tokens,
        attn.index_head_dim,
        layer.max_model_len,
        layer.indexer_workspace_size,
        attn.topk_indices_buffer,
    )

    mqa_q = mqa_q[:num_actual_toks]
    kv_cache = mla.kv_cache
    if mla.kv_cache_dtype.startswith("fp8") and mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    attn_out, _ = mla.impl.forward_mqa(mqa_q, kv_cache, mla_attn_metadata, mla)
    x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)

    out = output[:num_actual_toks].view(-1, mla.num_heads, mla.v_head_dim)
    out = out.transpose(0, 1)
    torch.bmm(x, mla.W_UV, out=out)
    return output


def monolithic_attn_fake(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, q_c, kv_c, k_pe, index_k, index_weights, layer_name
    return output


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=monolithic_attn,
    fake_impl=monolithic_attn_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
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
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.layer_name = prefix
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
        self.indexer_workspace_size = get_max_prefill_buffer_size(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len

        # Use the regular vLLM RMSNorm modules so the compiler sees the
        # canonical residual-add + RMSNorm pattern.
        dtype = torch.get_default_dtype()
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
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

    def fuse_indexer_weights(self) -> None:
        """Fuse Step 1 and Step 3 BF16 linears used by the monolithic path.

        Call after model weights are loaded.
        """
        attn = self.attn
        qkv_a = self.self_attn.fused_qkv_a_proj.weight.data  # [2112, 7168]
        wk = attn.indexer_wk.weight.data  # [128, 7168]
        wp = attn.indexer_weights_proj.weight.data  # [64, 7168]
        if not (qkv_a.dtype == wk.dtype == wp.dtype):
            raise ValueError(
                "Cannot fuse Step 1 weights: expected matching dtypes for "
                "fused_qkv_a_proj, indexer_wk, and indexer_weights_proj."
            )
        self._fused_step1_hidden_w = nn.Parameter(
            torch.cat([qkv_a, wk, wp], dim=0),  # [2304, 7168]
            requires_grad=False,
        )
        self._step1_split_sizes = [
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            wk.shape[0],
            wp.shape[0],
        ]

        wq_b = attn.indexer_wq_b.weight.data
        q_b = attn.q_b_proj.weight.data
        if wq_b.dtype != q_b.dtype:
            raise ValueError(
                "Cannot fuse Step 3 weights: expected matching dtypes for "
                "indexer_wq_b and q_b_proj."
            )
        self._fused_step3_q_w = nn.Parameter(
            torch.cat([wq_b, q_b], dim=0),
            requires_grad=False,
        )
        self._step3_index_q_dim = wq_b.shape[0]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Step 1. hidden_states -> q_c, kv_c, k_pe, index_k, index_weights
        step1_out = torch.mm(hidden_states, self._fused_step1_hidden_w.T)
        q_c, kv_c, k_pe, index_k, index_weights = step1_out.split(
            self._step1_split_sizes,
            dim=-1,
        )

        # Steps 2-6. Combined: fused norm/rope + Q projections + sparse MLA.
        mla = self.attn.mla_attn
        output_shape = (hidden_states.shape[0], mla.num_heads * mla.v_head_dim)
        output_dtype = mla.W_UV.dtype
        attn_out = torch.empty(
            output_shape,
            dtype=output_dtype,
            device=hidden_states.device,
        )
        attn_out = torch.ops.vllm.monolithic_attn(
            positions,
            q_c,
            kv_c,
            k_pe,
            index_k,
            index_weights,
            attn_out,
            self.layer_name,
        )

        hidden_states, _ = self.attn.o_proj(attn_out)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
