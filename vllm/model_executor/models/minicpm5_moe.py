# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM5 MoE adapted from the existing checkpoint implementation.

Reuses upstream vLLM attention, fused MoE, and quantized Linear kernels.
The checkpoint uses GQA with head-interleaved query/output gates, no QK
normalization, full RoPE, and sigmoid/noaux_tc routing in the model dtype,
matching the team SGLang implementation.
"""

import copy
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.utils import (
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
)


class MiniCPM5Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        c = vllm_config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        if c.num_attention_heads % tp:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if c.num_key_value_heads >= tp:
            if c.num_key_value_heads % tp:
                raise ValueError("num_key_value_heads must be divisible by TP size")
        elif tp % c.num_key_value_heads:
            raise ValueError("TP size must be divisible by num_key_value_heads")
        self.num_heads = c.num_attention_heads // tp
        self.num_kv_heads = max(1, c.num_key_value_heads // tp)
        self.head_dim = c.head_dim
        self.gated = c.use_gated_attention
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            c.hidden_size,
            self.head_dim,
            c.num_attention_heads * (2 if self.gated else 1),
            c.num_key_value_heads,
            bias=c.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            c.num_attention_heads * self.head_dim,
            c.hidden_size,
            bias=c.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        rope_parameters = getattr(c, "rope_parameters", None) or {
            "rope_type": "default",
            "rope_theta": c.rope_theta,
        }
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=c.max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size * (2 if self.gated else 1), self.kv_size, self.kv_size],
            dim=-1,
        )
        gate = None
        if self.gated:
            # HF reference: [token, head, 2 * head_dim], then split last axis.
            q, gate = q.reshape(-1, self.num_heads, 2 * self.head_dim).chunk(2, -1)
            q = q.reshape(-1, self.q_size)
            gate = gate.reshape(-1, self.q_size)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        if gate is not None:
            out = out * gate.sigmoid()
        return self.o_proj(out)[0]


class MiniCPM5DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        c = vllm_config.model_config.hf_config
        self.self_attn = MiniCPM5Attention(vllm_config, f"{prefix}.self_attn")
        layer = extract_layer_index(prefix)
        if layer < c.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(
                c.hidden_size,
                c.intermediate_size,
                c.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            # HF MiniCPM5 router hard-codes these semantics, while config's
            # inherited topk_method default is "greedy". Do not mutate config.
            moe_config = copy.copy(c)
            moe_config.topk_method = "noaux_tc"
            moe_config.scoring_func = "sigmoid"
            moe_config.moe_router_dtype = None
            self.mlp = DeepseekV2MoE(
                moe_config,
                vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class MiniCPM5Model(LlamaModel):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked: list[tuple[str, str, str | int]] = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped, self):
                        break
                    param = params[mapped]
                    param.weight_loader(
                        param, weight, mapped, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded.add(mapped)
                    break
                else:
                    raise KeyError(f"Unmapped expert tensor: {name}")
                continue
            for param_name, weight_name, dense_shard_id in stacked:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    break
                param = params[mapped]
                param.weight_loader(param, weight, dense_shard_id)
                loaded.add(mapped)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith(".bias") and name not in params:
                    continue
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        return loaded


class MiniCPM5MoEForCausalLM(LlamaForCausalLM, DeepseekV2MixtureOfExperts):
    # The custom loader maps expert projections separately from dense projections.
    hf_to_vllm_mapper = WeightsMapper()

    def _init_model(self, vllm_config, prefix="", layer_type=None):
        return MiniCPM5Model(
            vllm_config=vllm_config, prefix=prefix, layer_type=MiniCPM5DecoderLayer
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.num_expert_groups = self.config.n_group
        self.moe_mlp_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(getattr(layer, "mlp", None), DeepseekV2MoE)
        ]
        self.moe_layers = [mlp.experts for mlp in self.moe_mlp_layers]
        self.extract_moe_parameters(
            self.moe_mlp_layers[-1] if self.moe_mlp_layers else None
        )

    def get_expert_mapping(self):
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )
