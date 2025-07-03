# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from transformers import Llama4TextConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from .utils import (AutoWeightsLoader, extract_layer_index, fast_topk,
                    is_pp_missing_parameter)


class Llama4MoE(nn.Module):

    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
        # psuedo-standard is that the router scores are floats
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))

    def __init__(self,
                 config: Llama4TextConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(config.hidden_size,
                                       config.num_local_experts,
                                       bias=False,
                                       quant_config=None,
                                       prefix=f"{prefix}.router")

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            apply_router_weight_on_input=True,
            reduce_results=False,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts")

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.shared_expert",
            reduce_results=self.experts.must_reduce_shared_expert_outputs(),
        )

    def forward(self, hidden_states):
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        experts_out = routed_out + shared_out

        if self.tp_size > 1:
            experts_out = self.experts.maybe_all_reduce_tensor_model_parallel(
                experts_out)

        return experts_out


class Llama4Attention(nn.Module):

    def __init__(self,
                 config: Llama4TextConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_idx] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        # TODO: attn_temperature_tuning should be a bool in huggingface
        self.attn_temperature_tuning = self.nope and \
            config.attn_temperature_tuning > 0

        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            has_weight=False,
            dtype=torch.float32,
        ) if self.use_qk_norm else None
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling if rope_scaling != "default" else None,
            is_neox_style=is_neox_style,
        ) if not self.nope else None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=None,
            use_irope=not self.nope,
            prefix=f"{prefix}.attn",
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0

        return attn_scale.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.qk_norm is not None:
            q = q.reshape(-1, self.num_heads, self.head_dim)
            q = self.qk_norm(q.float()).reshape(-1, self.q_size).to(q.dtype)
            k = k.reshape(-1, self.num_kv_heads, self.head_dim)
            k = self.qk_norm(k.float()).reshape(-1, self.kv_size).to(k.dtype)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399)
        # to NoPE layers, where the inference-time temperature tuning function
        # is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        #
        # We should apply temperature tuning between (after) rotary / QK norm
        # and (before) attention.
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        is_moe_layer = config.interleave_moe_layer_step > 0 and (
            self.layer_idx + 1) % config.interleave_moe_layer_step == 0
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.feed_forward",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Llama4Model(LlamaModel):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer):
        self.num_experts = vllm_config.model_config.hf_config.num_local_experts
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=layer_type)

    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        loaded_params: set[str],
        expert_params_mapping: list[tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool:
        expert_param_loaded = False
        if "experts.gate_up_proj" in name:
            loaded_weight = loaded_weight.chunk(2, dim=-1)
        for (param_name, weight_name, expert_id,
             shard_id) in expert_params_mapping:
            new_loaded_weight = loaded_weight
            if fused:
                e_str, _, proj_str, _ = weight_name.split('.')
                weight_name = f"{e_str}.{proj_str}"
                param_name = f"{param_name}weight"
            if weight_name not in name:
                continue
            full_param_name = name.replace(weight_name, param_name)
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue
            if ((name.endswith(".bias") or name.endswith("_bias"))
                    and name not in params_dict):
                continue
            param = params_dict[full_param_name]
            weight_loader = param.weight_loader
            if fused:
                if "w13" in full_param_name:
                    shard_idx = 0 if shard_id == "w1" else 1
                    new_loaded_weight = new_loaded_weight[shard_idx]
                new_loaded_weight = new_loaded_weight.transpose(-1, -2)
                layer_idx = extract_layer_index(name)
                # EP mapping
                expert_map = self.layers[
                    layer_idx].feed_forward.experts.expert_map
                if expert_map is not None:
                    local_expert_indices = (expert_map != -1) \
                                            .nonzero() \
                                            .flatten() \
                                            .to(new_loaded_weight.device)
                    new_loaded_weight = new_loaded_weight[local_expert_indices]
                    expert_id = local_expert_indices[0].item()
            else:
                # TODO: add EP support for non fused weights
                pass
            weight_loader(param,
                          new_loaded_weight,
                          full_param_name,
                          shard_id=shard_id,
                          expert_id=expert_id)

            loaded_params.add(full_param_name)
            expert_param_loaded = True
        return expert_param_loaded

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        fused_experts_params = False
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts)
        expert_params_mapping_fused = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="gate_up_proj",
            num_experts=1)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                fused_experts_params = True
                expert_params_mapping = expert_params_mapping_fused
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                moe_loaded = self.load_moe_expert_weights(
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    expert_params_mapping,
                    fused=fused_experts_params)

                if not moe_loaded:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params


class Llama4ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # update temperature tuning config from generation config
        gen_config = vllm_config.model_config.try_get_generation_config()
        gen_config.update(vllm_config.model_config.override_generation_config)
        # enable temperature tuning by default when max_model_len > 32K
        default_attn_temperature_tuning = \
            vllm_config.model_config.max_model_len > 32768
        vllm_config.model_config.hf_config.attn_temperature_tuning \
            = gen_config.get(
                "attn_temperature_tuning", default_attn_temperature_tuning)

        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=Llama4DecoderLayer)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer):
        return Llama4Model(vllm_config=vllm_config,
                           prefix=prefix,
                           layer_type=layer_type)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        weights = [
            self.permute_qk_weight_for_rotary(name, loaded_weight)
            for name, loaded_weight in weights
        ]
        return loader.load_weights(weights)

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) \
           and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif ("wq" in modules or "q_proj" in modules) \
                and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        return name, loaded_weight
