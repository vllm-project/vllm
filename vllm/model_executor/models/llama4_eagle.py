# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama4 import (Llama4DecoderLayer,
                                               Llama4ForCausalLM)
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              extract_layer_index,
                                              is_pp_missing_parameter,
                                              maybe_prefix)


@support_torch_compile
class EagleLlama4Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 start_layer_id: int = 0):

        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        if vllm_config.speculative_config.quantization:
            self.quant_config = vllm_config.quant_config
        else:
            self.quant_config = None

        self.layers = nn.ModuleList([
            Llama4DecoderLayer(
                config=self.config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
            )
        ])
        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)
        self.num_experts = self.config.num_local_experts

        self.norm = RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(
            torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

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


class EagleLlama4ForCausalLM(Llama4ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)

        start_layer_id = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        if start_layer_id > 0:
            original_no_rope_layers = self.config.no_rope_layers

            # If start_layer_id is 0, we will hit NotImplementedError in
            # vllm/v1/utils.py. If we don't pad no_rope_layers, will get
            # index out of bounds in constructor of Llama4Attention layer.
            self.config.no_rope_layers = [None] * start_layer_id
            self.config.no_rope_layers.extend(original_no_rope_layers)

        self.model = EagleLlama4Model(vllm_config=vllm_config,
                                      prefix="model",
                                      start_layer_id=start_layer_id)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)

        model_weights = {}
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight
        loader.load_weights(model_weights.items())
