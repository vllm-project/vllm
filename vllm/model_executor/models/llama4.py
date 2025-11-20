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

import torch
from torch import nn
from transformers import Llama4TextConfig

from vllm.attention import Attention
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import sequence_parallel_chunk

from .llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    fast_topk,
    is_pp_missing_parameter,
)

logger = init_logger(__name__)


class Llama4MoE(nn.Module):
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
        # pseudo-standard is that the router scores are floats
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.router",
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.shared_expert",
            reduce_results=False,
            disable_tp=self.is_sequence_parallel,
        )

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config if parallel_config else None
        self.enable_eplb = parallel_config.enable_eplb if parallel_config else False
        self.n_redundant_experts = (
            eplb_config.num_redundant_experts if eplb_config else 0
        )

        self.n_routed_experts: int = config.num_local_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_shared_experts: int = 1
        self.n_local_experts: int = config.num_local_experts
        self.n_physical_experts = self.n_local_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            apply_router_weight_on_input=True,
            reduce_results=False,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.is_sequence_parallel,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.router(hidden_states)

        shared_out, routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        experts_out = routed_out + shared_out

        if self.is_sequence_parallel:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]
        elif self.tp_size > 1:
            experts_out = self.experts.maybe_all_reduce_tensor_model_parallel(
                experts_out
            )

        return experts_out


class Llama4Attention(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
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
        self.attn_temperature_tuning = self.nope and config.attn_temperature_tuning

        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                dtype=torch.float32,
            )
            if self.use_qk_norm
            else None
        )
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

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=is_neox_style,
            )
            if not self.nope
            else None
        )

        use_chunked_local_attn = not self.nope and config.attention_chunk_size
        attn_cls = ChunkedLocalAttention if use_chunked_local_attn else Attention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **(
                {"attention_chunk_size": config.attention_chunk_size}
                if use_chunked_local_attn
                else {}
            ),
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
            # Normalization is applied on the head_dim dimension. The rest of
            # the dimensions are collapsed into a single dimension to support
            # custom rms_norm cuda kernel.
            q = q.reshape(-1, self.head_dim)
            q = self.qk_norm(q.float()).reshape(-1, self.q_size).to(q.dtype)
            k = k.reshape(-1, self.head_dim)
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
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Llama4TextConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_idx = extract_layer_index(prefix)
        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
        self.hidden_size = config.hidden_size
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        is_moe_layer = (
            config.interleave_moe_layer_step > 0
            and (self.layer_idx + 1) % config.interleave_moe_layer_step == 0
        )
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                vllm_config=vllm_config,
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Llama4Model(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        self.num_experts = vllm_config.model_config.hf_config.num_local_experts
        self.n_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        loaded_params: set[str],
        expert_params_mapping: list[tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool:
        """
        Load MoE expert weights.

        Args:
            name: The name of the weight to load.
            loaded_weight: The weight to load.
            params_dict: The dictionary of module parameters.
            loaded_params: The set of already loaded parameters.
            expert_params_mapping: The mapping of expert parameters. Must be
                generated by SharedFusedMoE.make_expert_params_mapping().
            fused: Whether the expert weights are fused into a single weight
                tensor or are separate weight tensors for each expert.
                When fused is True, loaded_weight should have shape of:
                [num_experts, hidden_in, hidden_out] for gate/up/down proj and
                [hidden_out, hidden_in] for the others like router.
                When fused is False, loaded_weight should have shape of:
                [hidden_out, hidden_in].

        Returns:
            True if loaded_weight is one of MoE weights and the MoE expert
            weights are loaded successfully, False otherwise.
        """

        # Whether the MoE expert weights are loaded successfully.
        expert_param_loaded = False

        # If fused is True, the loaded weight is in the layout of:
        # [num_experts, hidden_in, hidden_out], so we must transpose the last
        # two dimensions to match the expected layout of the parameters.
        if fused and loaded_weight.ndim == 3:
            loaded_weight = loaded_weight.transpose(-1, -2)

            # If the gate_proj and up_proj weights are fused into a single
            # weight tensor, we need to split the weight tensor into a tuple
            # of two weight tensors along the hidden_out dimension.
            if "experts.gate_up_proj" in name:
                loaded_weight = loaded_weight.chunk(2, dim=-2)

        # Iterate over all the expert parameters and load the weights if we find
        # a match in weight name.
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            # Get a view of the loaded_weight to avoid modifying the original
            # one across iterations.
            new_loaded_weight = loaded_weight

            # If expert weights are fused into a single weight tensor, remove
            # the expert index from the expected weight name.
            if fused:
                # The string between e_str and proj_str is the expert index.
                e_str, _, proj_str, _ = weight_name.split(".")
                weight_name = f"{e_str}.{proj_str}"
                param_name = f"{param_name}weight"

            # Skip if the current weight is not one of the MoE weights.
            if weight_name not in name:
                continue

            # Replace the weight name with the parameter name.
            full_param_name = name.replace(weight_name, param_name)

            # Skip if the current weight corresponds to a parameter that
            # does not exist on the current PP (pipeline parallel) rank.
            if is_pp_missing_parameter(name, self):
                continue

            # Skip if the current weight is for the bias.
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue

            param = params_dict[full_param_name]
            weight_loader = param.weight_loader

            if fused:
                # If the parameter is for w13 together, the corresponding weight
                # will be a tuple, so we must select the correct weight
                # depending on the shard id, which is either "w1" or "w3".
                if "w13" in full_param_name:
                    assert shard_id in ["w1", "w3"]
                    shard_idx = 0 if shard_id == "w1" else 1
                    new_loaded_weight = new_loaded_weight[shard_idx]

                # If EP (expert parallel) is enabled, update expert_id to the
                # starting expert index for the current EP rank and extract the
                # corresponding expert weights.
                layer_idx = extract_layer_index(name)
                expert_map = self.layers[layer_idx].feed_forward.experts.expert_map
                if expert_map is not None:
                    local_expert_indices = (
                        (expert_map != -1)
                        .nonzero()
                        .flatten()
                        .to(new_loaded_weight.device)
                    )
                    new_loaded_weight = new_loaded_weight[local_expert_indices]
                    expert_id = local_expert_indices[0].item()
            else:
                # TODO: add EP support for non fused weights
                pass

            # Load the weight into the module parameter with corresponding
            # shard id and expert id.
            weight_loader(
                param,
                new_loaded_weight,
                full_param_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            loaded_params.add(full_param_name)
            expert_param_loaded = True

        return expert_param_loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Name mapping from the parameter name to the shard name and
        # corresponding shard id.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # Indicate whether the expert weights are fused into a single weight
        # tensor.
        fused_experts_params = False
        # Expert parameter mapping for the case where the expert weights are
        # not fused into a single weight tensor.
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts,
            num_redundant_experts=self.n_redundant_experts,
        )
        # Expert parameter mapping for the case where the expert weights are
        # fused into a single weight tensor.
        expert_params_mapping_fused = SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="gate_up_proj",
            num_experts=1,
        )
        # All the module parameters.
        params_dict = dict(self.named_parameters())
        # The module parameters that have been loaded.
        loaded_params: set[str] = set()

        # Iterate over all the weights and load them into module parameters.
        for name, loaded_weight in weights:
            # If the name contains "experts.gate_up_proj" or "experts.down_proj"
            # without the expert indices, it means the expert weights are fused
            # into a single weight tensor across all experts.
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                fused_experts_params = True
                expert_params_mapping = expert_params_mapping_fused

            # If kv cache quantization scales exist and the weight name
            # corresponds to one of the kv cache quantization scales, load
            # them.
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # Iterate over stacked_params_mapping to check if the current weight
            # is one of the stacked parameters. If so, load the weight with the
            # corresponding shard id. Note that MoE weights are handled
            # separately in the else block.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip if the current weight is not one of the stacked
                # parameters or if the current weight is a MoE weight.
                if weight_name not in name or "experts" in name:
                    continue

                # For ModelOpt checkpoints, we need to rename the self_attn
                # weight/weight_scale names except for kv cache scales.
                if not (
                    name.endswith((".k_scale", ".v_scale")) and "self_attn" in name
                ):
                    name = name.replace(weight_name, param_name)

                # Skip if the current weight corresponds to a parameter that
                # does not exist on the current PP (pipeline parallel) rank.
                if is_pp_missing_parameter(name, self):
                    continue

                # Remap kv cache scale names for ModelOpt checkpoints.
                # TODO: ModelOpt should implement get_cache_scale() such that
                #       kv cache scale name remapping can be done there.
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                # Load the weight into the module parameter with corresponding
                # shard id and exit the for loop and the else block.
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)

                loaded_params.add(name)
                break

            # Handle normal (non-stacked) weights and MoE weights.
            else:
                # First, try to load MoE weights using load_moe_expert_weights.
                # If successful, move on to next loaded weight.
                if self.load_moe_expert_weights(
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    expert_params_mapping,
                    fused=fused_experts_params,
                ):
                    continue

                # Skip if the current weight corresponds to a parameter that
                # does not exist on the current PP (pipeline parallel) rank.
                if is_pp_missing_parameter(name, self):
                    continue

                # Handle flat expert scale parameters that don't match
                # per-expert patterns, i.e. one weight scale tensor for all
                # experts.
                scale_names = [
                    "w13_input_scale",
                    "w13_weight_scale",
                    "w2_input_scale",
                    "w2_weight_scale",
                ]
                if "experts." in name and any(
                    scale_name in name for scale_name in scale_names
                ):
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )

                    # If weight loader supports special moe loading, use it to
                    # avoid expensive runtime reflection
                    if getattr(weight_loader, "supports_moe_loading", False):
                        # Map the weight name to the corresponding shard id.
                        shard_id = "w2" if "w2_" in name else "w1"

                        # Transpose if weight scales are FP8 block scales with
                        # three dimensions:
                        # [num_experts, hidden_in, hidden_out].
                        if (
                            name.endswith("weight_scale")
                            and loaded_weight.dtype == torch.float8_e4m3fn
                            and loaded_weight.ndim == 3
                        ):
                            loaded_weight = loaded_weight.transpose(-1, -2)

                        # Load the weight into the module parameter with
                        # corresponding shard id and expert id.
                        weight_loader(
                            param, loaded_weight, name, shard_id=shard_id, expert_id=0
                        )

                    else:
                        # Regular weight loader (handles both
                        # param.weight_loader and default_weight_loader)
                        weight_loader(param, loaded_weight)

                    loaded_params.add(name)
                    continue

                # Handle normal (non-stacked, non-MoE) weights.
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Finally, return the set of loaded parameters.
        return loaded_params


class Llama4ForCausalLM(LlamaForCausalLM, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # update temperature tuning config from generation config
        gen_config = vllm_config.model_config.try_get_generation_config()
        gen_config.update(vllm_config.model_config.override_generation_config)
        # enable temperature tuning by default when max_model_len > 32K
        default_attn_temperature_tuning = vllm_config.model_config.max_model_len > 32768
        vllm_config.model_config.hf_config.attn_temperature_tuning = gen_config.get(
            "attn_temperature_tuning", default_attn_temperature_tuning
        )

        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=Llama4DecoderLayer
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            assert isinstance(layer, Llama4DecoderLayer)
            if isinstance(layer.feed_forward, Llama4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No Llama4MoE layer found in model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.feed_forward, Llama4MoE):
                moe = layer.feed_forward
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        return Llama4Model(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
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
        # Helper function to permute the weight's channels
        def permute(w: torch.Tensor, n_heads: int, is_weight_scale: bool):
            # Calculate the expected shape of the weight.
            # Do not rely on w's shape, as it may be in another layout.
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            # If the weight is FP4 packed as uint8, we need to divide attn_out
            # by 2.
            if w.dtype == torch.uint8 and w.shape[1] * 2 == attn_out:
                attn_out = attn_out // 2

            # If the weight is a weight scale, we need to divide attn_out by
            # block size, which is currently 16.
            elif (
                w.dtype == torch.float8_e4m3fn
                and is_weight_scale
                and w.shape[1] * 16 == attn_out
            ):
                attn_out = attn_out // 16

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # Permute Q/K weights and weight block scales for rotary embedding
        is_weight = modules[-1] == "weight"
        is_nvfp4_weight_scale = (
            modules[-1] == "weight_scale" and loaded_weight.dtype == torch.float8_e4m3fn
        )

        if is_weight or is_nvfp4_weight_scale:
            if "wk" in modules or "k_proj" in modules:
                loaded_weight = permute(
                    loaded_weight,
                    self.config.num_key_value_heads,
                    is_nvfp4_weight_scale,
                )
            elif "wq" in modules or "q_proj" in modules:
                loaded_weight = permute(
                    loaded_weight,
                    self.config.num_attention_heads,
                    is_nvfp4_weight_scale,
                )

        return name, loaded_weight
