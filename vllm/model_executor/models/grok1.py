# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/ROCm/vllm/blob/cea7419f151cc50293a05b7fac8547f8f887c9f6/vllm/model_executor/models/grok1.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Grok1 model."""
from collections.abc import Iterable
from itertools import islice
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.grok1_scaling_rope import (
    Grok1ScalingRotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# Default Grok1-specific constants, overridden by config values if present
DEFAULT_ATTN_OUTPUT_MULTIPLIER = 0.08838834764831845
DEFAULT_OUTPUT_MULTIPLIER_SCALE = 0.5773502691896257
DEFAULT_EMBEDDING_MULTIPLIER_SCALE = 78.38367176906169


class Grok1MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results=True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 tp_size: Optional[int] = None,
                 reduce_results: bool = True,
                 use_presharded_weights: bool = False,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(hidden_size,
                                     num_experts,
                                     bias=False,
                                     params_dtype=params_dtype,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = FusedMoE(num_experts=num_experts,
                                top_k=top_k,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                params_dtype=params_dtype,
                                reduce_results=reduce_results,
                                renormalize=True,
                                quant_config=quant_config,
                                tp_size=tp_size,
                                activation="gelu",
                                use_presharded_weights=use_presharded_weights,
                                prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        router_logits = 30.0 * F.tanh(router_logits / 30.0)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


def get_rope_scaling(config):
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", None)
        scaling_factor = getattr(config, "scaling_factor", None)
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)
        attn_factor = getattr(config, "attn_factor", 1.0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)
        rope_scaling = {
            "extra_method": rope_type,
            "max_position_embeddings": original_max_position_embeddings,
            "scaling_factor": scaling_factor,
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "dtype": torch.float,
        }
        return rope_scaling
    else:
        return None


class Grok1Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        config=None,  # Added config parameter
        reduce_results: bool = True,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config  # Store config reference
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        rope_scaling = get_rope_scaling(config)
        self.alt_stream = alt_stream or torch.cuda.Stream()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        attn_logits_soft_cap = max(
            getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims",
                                             False)

        if rope_scaling is not None:
            self.rotary_emb = Grok1ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(self.head_dim if not self.rope_rotate_half_dims
                            else self.head_dim // 2),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=(self.head_dim if not self.rope_rotate_half_dims
                            else self.head_dim // 2),
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              logits_soft_cap=attn_logits_soft_cap,
                              prefix=f"{prefix}.attn")
        self.attn_multiplier = getattr(
            self.config, "attn_output_multiplier",
            DEFAULT_ATTN_OUTPUT_MULTIPLIER) if self.config else 1.0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        output *= self.attn_multiplier
        return output


class Grok1DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        load_presharded_moe: bool = False,
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_grok1: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts if is_grok1 else config.num_local_experts
        # Check for fp8 quantization
        self.use_fp8 = False
        if quant_config is not None:
            self.use_fp8 = getattr(quant_config, "is_fp8_w8a8",
                                   lambda: False)()
            if not self.use_fp8 and hasattr(quant_config, "is_fp8"):
                self.use_fp8 = quant_config.is_fp8
        self.residual_moe = getattr(config, "residual_moe", False)
        self.alt_stream = alt_stream or torch.cuda.Stream()
        self.is_grok1 = is_grok1

        # Requires transformers > 4.32.0
        # Default rope_theta value if not in config
        rope_theta = getattr(config, "rope_theta", 10000)
        if is_grok1:
            self.attn = Grok1Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                max_position=(config.context_len if hasattr(
                    config, "context_len") else
                              config.max_position_embeddings),
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                config=config,  # Pass config to Grok1Attention
                alt_stream=self.alt_stream,
            )
        else:
            self.self_attn = Grok1Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                max_position=(config.context_len if hasattr(
                    config, "context_len") else
                              config.max_position_embeddings),
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                config=config,  # Pass config to Grok1Attention
                alt_stream=self.alt_stream,
            )

        if self.num_experts > 0:
            if is_grok1:
                self.moe_block = Grok1MoE(
                    num_experts=self.num_experts,
                    top_k=config.num_experts_per_tok,
                    hidden_size=config.hidden_size,
                    intermediate_size=getattr(
                        config,
                        "moe_intermediate_size",
                        getattr(config, "intermediate_size", None),
                    ),
                    quant_config=quant_config,
                    reduce_results=not self.residual_moe,
                    use_presharded_weights=load_presharded_moe,
                    prefix=f"{prefix}.moe_block")
            else:
                self.block_sparse_moe = Grok1MoE(
                    num_experts=self.num_experts,
                    top_k=config.num_experts_per_tok,
                    hidden_size=config.hidden_size,
                    intermediate_size=getattr(
                        config,
                        "moe_intermediate_size",
                        getattr(config, "intermediate_size", None),
                    ),
                    quant_config=quant_config,
                    reduce_results=not self.residual_moe,
                    use_presharded_weights=load_presharded_moe,
                    prefix=f"{prefix}.block_sparse_moe")
            if self.residual_moe:
                self.mlp = Grok1MLP(hidden_size=config.hidden_size,
                                    intermediate_size=config.intermediate_size,
                                    quant_config=quant_config,
                                    reduce_results=False,
                                    prefix=f"{prefix}.mlp")
        else:
            raise NotImplementedError("Number of experts must be > 0.")

        self.pre_attn_norm = RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

        if self.num_experts > 0:
            if self.residual_moe:
                # NOTE: self.block_sparse_moe modifies the input in-place,
                # so we have to call it later. Be aware of any possible related errors.
                if get_tensor_model_parallel_world_size() > 1:
                    self.ffn = lambda x: tensor_model_parallel_all_reduce(
                        self.moe_with_rmoe(x))
                else:
                    self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.moe_block if is_grok1 else self.block_sparse_moe
        else:
            raise NotImplementedError("Number of experts must be > 0.")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_attn_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_attn_norm(
                hidden_states, residual)

        if self.is_grok1:
            hidden_states = self.attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        # Post attention normalization
        hidden_states = self.post_attn_norm(hidden_states)

        # Fully Connected
        hidden_states, residual = self.pre_moe_norm(hidden_states, residual)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)

        return hidden_states, residual

    def moe_with_rmoe(self, x):
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        mlp_result = self.mlp(x)
        with torch.cuda.stream(self.alt_stream):
            # moe should not be inplace because of stream race condition
            moe_result = self.moe_block(
                x) if self.is_grok1 else self.block_sparse_moe(x)
        current_stream.wait_stream(self.alt_stream)
        return (mlp_result + moe_result) / 1.4142135623730951


@support_torch_compile
class Grok1Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 load_presharded_moe: bool = False,
                 is_grok1: bool = False,
                 prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embedding_multiplier_scale = getattr(
            config, "embedding_multiplier_scale",
            DEFAULT_EMBEDDING_MULTIPLIER_SCALE)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )

        self.alt_stream = torch.cuda.Stream()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Grok1DecoderLayer(config,
                                             cache_config,
                                             quant_config=quant_config,
                                             load_presharded_moe=
                                             load_presharded_moe,
                                             alt_stream=self.alt_stream,
                                             is_grok1=is_grok1,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.is_grok1 = is_grok1

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embedding_multiplier_scale
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Map Grok1's unique expert parameter names to standard names
        num_experts = self.config.num_experts if self.is_grok1 else self.config.num_local_experts
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="linear" if self.is_grok1 else "w1",
            ckpt_down_proj_name="linear_1" if self.is_grok1 else "w2",
            ckpt_up_proj_name="linear_v" if self.is_grok1 else "w3",
            num_experts=num_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
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

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    # Handle Grok1-specific norm.scale naming
                    if "norm.scale" in name:
                        name = name.replace("scale", "weight")

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Grok1ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        # Check the model architecture to handle parameter differences between Grok1 and Grok2.
        architectures = getattr(config, "architectures", [])
        is_grok1 = architectures and architectures[0] == "Grok1ModelForCausalLM"
        num_experts = config.num_experts if is_grok1 else config.num_local_experts

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.load_presharded_moe = (
            getattr(config, "load_presharded_moe", True) and num_experts > 0
            and get_tensor_model_parallel_world_size() > 1 and not is_grok1)

        self.model = Grok1Model(vllm_config=vllm_config,
                                load_presharded_moe=self.load_presharded_moe,
                                is_grok1=is_grok1,
                                prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.output_multiplier_scale = getattr(
            config, "output_multiplier_scale", DEFAULT_OUTPUT_MULTIPLIER_SCALE)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                self.output_multiplier_scale)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Skip lm_head when tie_word_embeddings is True
        skip_prefixes = (["lm_head"]
                         if self.config.tie_word_embeddings else None)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
