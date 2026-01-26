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
"""Inference-only Grok (Grok1/Grok2) model."""

import math
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# Default Grok1-specific constants, overridden by config values if present
DEFAULT_ATTN_OUTPUT_MULTIPLIER = 0.08838834764831845
DEFAULT_OUTPUT_MULTIPLIER_SCALE = 0.5773502691896257
DEFAULT_EMBEDDING_MULTIPLIER_SCALE = 78.38367176906169
DEFAULT_ROUTER_LOGIT_SOFTCAP = 30.0

logger = init_logger(__name__)


def _get_num_experts(config) -> int:
    return getattr(config, "num_experts", getattr(config, "num_local_experts", 8))


def _get_moe_intermediate_size(config) -> int:
    return getattr(config, "moe_intermediate_size", config.intermediate_size)


def _get_grok_version(config) -> str:
    """Detect Grok version from HF config using multiple heuristics."""
    # Check for Grok2-specific attributes (both for robust detection)
    has_residual_moe = getattr(config, "residual_moe", False)
    has_moe_intermediate_size = hasattr(config, "moe_intermediate_size")

    if has_residual_moe or has_moe_intermediate_size:
        return "grok2"

    return "grok1"  # Default to Grok1


def _get_rope_parameters(config) -> dict[str, Any] | None:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        rope_type = getattr(config, "rope_type", None)
        if rope_type is None:
            return None
        rope_parameters = {"rope_type": rope_type}
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is not None:
            rope_parameters["rope_theta"] = rope_theta
        scaling_factor = getattr(config, "scaling_factor", None)
        if scaling_factor is not None:
            rope_parameters["factor"] = scaling_factor
        for name in (
            "original_max_position_embeddings",
            "extrapolation_factor",
            "attn_factor",
            "beta_fast",
            "beta_slow",
        ):
            value = getattr(config, name, None)
            if value is not None:
                rope_parameters[name] = value

    if rope_parameters.get("rope_type") == "original":
        rope_parameters = dict(rope_parameters)
        rope_parameters["rope_type"] = "default"
    return rope_parameters


def _get_moe_renormalize(config) -> bool:
    explicit_value = getattr(
        config, "moe_router_renormalize", getattr(config, "moe_renormalize", None)
    )
    if explicit_value is not None:
        return bool(explicit_value)
    return not getattr(config, "residual_moe", False)


class Grok1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        router_logit_soft_cap: float = 0.0,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        renormalize: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=renormalize,
            quant_config=quant_config,
            tp_size=tp_size,
            activation="gelu",
            prefix=f"{prefix}.experts",
        )
        self.router_logit_soft_cap = router_logit_soft_cap

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if self.router_logit_soft_cap > 0:
            router_logits = self.router_logit_soft_cap * F.tanh(
                router_logits / self.router_logit_soft_cap
            )
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class Grok1Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_parameters: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        config=None,  # Added config parameter
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
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        attn_logits_soft_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        attn_logit_softcapping_method = getattr(
            config, "attn_logit_softcapping_method", None
        )
        if attn_logit_softcapping_method not in (None, "tanh"):
            logger.warning_once(
                "Grok attention logit softcapping method '%s' is not "
                "supported; falling back to default behavior.",
                attn_logit_softcapping_method,
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            prefix=f"{prefix}.attn",
        )
        self.attn_multiplier = (
            getattr(self.config, "attn_output_multiplier", 1.0) if self.config else 1.0
        )

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
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Check for fp8 quantization
        self.use_fp8 = False
        if quant_config is not None:
            self.use_fp8 = getattr(quant_config, "is_fp8_w8a8", lambda: False)()
            if not self.use_fp8 and hasattr(quant_config, "is_fp8"):
                self.use_fp8 = quant_config.is_fp8

        self.attn = Grok1Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=_get_rope_parameters(config),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            config=config,
        )  # Pass config to Grok1Attention

        num_experts = _get_num_experts(config)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        moe_intermediate_size = _get_moe_intermediate_size(config)
        moe_renormalize = _get_moe_renormalize(config)

        self.moe_block = Grok1MoE(
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size,
            router_logit_soft_cap=max(
                getattr(
                    config,
                    "router_logit_softcapping",
                    DEFAULT_ROUTER_LOGIT_SOFTCAP,
                ),
                0.0,
            ),
            quant_config=quant_config,
            renormalize=moe_renormalize,
            prefix=f"{prefix}.moe_block",
        )
        self.residual_moe = getattr(config, "residual_moe", False)
        self.residual_moe_scale = 1.0 / math.sqrt(2.0)

        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = None
        if self.residual_moe:
            self.mlp = Grok1MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
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
            hidden_states = self.pre_attn_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_attn_norm(hidden_states, residual)

        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Post attention normalization
        hidden_states = self.post_attn_norm(hidden_states)

        # MoE block with normalization
        hidden_states, residual = self.pre_moe_norm(hidden_states, residual)
        if self.residual_moe:
            assert self.mlp is not None
            hidden_states = (
                self.moe_block(hidden_states) + self.mlp(hidden_states)
            ) * self.residual_moe_scale
        else:
            hidden_states = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Grok1Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        ckpt_gate_proj_name: str = "linear",
        ckpt_down_proj_name: str = "linear_1",
        ckpt_up_proj_name: str = "linear_v",
        weight_name_remapping: dict[str, str] | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id

        # Store expert naming for weight loading
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        self.weight_name_remapping = weight_name_remapping or {}

        self.vocab_size = config.vocab_size

        self.embedding_multiplier_scale = getattr(
            config, "embedding_multiplier_scale", DEFAULT_EMBEDDING_MULTIPLIER_SCALE
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Grok1DecoderLayer(
                config, cache_config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embedding_multiplier_scale
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Map expert parameter names to standard names
        num_experts = _get_num_experts(self.config)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name=self.ckpt_gate_proj_name,
            ckpt_down_proj_name=self.ckpt_down_proj_name,
            ckpt_up_proj_name=self.ckpt_up_proj_name,
            num_experts=num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            # Apply version-specific weight name remapping
            for old_pattern, new_pattern in self.weight_name_remapping.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
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
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
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

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GrokBaseForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """Base class for Grok models with shared logic."""

    fall_back_to_pt_during_load = False

    # Subclasses should override these
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # Expert weight naming - subclasses override these
    ckpt_gate_proj_name: str = "linear"
    ckpt_down_proj_name: str = "linear_1"
    ckpt_up_proj_name: str = "linear_v"

    def get_weight_name_remapping(self) -> dict[str, str]:
        """Return weight name remapping for this version. Override in subclasses."""
        return {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = Grok1Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            ckpt_gate_proj_name=self.ckpt_gate_proj_name,
            ckpt_down_proj_name=self.ckpt_down_proj_name,
            ckpt_up_proj_name=self.ckpt_up_proj_name,
            weight_name_remapping=self.get_weight_name_remapping(),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.output_multiplier_scale = getattr(
            config, "output_multiplier_scale", DEFAULT_OUTPUT_MULTIPLIER_SCALE
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            scale=self.output_multiplier_scale,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Skip lm_head when tie_word_embeddings is True
        skip_prefixes = ["lm_head"] if self.config.tie_word_embeddings else None

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class Grok1ForCausalLM(GrokBaseForCausalLM):
    """Grok1-specific implementation."""

    # Grok1 expert weight naming
    ckpt_gate_proj_name = "linear"
    ckpt_down_proj_name = "linear_1"
    ckpt_up_proj_name = "linear_v"

    def get_weight_name_remapping(self) -> dict[str, str]:
        # Grok1 uses standard naming, no remapping needed
        return {}


class Grok2ForCausalLM(GrokBaseForCausalLM):
    """Grok2-specific implementation."""

    # Grok2 has additional packed modules for MLP
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Grok2 expert weight naming
    ckpt_gate_proj_name = "w1"
    ckpt_down_proj_name = "w2"
    ckpt_up_proj_name = "w3"

    def get_weight_name_remapping(self) -> dict[str, str]:
        # Grok2 checkpoint uses different naming conventions
        return {
            ".self_attn.": ".attn.",
            ".block_sparse_moe.": ".moe_block.",
        }


# Version dispatch mapping
_GROK_VERSIONS: dict[str, type[GrokBaseForCausalLM]] = {
    "grok1": Grok1ForCausalLM,
    "grok2": Grok2ForCausalLM,
}


class GrokForCausalLM(GrokBaseForCausalLM):
    """Factory class that dispatches to version-specific implementation."""

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        version = _get_grok_version(config)

        instance_cls = _GROK_VERSIONS.get(version)
        if instance_cls is None:
            raise ValueError(f"Unsupported Grok version: {version}")

        # Merge class attributes for LoRA/quantization compatibility
        cls.packed_modules_mapping = dict(cls.packed_modules_mapping)
        cls.packed_modules_mapping.update(instance_cls.packed_modules_mapping)

        return instance_cls(vllm_config=vllm_config, prefix=prefix)
