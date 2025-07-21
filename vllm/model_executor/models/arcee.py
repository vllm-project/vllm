# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2025 vLLM Team
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Inference-only Arcee (AFM) model – adds support for ReLU^2 feed-forward
# activation.

from collections.abc import Iterable
from typing import Any, Callable, Optional  # Removed Type per UP035

import torch
from torch import nn
from transformers import LlamaConfig  # Reusing HuggingFace LLaMA config for Arcee

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, is_pp_missing_parameter
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name


class ArceeMLP(nn.Module):
    """Feed-forward layer for Arcee using ReLU^2 activation
    (no gating as in LLaMA)."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[Any] = None,
                 bias: bool = False,
                 prefix: str = "",
                 reduce_results: bool = True) -> None:
        super().__init__()
        if hidden_act != "relu2":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only 'relu2' is supported for Arcee.")
        # No gate projection in Arcee
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # Define ReLU^2 activation: (ReLU(x))^2 elementwise
        self.act_fn: Callable[[torch.Tensor],
                               torch.Tensor] = lambda x: torch.pow(
                                   torch.relu(x), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)  # Project to intermediate size
        x = self.act_fn(x)  # Apply ReLU^2 activation elementwise
        x, _ = self.down_proj(x)  # Project back down to hidden size
        return x


class ArceeDecoderLayer(nn.Module):
    """Transformer decoder block for Arcee, with self-attention and
    ReLU^2 MLP."""

    def __init__(self,
                 config: LlamaConfig,
                 cache_config: Optional[Any] = None,
                 quant_config: Optional[Any] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Rotary embedding parameters (reuse LLaMA defaults)
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling[
                "original_max_position_embeddings"] = (
                    config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Determine if attention bias is needed (some variants use bias terms)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # Self-Attention (using LLaMA's attention structure)
        from vllm.model_executor.models.llama import (
            LlamaAttention)  # import here to avoid circular import
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=getattr(
                config, "attn_type",
                "decoder"),  # default to causal decoder
        )
        # MLP with ReLU^2 activation
        self.mlp = ArceeMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        # Layer normalization layers (RMSNorm as in LLaMA)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self, positions: torch.Tensor, hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention block
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Fused residual add + layernorm if supported
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        # Feed-forward block
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Add ArceeModel class after ArceeDecoderLayer
class ArceeModel(LlamaModel):
    """Custom model class for Arcee that handles weight loading properly."""
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Arcee-specific handling (no gate projection)."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # No gate_up_proj mapping for Arcee - it only has up_proj and down_proj
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            # Skip weights that don't apply to Arcee
            if "gate_proj" in name or "gate_up_proj" in name:
                continue
                
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
                
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
                
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is None:
                    continue
                name = remapped_name
                    
            # Check stacked params mapping
            weight_loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                    
                if is_pp_missing_parameter(name, self):
                    continue
                    
                param = params_dict[name]
                weight_loader = param.weight_loader  # type: ignore
                weight_loader(param, loaded_weight, shard_id)
                weight_loaded = True
                break
                
            if not weight_loaded:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                    
                if is_pp_missing_parameter(name, self):
                    continue
                    
                # Ensure the parameter exists
                if name not in params_dict:
                    # If it's a weight we expect but with different naming, skip it
                    continue
                    
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                
            loaded_params.add(name)
            
        return loaded_params


class ArceeForCausalLM(LlamaForCausalLM):
    """Arcee Model for causal language modeling, integrated with vLLM
    runtime."""
    # Map fused module names to their sub-module components
    # (for quantization and LoRA)
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        # NOTE: Arcee doesn't have gate_proj, only up_proj and down_proj
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        # Initialize LlamaForCausalLM
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=ArceeDecoderLayer,
        )
    
    def _init_model(self, vllm_config: Any, prefix: str = "", layer_type: type[nn.Module] = ArceeDecoderLayer) -> nn.Module:
        # Use ArceeModel with ArceeDecoderLayer
        return ArceeModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights into the model with Arcee-specific handling."""
        # Use AutoWeightsLoader for consistency with vLLM's loading mechanism
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None))
        
        # Filter out gate_proj weights (Arcee doesn't have gate projection)
        filtered_weights = [(name, weight) for name, weight in weights 
                           if "gate_proj" not in name]
        
        # AutoWeightsLoader handles weight name remapping, including fusing
        # separate q_proj, k_proj, v_proj into qkv_proj
        return loader.load_weights(filtered_weights)
