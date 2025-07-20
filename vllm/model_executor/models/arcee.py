# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2025 vLLM Team
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Inference-only Arcee (AFM) model – adds support for ReLU^2 feed-forward activation.

import logging
from collections.abc import Iterable
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    LlamaConfig)  # Reusing HuggingFace LLaMA config for Arcee

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, make_layers
from vllm.sequence import IntermediateTensors


class ArceeMLP(nn.Module):
    """Feed-forward layer for Arcee using ReLU^2 activation (no gating as in LLaMA)."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[Any] = None,
                 bias: bool = False,
                 prefix: str = "",
                 reduce_results: bool = True) -> None:
        super().__init__()
        # Single linear projection up to intermediate size (no separate gate projection)
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        # Down projection back to hidden size
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "relu2":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only 'relu2' is supported for AFM."
            )
        # Define ReLU^2 activation: (ReLU(x))^2 elementwise
        self.act_fn = lambda x: torch.pow(torch.relu(x), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)  # Project to intermediate size
        x = self.act_fn(x)  # Apply ReLU^2 activation elementwise
        x, _ = self.down_proj(x)  # Project back down to hidden size
        return x


class ArceeDecoderLayer(nn.Module):
    """Transformer decoder block for Arcee, with self-attention and ReLU^2 MLP."""

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
                "original_max_position_embeddings"] = config.original_max_position_embeddings
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
                "decoder"),  # assume decoder (causal) unless specified
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


@support_torch_compile
class ArceeModel(nn.Module):
    """The transformer model backbone for Arcee (embedding layer + stacked decoder blocks + final norm)."""

    def __init__(self,
                 *,
                 vllm_config,
                 prefix: str = "",
                 layer_type: type[nn.Module] = ArceeDecoderLayer) -> None:
        super().__init__()
        config: LlamaConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        # Calculate vocab size (include LoRA additional vocab if any)
        lora_vocab_extra = (lora_config.lora_extra_vocab_size *
                            (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab_extra
        self.org_vocab_size = config.vocab_size

        # Word embeddings (parallelized if using pipeline parallel)
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer(
            )  # placeholder on non-embedding ranks

        # Build decoder layers across pipeline ranks
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # Final RMSNorm on the last pipeline stage
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # For optional capturing of intermediate hidden states (not used by default)
        self.aux_hidden_state_layers: Tuple[int, ...] = tuple()

        # Prepare factory for empty intermediate tensors (for pipeline scheduling)
        from vllm.model_executor.models.utils import (
            make_empty_intermediate_tensors_factory)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors, Tuple[torch.Tensor,
                                                        List[torch.Tensor]]]:
        # Embedding lookup (on first pipeline rank)
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(
                input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None, "IntermediateTensors must be provided for non-first pipeline ranks"
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: List[torch.Tensor] = []
        for idx, layer in enumerate(
                self.layers[self.start_layer:self.end_layer]):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    hidden_states +
                    residual)  # capture pre-layer hidden state if needed
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            # Send intermediate results to the next pipeline stage
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # On last rank: apply final layer norm
        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load pre-trained weights from HuggingFace format into the model."""
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader, maybe_remap_kv_scale_name)
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        # Mapping for merging or renaming weight parameters from HF into our model
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params = set()

        for name, loaded_weight in weights:
            # Skip AFM's gate projection weights (they don't exist in ReLU^2 AFM)
            if "gate_proj" in name:
                continue
            # Skip rotary cache parameters if present (not actual model weights)
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Handle quantization KV cache scales
            if self.quant_config is not None and (
                    scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                maybe_name = maybe_remap_kv_scale_name(name, params_dict)
                if maybe_name is None:
                    continue
                name = maybe_name

            # Pipeline parallel: skip parameters not on this rank
            if is_pp_missing_parameter(name, self):
                continue

            # Handle combined stacked parameters (e.g., qkv_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(
                        ".bias") and mapped_name not in params_dict:
                    # Skip any unexpected biases (e.g., from certain quantization or GPTQ checkpoints)
                    break
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(
                        param, loaded_weight,
                        shard_id)  # load the shard into the combined param
                    loaded_params.add(mapped_name)
                else:
                    logging.warning(
                        f"Unexpected parameter in checkpoint: {name}")
                break
            else:
                # No special mapping, try direct load
                if name in params_dict:
                    # For tied embeddings, skip loading lm_head if it will be tied
                    if name.startswith("lm_head.") and getattr(
                            self.config, "tie_word_embeddings", False):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                else:
                    # Silently skip any unmatched parameters (e.g., vision tower weights in multimodal models)
                    logging.debug(
                        f"Ignoring unmatched checkpoint parameter: {name}")
        return loaded_params


class ArceeForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """Arcee Model for causal language modeling, integrated with vLLM runtime."""
    # Map fused module names to their sub-module components (for quantization and LoRA)
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    # Supported LoRA modules (same as LLaMA, minus gate_proj)
    supported_lora_modules = [
        "qkv_proj", "o_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"
    ]
    # (No MLP prefix since there's no gate_proj)
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        # Initialize the inner Transformer model (ArceeModel)
        self.model = ArceeModel(vllm_config=vllm_config,
                                prefix=f"{prefix}model")
        # On the last pipeline stage, set up the LM head and logits processor
        if get_pp_group().is_last_rank:
            # Determine vocabulary size (including any LoRA extra tokens for padded LM head)
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

            # Import DEFAULT_VOCAB_PADDING_SIZE
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                DEFAULT_VOCAB_PADDING_SIZE)

            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE if lora_config and lora_config.lora_extra_vocab_size > 0 else 0,
                quant_config=vllm_config.quant_config,
                bias=getattr(config, "lm_head_bias", False),
                prefix=f"{prefix}lm_head",
                input_is_parallel=True,
            )
            if config.tie_word_embeddings:
                # Tie output weights with input embedding matrix
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            # Placeholder for lm_head on non-last ranks
            self.lm_head = PPMissingLayer()
        # Provide a reference to the model's method for generating empty tensors (used in pipeline parallel schedule)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Forward pass through the Arcee model backbone
        model_output = self.model(input_ids=input_ids,
                                  positions=positions,
                                  intermediate_tensors=intermediate_tensors,
                                  inputs_embeds=inputs_embeds)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata) -> Optional[torch.Tensor]:
        # Compute final logits from hidden states (last pipeline rank only)
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load weights into the model (delegates to inner model and handles tied embeddings)."""
        # Use AutoWeightsLoader for consistency with vLLM's loading mechanism
        from vllm.model_executor.models.utils import AutoWeightsLoader
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None))
        # No special weight name remapping needed (AFM uses standard LLaMA naming except no gate_proj)
        return loader.load_weights(weights)
