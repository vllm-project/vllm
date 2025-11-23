# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .nemotron_h import NemotronHAttentionDecoderLayer
from .utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)


class NemotronHMultiTokenPredictorLayer(nn.Module):
    """MTP layer using NemotronH attention architecture."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        # Normalization layers for fusion
        self.mtp_emb_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mtp_hidden_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Fusion layer to combine embeddings with target hidden states
        self.mtp_linear_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        
        # Use NemotronH's native attention layer (compatible with NemotronHConfig)
        self.mtp_block = NemotronHAttentionDecoderLayer(
            config=config,  # type: ignore[arg-type]
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.mtp_block",
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        """Forward pass: fuse embeddings + hidden states, then apply attention."""
        assert inputs_embeds is not None
        
        # Normalize both inputs before fusion
        inputs_embeds_normed = self.mtp_emb_norm(inputs_embeds)
        previous_hidden_states_normed = self.mtp_hidden_norm(previous_hidden_states)

        # Fuse via concatenation and linear projection
        hidden_states = self.mtp_linear_proj(
            torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)
        )

        # Apply attention block with residual connection
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        
        # Final residual addition
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with attention layers (similar to Qwen3NextMTP)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size
        
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Create MTP layers with attention (using ModuleDict for layer indexing)
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): NemotronHMultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    layer_idx=idx,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward with PP support and proper residual handling."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        
        # Use the MTP layer (cycling for multi-step)
        layer_idx_str = str(self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers))
        hidden_states = self.layers[layer_idx_str](
            inputs_embeds,
            positions,
            hidden_states,  # previous hidden states from target
            spec_step_idx,
        )

        # Final normalization
        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class NemotronHMTP(nn.Module, SupportsPP):
    """NemotronH MTP model with attention layers."""

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
        cache_config = vllm_config.cache_config
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        assert not cache_config.enable_prefix_caching, (
            "NemotronHMTP currently does not support prefix caching"
        )

        # MTP predictor with attention layers
        self.model = NemotronHMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        
        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size
        )
        
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            # Skip tied embeddings if configured
            if self.config.tie_word_embeddings and name.endswith("lm_head.weight"):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Remap MTP layer names
            if "mtp" in name or "backbone.norm.weight" in name or "embeddings" in name:
                name = self._rewrite_spec_layer_name(self.config, name)
            
            # Handle stacked parameters (qkv_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "model.layers" not in name:  # Only MTP layers
                    continue
                    
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle non-stacked parameters
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue

                # Only load MTP-specific weights and shared embeddings/lm_head
                if "model.layers" not in name and "embed_tokens" not in name and "lm_head" not in name and "model.norm" not in name:
                    continue

                if name not in params_dict:
                    continue
                
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            
            
            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, config: PretrainedConfig, name: str) -> str:
        """
        Rewrite the weight name to match the internal model structure.
        """
        if "embeddings" in name:
            name = name.replace("embeddings", "embed_tokens")
        if name.startswith("backbone."):
            name = name.replace("backbone.", "model.")
        return name
