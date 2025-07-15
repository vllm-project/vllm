# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import (LlamaDecoderLayer,
                                              LlamaForCausalLM)

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: LlamaConfig,
        disable_input_layernorm: bool,
        prefix: str = "",
    ) -> None:
        super().__init__(config, prefix=prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if disable_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


@support_torch_compile
class LlamaModel(nn.Module):
    """
    Eagle draft model based on Llama architecture with projection layer.
    
    This model extends the standard Llama architecture for Eagle speculative decoding
    by adding a projection layer that combines input embeddings with hidden states
    from the target model. It also supports HASS (Hierarchical Aggregation for
    Sequence Sketching) variants that include additional layernorm layers.
    
    The projection layer takes concatenated input embeddings and hidden states
    (2 * hidden_size) and projects them back to hidden_size for processing
    through the transformer layers.
    """
    
    # Weight name mapping for speculators format compatibility
    SPECULATORS_WEIGHT_MAP = {
        "fusion_fc.weight": "projection_layer.weight",
        "fusion_fc.bias": "projection_layer.bias",
        "embedding_layernorm.weight": "embedding_layernorm.weight",
        "pre_lm_head_layernorm.weight": "hidden_states_layernorm.weight",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                self.config,
                i == 0,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
            ) for i in range(self.config.num_hidden_layers)
        ])
        
        # Projection layer: combines input embeddings with target hidden states
        self.projection_layer = torch.nn.Linear(self.config.hidden_size * 2,
                                               self.config.hidden_size,
                                               bias=False)
        
        # Support for additional layernorms (HASS variant)
        # HASS adds layernorms to input embeddings and hidden states for better
        # representation alignment between draft and target models
        self.has_embedding_layernorms = False
        if hasattr(self.config, "add_para_norm") and self.config.add_para_norm:
            self.embedding_layernorm = RMSNorm(self.config.hidden_size, 
                                             eps=self.config.rms_norm_eps)
            self.hidden_states_layernorm = RMSNorm(self.config.hidden_size,
                                                 eps=self.config.rms_norm_eps)
            self.has_embedding_layernorms = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Eagle draft model.
        
        Args:
            input_ids: Input token IDs for the draft model
            positions: Position indices for the tokens
            hidden_states: Hidden states from the target model at the same positions
        
        Returns:
            Tuple of (output_hidden_states, output_hidden_states) for compatibility
        """
        input_embeds = self.embed_tokens(input_ids)
        
        # Apply layernorms if enabled (HASS variant)
        # HASS normalizes both input embeddings and target hidden states
        # before combining them to improve alignment
        if self.has_embedding_layernorms:
            input_embeds = self.embedding_layernorm(input_embeds)
            hidden_states = self.hidden_states_layernorm(hidden_states)
        
        # Project concatenated embeddings and hidden states
        # This combines information from both the input tokens and target model
        hidden_states = self.projection_layer(
            torch.cat((input_embeds, hidden_states), dim=-1))
        
        # Process through transformer layers
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states

    def _remap_weight_name(self, name: str) -> str | None:
        """
        Remap speculators format weight names to vLLM names.
        
        Args:
            name: Original weight name from the checkpoint
        
        Returns:
            Remapped weight name, or None if the weight should be skipped
        """
        if name in self.SPECULATORS_WEIGHT_MAP:
            return self.SPECULATORS_WEIGHT_MAP[name]
        elif name.startswith("transformer."):
            # Skip transformer weights - they're loaded separately by the target model
            return None
        return name
    
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """
        Load model weights with support for speculators format.
        
        This method handles weight name mapping between speculators format
        and vLLM's expected naming convention, ensuring compatibility
        with both standard Eagle models and speculators-packaged models.
        
        Args:
            weights: Iterable of (weight_name, weight_tensor) pairs
        
        Returns:
            Set of parameter names that were successfully loaded
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            # Remap weight names for speculators compatibility
            remapped_name = self._remap_weight_name(name)
            if remapped_name is None:
                continue
            name = remapped_name
            
            # Handle stacked parameters (attention and MLP projections)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip embedding weights if pipeline parallelism is disabled
                # In this case, draft model shares embeddings with target model
                if get_pp_group().world_size == 1 and \
                    "embed_tokens." in name:
                    continue

                # Skip weights that don't exist in the model
                if name not in params_dict:
                    continue
                    
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EagleLlamaForCausalLM(LlamaForCausalLM):
    """
    Eagle draft model for causal language modeling.
    
    This class implements the Eagle draft model architecture for speculative
    decoding with Llama-based models. It consists of:
    1. A subset of transformer layers (starting after the target model layers)
    2. A projection layer that combines input embeddings with target hidden states
    3. Optional layernorms for HASS variant
    4. Logits processing for token generation
    
    The model generates draft tokens by processing the combination of input
    embeddings and hidden states from the target model, enabling faster
    speculative decoding.
    """
    
    # Weight name mapping for speculators format compatibility
    SPECULATORS_WEIGHT_MAP = {
        "fusion_fc.weight": "projection_layer.weight",
        "fusion_fc.bias": "projection_layer.bias",
        "embedding_layernorm.weight": "embedding_layernorm.weight",
        "pre_lm_head_layernorm.weight": "hidden_states_layernorm.weight",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix="model",
                                start_layer_id=target_layer_num)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Eagle draft model.
        
        Args:
            input_ids: Input token IDs for the draft model
            positions: Position indices for the tokens  
            hidden_states: Hidden states from the target model
        
        Returns:
            Tuple of (output_hidden_states, output_hidden_states) for compatibility
        """
        return self.model(input_ids, positions, hidden_states)

    def _remap_weight_name(self, name: str) -> str | None:
        """
        Remap speculators format weight names to vLLM names.
        
        Args:
            name: Original weight name from the checkpoint
        
        Returns:
            Remapped weight name, or None if the weight should be skipped
        """
        if name in self.SPECULATORS_WEIGHT_MAP:
            return self.SPECULATORS_WEIGHT_MAP[name]
        elif name.startswith("transformer."):
            # Skip transformer weights - they're loaded separately by the target model
            return None
        return name
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load model weights with support for speculators format.
        
        This method handles weight name mapping between speculators format
        and vLLM's expected naming convention.
        
        Args:
            weights: Iterable of (weight_name, weight_tensor) pairs
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
        )

        model_weights = {}
        for name, loaded_weight in weights:
            # Remap weight names for speculators compatibility
            remapped_name = self._remap_weight_name(name)
            if remapped_name is None:
                continue
            name = remapped_name
                
            # Add model prefix for non-lm_head weights
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight
        loader.load_weights(model_weights.items())
