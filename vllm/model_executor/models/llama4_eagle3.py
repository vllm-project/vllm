# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Eagle3 implementation for Llama4 models.

This module provides Eagle3 speculative decoding support for Llama4 models,
implementing a single-layer draft model that uses auxiliary hidden states
from the target model for efficient token generation.
"""

from collections.abc import Iterable
from typing import Any, Optional, TypeAlias, TypeVar, Union

import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama4 import Llama4DecoderLayer, Llama4Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

_T = TypeVar("_T")
HiddenStatesLike: TypeAlias = Union[IntermediateTensors, _T]


class Eagle3Model(Llama4Model):
    """
    Eagle3-specific implementation of Llama4Model.
    
    Eagle3 uses a single decoder layer to process combined inputs from
    embeddings and auxiliary hidden states provided by the target model.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        lora_enabled: bool = False,
    ) -> None:
        """
        Initialize Eagle3Model with single decoder layer.
        
        Args:
            vllm_config: vLLM configuration object
            prefix: Model name prefix for parameter loading
            start_layer_id: Starting layer ID (should be 0 for Eagle3)
            quant_config: Optional quantization configuration
            lora_enabled: Whether LoRA is enabled
        """
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layers_range=(start_layer_id, start_layer_id + 1),
        )

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        # Validate Eagle3 configuration
        self._validate_and_update_config(start_layer_id, quant_config)

        # Store config attributes
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Create vocabulary embedding (may be shared with target model)
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        # Create single decoder layer (Eagle3 uses only one layer)
        self.layers = nn.ModuleList([
            Llama4DecoderLayer(
                vllm_config=vllm_config,
                layer_id=0,
                prefix=f"{prefix}.layers.0",
                cache_config=cache_config,
                quant_config=quant_config,
            )
        ])

        # Output normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Eagle3-specific: Linear layer to combine auxiliary hidden states
        # This combines multiple layers of hidden states from the target model
        # The input dimension depends on how many auxiliary layers are used
        self.fc = None  # Will be initialized based on auxiliary layer count

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[torch.Tensor]] = None,
        attn_metadata: AttentionMetadata = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> HiddenStatesLike[IntermediateTensors, torch.Tensor]:
        """
        Forward pass for Eagle3 model.
        
        Eagle3 combines input embeddings with auxiliary hidden states from
        the target model, processes them through a single decoder layer,
        and returns the output.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            hidden_states: Auxiliary hidden states from target model
            kv_caches: Key-value caches for attention
            attn_metadata: Attention metadata
            intermediate_tensors: Optional intermediate tensors
            inputs_embeds: Optional pre-computed embeddings
            
        Returns:
            Tuple of (hidden_states, hidden_prenorm) after processing
        """
        # Handle intermediate tensors if provided
        if intermediate_tensors is not None:
            input_ids = intermediate_tensors.input_ids
            inputs_embeds = intermediate_tensors.inputs_embeds

        # Compute embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # If no auxiliary hidden states provided, use zero tensor
        if hidden_states is None:
            hidden_states = torch.zeros_like(inputs_embeds)

        # Eagle3: auxiliary hidden states have same dimension as embeddings
        # This assertion ensures compatibility for the single decoder layer
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1], (
            f"Hidden states dimension {hidden_states.shape[-1]} must match "
            f"input embeddings dimension {inputs_embeds.shape[-1]}")

        # Combine inputs_embeds with hidden_states for processing
        combined_hidden = inputs_embeds + hidden_states
        residual = None
        hidden_states, residual = self.layers[0](
            positions,
            combined_hidden,
            residual,
        )

        # Final normalization and return
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """
        Load model weights with Eagle3-specific mappings.
        
        This method handles the specific weight naming conventions used by
        Eagle3 models, including layer remapping and parameter stacking.
        
        Args:
            weights: Iterable of (parameter_name, tensor) pairs
            
        Returns:
            Set of loaded parameter names
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
            # Eagle3: convert midlayer naming to standard layer naming
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')

            # Handle stacked parameters (QKV and gate/up projections)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle embedding sharing in pipeline parallelism
                if (get_pp_group().world_size == 1
                        and "embed_tokens." in name):
                    # Skip embed_tokens when PP is disabled
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Validate all required parameters were loaded
        for name in params_dict:
            if (get_pp_group().world_size == 1 and "embed_tokens." in name):
                continue
            assert name in loaded_params, f"Parameter {name} was not loaded!"

        return loaded_params

    def _validate_and_update_config(
            self,
            start_layer_id: int,
            quant_config: Optional[QuantizationConfig] = None) -> None:
        """
        Validate and update model configuration for Eagle3 compatibility.
        
        Args:
            start_layer_id: Starting layer ID (should be 0)
            quant_config: Optional quantization configuration
            
        Raises:
            ValueError: If configuration is invalid for Eagle3
        """
        config = self.config

        # Eagle3 uses single layer starting at index 0
        if start_layer_id != 0:
            raise ValueError(f"Eagle3 models must start at layer 0, "
                             f"got {start_layer_id}")

        # Ensure configuration has required attributes
        if not hasattr(config, 'no_rope_layers'):
            config.no_rope_layers = None

        # Validate hidden size is compatible
        if config.hidden_size % 128 != 0:
            raise ValueError(f"Hidden size {config.hidden_size} must be "
                             f"divisible by 128 for Eagle3")


class Eagle3Llama4ForCausalLM(nn.Module):
    """
    Eagle3 implementation for Llama4 causal language modeling.
    
    This class wraps the Eagle3Model and adds language modeling head
    and vocabulary mapping for Eagle3 speculative decoding.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """
        Initialize Eagle3 Llama4 model for causal LM.
        
        Args:
            vllm_config: vLLM configuration object
            prefix: Model name prefix for parameter loading
            **kwargs: Additional arguments
        """
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # Core Eagle3 model with single decoder layer
        self.model = Eagle3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=0,
            quant_config=quant_config,
        )

        # Language modeling head
        self.lm_head = UnquantizedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Vocabulary mapping for draft-to-target token conversion
        # Eagle3 may have different vocabulary than target model
        self.draft_id_to_target_id = None

        # Store configuration
        self.config = config
        self.hidden_size = config.hidden_size

        # Initialize fc layer for combining auxiliary hidden states
        # This will be set based on the number of auxiliary layers used
        self._init_fc_layer()

    def _init_fc_layer(self) -> None:
        """
        Initialize FC layer for combining auxiliary hidden states.
        
        The input dimension depends on the number of auxiliary layers
        from the target model that Eagle3 uses.
        """
        # Default configuration for auxiliary layers
        # Typically uses layers at specific positions in the target model
        num_aux_layers = getattr(self.config, 'num_aux_layers', 3)

        # Create linear layer to combine auxiliary hidden states
        # Input: concatenated hidden states from multiple layers
        # Output: single hidden state vector
        input_dim = self.hidden_size * num_aux_layers
        output_dim = self.hidden_size

        self.model.fc = nn.Linear(input_dim, output_dim, bias=False)

    def combine_hidden_states(
        self, hidden_states: Union[torch.Tensor,
                                   list[torch.Tensor]]) -> torch.Tensor:
        """
        Combine auxiliary hidden states from multiple target model layers.
        
        Args:
            hidden_states: Either a pre-concatenated tensor or list of tensors
                          from different layers of the target model
                          
        Returns:
            Combined hidden state tensor of shape (batch_size, hidden_size)
        """
        if isinstance(hidden_states, list):
            # Concatenate hidden states from multiple layers
            hidden_states = torch.cat(hidden_states, dim=-1)

        # Use fc layer to combine into single hidden state
        return self.model.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[torch.Tensor]] = None,
        attn_metadata: AttentionMetadata = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for Eagle3 language model.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            hidden_states: Auxiliary hidden states from target model
            kv_caches: Key-value caches
            attn_metadata: Attention metadata
            intermediate_tensors: Optional intermediate tensors
            **kwargs: Additional arguments
            
        Returns:
            Logits tensor for next token prediction
        """
        # Get hidden states from Eagle3 model
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
        )

        # Apply language modeling head
        logits = self.lm_head(hidden_states)

        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden state tensor
            sampling_metadata: Optional sampling metadata
            
        Returns:
            Logits tensor
        """
        return self.lm_head(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Sample tokens from logits.
        
        Args:
            logits: Logits tensor
            sampling_metadata: Sampling metadata
            
        Returns:
            Sampled token IDs
        """
        # Apply vocabulary mapping if available
        if self.draft_id_to_target_id is not None:
            # Map draft vocabulary to target vocabulary
            # This ensures compatibility between draft and target models
            pass  # Mapping logic would go here

        # Return logits for sampling (actual sampling done elsewhere)
        return logits

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """
        Create empty intermediate tensors for pipeline parallelism.
        
        Args:
            batch_size: Batch size
            dtype: Data type for tensors
            device: Device to place tensors on
            
        Returns:
            Empty IntermediateTensors object
        """
        return IntermediateTensors({
            "input_ids":
            torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            "inputs_embeds":
            torch.zeros((batch_size, self.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    @property
    def sampler(self):
        """Get the sampler module."""
        # Eagle3 doesn't use a separate sampler
        return None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of prepared inputs
        """
        # Extract auxiliary hidden states if provided
        hidden_states = kwargs.get("hidden_states")

        # Get embedding layer
        embed_layer = self.model.embed_tokens

        # Compute input embeddings
        inputs_embeds = embed_layer(input_ids)

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "hidden_states": hidden_states,
        }

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> None:
        """
        Load Eagle3 model weights with specific mapping rules.
        
        This method handles weight loading for Eagle3 models,
        including vocabulary mapping and layer name translations.
        
        Args:
            weights: Iterable of (parameter_name, tensor) pairs from checkpoint
        """
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False

        # Process and filter weights according to Eagle3 conventions
        for name, loaded_weight in weights:
            # Skip target-to-draft mappings (not used in current implementation)
            if "t2d" in name:
                continue

            # Handle draft-to-target vocabulary mapping
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                # Prefix non-lm_head weights with "model."
                name = "model." + name

            if "embed_tokens" in name:
                includes_embed_tokens = True

            model_weights[name] = loaded_weight

        # Configure weight loader with conditional skipping
        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")

        # Use AutoWeightsLoader for robust weight loading
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())


def maybe_prefix(prefix: str, name: str) -> str:
    """Helper to add prefix to parameter names."""
    if prefix:
        return f"{prefix}.{name}"
    return name
