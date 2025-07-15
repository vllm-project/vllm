# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.eagle import EAGLEConfig


class SpeculatorsEagleConfig(EAGLEConfig):
    """
    Adapter for speculators Eagle configs to make them compatible with vLLM.
    
    This class handles the conversion between speculators config format and
    vLLM's expected Eagle config format.
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "SpeculatorsEagleConfig":
        """
        Load a speculators Eagle config and convert it to vLLM format.
        """
        # Use the parent class method to load config dict
        # This handles both local paths and HuggingFace model IDs
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        
        # Check if this is a speculators format config
        speculators_type = config_dict.get("speculators_model_type")
        if speculators_type not in ["eagle", "eagle3"]:
            # Not a speculators config, use standard loading
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Convert speculators format to vLLM format
        vllm_config = cls._convert_speculators_to_vllm(config_dict)
        
        return cls(**vllm_config)
    
    @classmethod
    def _convert_speculators_to_vllm(cls, speculators_config: dict) -> dict:
        """
        Convert speculators Eagle config format to vLLM format.
        
        Supports both Eagle and Eagle-3 models based on speculators_model_type.
        """
        speculators_model_type = speculators_config.get("speculators_model_type")
        assert speculators_model_type, "`speculators_model_type` must be specified in the config"

        transformer_config = speculators_config.get("transformer_layer_config", {})
        
        # Extract num_lookahead_tokens from proposal_methods
        proposal_methods = speculators_config.get("speculators_config", {}).get("proposal_methods", [])
        assert proposal_methods, "speculators_config must have at least one proposal method"
        
        # Only one proposal method is supported for now
        proposal_method: dict = proposal_methods[0]
        num_lookahead_tokens = proposal_method.get("speculative_tokens")
        assert num_lookahead_tokens, "speculative_tokens must be specified in proposal_methods[0]"
        
        vllm_config = {
            "model": transformer_config,
            "method": speculators_model_type,
            "num_lookahead_tokens": num_lookahead_tokens,
        }
        
        if speculators_model_type == "eagle":
            # Eagle-1 specific handling
            # Handle layernorms flag
            if speculators_config.get("layernorms", False):
                transformer_config["add_para_norm"] = True
                # Ensure skip flags are set correctly for extra layernorms
                transformer_config["skip_prenorm"] = False
                transformer_config["skip_output_norm"] = False
            
            # Eagle-1 specific fields
            vllm_config["eagle_fc_bias"] = speculators_config.get("fusion_bias", False)
            vllm_config["truncated_vocab_size"] = transformer_config.get("vocab_size")
            vllm_config["architectures"] = ["EAGLEModel"]
            
        elif speculators_model_type == "eagle3":
            # Eagle-3 specific handling
            # Copy Eagle-3 specific fields from speculators config
            if speculators_config.get("draft_vocab_size") is not None:
                vllm_config["draft_vocab_size"] = speculators_config["draft_vocab_size"]
            
            # Handle target_hidden_size - if not provided, it should be set by vLLM
            # based on the target model, but we can try to infer from transformer config
            if speculators_config.get("target_hidden_size") is not None:
                vllm_config["target_hidden_size"] = speculators_config["target_hidden_size"]
            else:
                # Use the draft model's hidden size as target_hidden_size
                # This will be the same as the target model's hidden size
                vllm_config["target_hidden_size"] = transformer_config.get("hidden_size", 4096)
                
            if "norm_before_residual" in speculators_config:
                vllm_config["norm_before_residual"] = speculators_config["norm_before_residual"]
            
            # Eagle-3 uses different architecture
            vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
        
        # Ensure transformer config has required fields
        if "architectures" not in transformer_config:
            # Infer from transformer_layer_architecture
            arch = speculators_config.get("transformer_layer_architecture", "LlamaDecoderLayer")
            if arch == "LlamaDecoderLayer":
                transformer_config["architectures"] = ["LlamaForCausalLM"]
            else:
                transformer_config["architectures"] = [arch]
        
        speculators_specific_fields: set = {"speculators_model_type", "transformer_layer_config", 
                          "layernorms", "fusion_bias", "architectures",
                          "draft_vocab_size", "target_hidden_size", "norm_before_residual"}
        
        # Preserve any additional fields that might be needed
        for key, value in speculators_config.items():
            if key not in speculators_specific_fields:
                vllm_config[key] = value
        return vllm_config


def is_speculators_eagle_config(config_path: Union[str, os.PathLike]) -> bool:
    """
    Check if a config file is in speculators Eagle format.
    """
    supported_model_types = ["eagle", "eagle3"]
    config_dict, _ = PretrainedConfig.get_config_dict(config_path)
    return config_dict.get("speculators_model_type") in supported_model_types
