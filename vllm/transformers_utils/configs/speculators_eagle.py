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
        config_path = Path(pretrained_model_name_or_path) / "config.json"
        
        if not config_path.exists():
            # Fall back to standard loading if not a local path
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Check if this is a speculators format config
        if "speculators_model_type" not in config_dict:
            # Not a speculators config, use standard loading
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Convert speculators format to vLLM format
        vllm_config = cls._convert_speculators_to_vllm(config_dict)
        
        return cls(**vllm_config)
    
    @classmethod
    def _convert_speculators_to_vllm(cls, speculators_config: dict) -> dict:
        """
        Convert speculators Eagle config format to vLLM format.
        
        Speculators format:
        {
            "speculators_model_type": "eagle",
            "transformer_layer_config": {...},
            "layernorms": true/false,
            "fusion_bias": true/false
        }
        
        vLLM format:
        {
            "model_type": "eagle",
            "model": {...},
            "eagle_fc_bias": true/false,
            "truncated_vocab_size": vocab_size
        }
        """
        # Extract transformer config
        transformer_config = speculators_config.get("transformer_layer_config", {})
        
        # Handle layernorms flag
        if speculators_config.get("layernorms", False):
            transformer_config["add_para_norm"] = True
            # Ensure skip flags are set correctly for extra layernorms
            transformer_config["skip_prenorm"] = False
            transformer_config["skip_output_norm"] = False
        
        # Ensure transformer config has required fields
        if "architectures" not in transformer_config:
            # Infer from transformer_layer_architecture
            arch = speculators_config.get("transformer_layer_architecture", "LlamaDecoderLayer")
            if arch == "LlamaDecoderLayer":
                transformer_config["architectures"] = ["LlamaForCausalLM"]
            else:
                transformer_config["architectures"] = [arch]
        
        # Build vLLM config
        vllm_config = {
            "model_type": "eagle",
            "model": transformer_config,
            "eagle_fc_bias": speculators_config.get("fusion_bias", False),
            "truncated_vocab_size": transformer_config.get("vocab_size"),
        }
        
        # Preserve any additional fields that might be needed
        for key, value in speculators_config.items():
            if key not in ["speculators_model_type", "transformer_layer_config", 
                          "layernorms", "fusion_bias", "architectures"]:
                vllm_config[key] = value
        
        # Set architectures for vLLM
        vllm_config["architectures"] = ["EAGLEModel"]
        
        return vllm_config


def is_speculators_eagle_config(config_path: Union[str, os.PathLike]) -> bool:
    """
    Check if a config file is in speculators Eagle format.
    """
    config_file = Path(config_path) / "config.json"
    if not config_file.exists():
        return False
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config.get("speculators_model_type") == "eagle"
    except:
        return False