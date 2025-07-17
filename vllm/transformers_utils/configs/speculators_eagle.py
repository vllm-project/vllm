# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Optional, Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.eagle import EAGLEConfig

# Constants for speculators format
SUPPORTED_SPECULATORS_TYPES = frozenset({"eagle", "eagle3"})
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_NUM_LOOKAHEAD_TOKENS = 5


class SpeculatorsEagleConfig(EAGLEConfig):
    """Configuration adapter for speculators Eagle models.
    
    Translates between speculators library format and vLLM's Eagle format.
    Supports both Eagle-1 and Eagle-3 variants.
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "SpeculatorsEagleConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        
        speculators_type = config_dict.get("speculators_model_type")
        if speculators_type not in SUPPORTED_SPECULATORS_TYPES:
            return super().from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        
        cls._validate_speculators_config(config_dict)
        vllm_config = cls._convert_speculators_to_vllm(config_dict)
        
        return cls(**vllm_config)
    
    @classmethod
    def _validate_speculators_config(cls, config: dict[str, Any]) -> None:
        """Validate required speculators format fields."""
        # Check required top-level fields
        if "speculators_model_type" not in config:
            raise ValueError(
                "Missing 'speculators_model_type' in config. "
                f"Expected one of: {sorted(SUPPORTED_SPECULATORS_TYPES)}. "
                "Please ensure you're loading a speculators-format Eagle model."
            )
        
        model_type = config["speculators_model_type"]
        if model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Unsupported speculators_model_type: '{model_type}'. "
                f"Supported types: {sorted(SUPPORTED_SPECULATORS_TYPES)}"
            )
        
        # Check transformer config
        if "transformer_layer_config" not in config:
            raise ValueError(
                "Missing 'transformer_layer_config' in speculators config. "
                "This field should contain the transformer architecture configuration."
            )
        
        # Check proposal methods
        speculators_cfg = config.get("speculators_config", {})
        if not isinstance(speculators_cfg, dict):
            raise ValueError(
                "'speculators_config' must be a dictionary. "
                f"Got: {type(speculators_cfg).__name__}"
            )
        
        proposal_methods = speculators_cfg.get("proposal_methods", [])
        if not proposal_methods:
            raise ValueError(
                "No proposal methods found in speculators_config. "
                "Expected: {'speculators_config': {'proposal_methods': "
                "[{'speculative_tokens': N}]}}. "
                "Check that your model config follows the speculators format."
            )
    
    @classmethod
    def _convert_speculators_to_vllm(cls, speculators_config: dict[str, Any]) -> dict[str, Any]:
        """
        Convert speculators Eagle config format to vLLM format.
        
        This method handles the translation of field names and structure
        between speculators and vLLM formats. It supports both Eagle-1
        and Eagle-3 variants based on speculators_model_type.
        
        Args:
            speculators_config: Dictionary containing speculators format config
        
        Returns:
            Dictionary with vLLM-compatible Eagle configuration
        """
        speculators_model_type = speculators_config["speculators_model_type"]
        transformer_config = speculators_config["transformer_layer_config"]
        
        # Extract num_lookahead_tokens from proposal_methods
        num_lookahead_tokens = cls._extract_num_lookahead_tokens(speculators_config)
        
        # Build base vLLM config
        vllm_config = {
            "model": transformer_config,
            "method": speculators_model_type,  # Use speculators_model_type as method
            "num_lookahead_tokens": num_lookahead_tokens,
        }
        
        # Apply version-specific conversions
        if speculators_model_type == "eagle":
            cls._apply_eagle_v1_config(speculators_config, transformer_config, vllm_config)
        elif speculators_model_type == "eagle3":
            cls._apply_eagle_v3_config(speculators_config, transformer_config, vllm_config)
        
        # Ensure transformer config has required fields
        cls._ensure_transformer_architectures(speculators_config, transformer_config)
        
        # Preserve additional fields not handled by specific conversions
        cls._preserve_additional_fields(speculators_config, vllm_config)
        
        return vllm_config
    
    @classmethod
    def _extract_num_lookahead_tokens(cls, config: dict[str, Any]) -> int:
        """
        Extract number of lookahead tokens from proposal methods.
        
        Args:
            config: Speculators config dictionary
        
        Returns:
            Number of speculative tokens
        
        Note:
            Currently only supports the first proposal method.
            Future versions may support multiple proposal methods.
        """
        speculators_cfg = config["speculators_config"]
        proposal_methods = speculators_cfg["proposal_methods"]
        
        # Currently we only support one proposal method
        first_method = proposal_methods[0]
        num_lookahead_tokens = first_method.get("speculative_tokens")
        
        if num_lookahead_tokens is None:
            raise ValueError(
                "Missing 'speculative_tokens' in proposal method. "
                f"Got: {first_method}"
            )
        
        return num_lookahead_tokens
    
    @classmethod
    def _apply_eagle_v1_config(
        cls,
        speculators_config: dict[str, Any],
        transformer_config: dict[str, Any],
        vllm_config: dict[str, Any]
    ) -> None:
        """
        Apply Eagle-1 specific configuration transformations.
        
        Eagle-1 specific fields:
        - fusion_bias → eagle_fc_bias
        - layernorms → add_para_norm (for HASS variant)
        - Uses truncated_vocab_size
        """
        # Handle HASS variant with additional layernorms
        if speculators_config.get("layernorms", False):
            transformer_config["add_para_norm"] = True
            # When using extra layernorms, ensure skip flags are set correctly
            # to maintain the expected architecture behavior
            transformer_config["skip_prenorm"] = False
            transformer_config["skip_output_norm"] = False
        
        if speculators_config.get("fusion_bias", False):
            # If fusion_bias is set, add it to the transformer config
            transformer_config["fusion_bias"] = True
        

        
        # Map Eagle-1 specific fields
        vocab_size = transformer_config.get("vocab_size")
        vllm_config["truncated_vocab_size"] = vocab_size
        vllm_config["architectures"] = ["EAGLEModel"]
    
    @classmethod
    def _apply_eagle_v3_config(
        cls,
        speculators_config: dict[str, Any],
        transformer_config: dict[str, Any],
        vllm_config: dict[str, Any]
    ) -> None:
        """
        Apply Eagle-3 specific configuration transformations.
        
        Eagle-3 specific fields:
        - draft_vocab_size: Size of the draft model's vocabulary
        - target_hidden_size: Hidden size of the target model
        - norm_before_residual: Whether to apply norm before residual connection
        """
        # Copy Eagle-3 specific fields
        if speculators_config.get("draft_vocab_size") is not None:
            draft_vocab_size = speculators_config["draft_vocab_size"]
            vllm_config["draft_vocab_size"] = draft_vocab_size
        
        # Handle target_hidden_size
        if speculators_config.get("target_hidden_size") is not None:
            target_hidden_size = speculators_config["target_hidden_size"]
            vllm_config["target_hidden_size"] = target_hidden_size
        else:
            # Default to the draft model's hidden size
            # In practice, this should match the target model's hidden size
            vllm_config["target_hidden_size"] = transformer_config.get(
                "hidden_size", DEFAULT_HIDDEN_SIZE
            )
        
        if "norm_before_residual" in speculators_config:
            # Add to transformer config which becomes the model config
            transformer_config["norm_before_residual"] = speculators_config["norm_before_residual"]
        
        # Eagle-3 uses a different architecture
        vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
    
    @classmethod
    def _ensure_transformer_architectures(
        cls,
        speculators_config: dict[str, Any],
        transformer_config: dict[str, Any]
    ) -> None:
        """Ensure transformer config has required architecture field."""
        if "architectures" not in transformer_config:
            default_arch = "LlamaDecoderLayer"
            arch = speculators_config.get(
                "transformer_layer_architecture", default_arch
            )
            if arch == "LlamaDecoderLayer":
                transformer_config["architectures"] = ["LlamaForCausalLM"]
            else:
                transformer_config["architectures"] = [arch]
    
    @classmethod
    def _preserve_additional_fields(
        cls,
        speculators_config: dict[str, Any],
        vllm_config: dict[str, Any]
    ) -> None:
        """Preserve additional fields for forward compatibility."""
        handled_fields = {
            "speculators_model_type",
            "transformer_layer_config",
            "speculators_config",
            "layernorms",
            "fusion_bias",
            "architectures",
            "draft_vocab_size",
            "target_hidden_size",
            "norm_before_residual",
        }
        
        for key, value in speculators_config.items():
            if key not in handled_fields:
                vllm_config[key] = value


def is_speculators_eagle_config(config_path: Union[str, os.PathLike]) -> bool:
    """Check if a config file is in speculators Eagle format."""
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(config_path)
        
        if "speculators_model_type" not in config_dict:
            return False
        
        model_type = config_dict.get("speculators_model_type")
        return model_type in SUPPORTED_SPECULATORS_TYPES
    except Exception:
        return False


def extract_speculators_info(model_path: Union[str, os.PathLike]) -> Optional[dict[str, Any]]:
    """
    Extract target model and config from speculators format model.
    
    Returns dict with:
    - target_model: str - The target model name/path
    - method: str - The speculative method (eagle/eagle3)
    - num_tokens: int - Number of speculative tokens
    
    Returns None if not speculators format or missing target model.
    """
    try:
        # Check if it's speculators format
        if not is_speculators_eagle_config(model_path):
            return None
            
        # Load the config
        config_dict, _ = PretrainedConfig.get_config_dict(model_path)
        
        # Extract method
        method = config_dict.get("speculators_model_type", "eagle")
        
        # Extract num tokens
        num_tokens = DEFAULT_NUM_LOOKAHEAD_TOKENS  # default
        speculators_cfg = config_dict.get("speculators_config", {})
        proposal_methods = speculators_cfg.get("proposal_methods", [])
        if proposal_methods:
            num_tokens = proposal_methods[0].get("speculative_tokens", DEFAULT_NUM_LOOKAHEAD_TOKENS)
        
        # Extract target model - try multiple possible locations
        target_model = None
        
        # Try target_config.model_name (original format)
        target_config = speculators_cfg.get("target_config", {})
        target_model = target_config.get("model_name")
        
        # Try verifier.name_or_path (new format)
        if not target_model:
            verifier_config = speculators_cfg.get("verifier", {})
            target_model = verifier_config.get("name_or_path")
        
        # If no target model in config, return None
        # This will require user to specify target model explicitly
        if not target_model:
            return None
            
        return {
            "target_model": target_model,
            "method": method,
            "num_tokens": num_tokens
        }
    except Exception:
        # If any error occurs, treat as not speculators format
        return None
