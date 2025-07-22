# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any, Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.speculators import (
    Eagle3SpeculatorsConfig, EagleSpeculatorsConfig)

DEFAULT_NUM_LOOKAHEAD_TOKENS = 5
SPECULATORS_WEIGHT_MAP = {
    "fusion_fc.weight": "fc.weight",
    "fusion_fc.bias": "fc.bias",
    "embedding_layernorm.weight": "embedding_layernorm.weight",
    "pre_lm_head_layernorm.weight": "hidden_states_layernorm.weight",
}

SUPPORTED_SPECULATORS_TYPES = {
    "eagle": EagleSpeculatorsConfig,
    "eagle3": Eagle3SpeculatorsConfig
}


class SpeculatorsConfig(PretrainedConfig):

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "SpeculatorsConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path,
                                             **kwargs)

        speculators_type = config_dict.get("speculators_model_type")
        if speculators_type not in SUPPORTED_SPECULATORS_TYPES:
            return super().from_pretrained(pretrained_model_name_or_path,
                                           **kwargs)

        spec_class = SUPPORTED_SPECULATORS_TYPES.get(speculators_type)
        spec_class_instance = spec_class(config_dict)

        # Validate
        spec_class_instance.validate_speculators_config()
        num_lookahead_tokens = spec_class_instance.extract_num_lookahead_tokens(  # noqa: E501
        )
        vllm_config, transformer_config = spec_class_instance.convert_speculators_to_vllm(  # noqa: E501
            num_lookahead_tokens)

        # Process / Update
        spec_class_instance.update_defaults(transformer_config, vllm_config)
        spec_class_instance.ensure_transformer_architectures(
            transformer_config)  # Is this needed?
        vllm_config = spec_class_instance.preserve_additional_fields(
            vllm_config)

        # Create
        return cls(**vllm_config)

    def extract_num_lookahead_tokens(self) -> int:
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
        speculators_cfg = self.config["speculators_config"]
        proposal_methods = speculators_cfg["proposal_methods"]

        # Currently we only support one proposal method
        first_method = proposal_methods[0]
        num_lookahead_tokens = first_method.get("speculative_tokens")

        if num_lookahead_tokens is None:
            raise ValueError(
                "Missing 'speculative_tokens' in proposal method. "
                f"Got: {first_method}")

        return num_lookahead_tokens

    def validate_speculators_config(self) -> None:
        """Validate required speculators format fields."""
        # Check required top-level fields
        if "speculators_model_type" not in self.config_dict:
            raise ValueError(
                "Missing 'speculators_model_type' in config. "
                f"Expected one of: {sorted(SUPPORTED_SPECULATORS_TYPES)}. "
                "Please ensure you're loading a speculators-format Eagle model."
            )

        model_type = self.config["speculators_model_type"]
        if model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Unsupported speculators_model_type: '{model_type}'. "
                f"Supported types: {sorted(SUPPORTED_SPECULATORS_TYPES)}")

        # Check transformer config
        if "transformer_layer_config" not in self.config_dict:
            raise ValueError(
                "Missing 'transformer_layer_config' in speculators config. "
                "This field should contain the transformer architecture "
                "configuration.")

        # Check proposal methods
        speculators_cfg = self.config.get("speculators_config", {})
        if not isinstance(speculators_cfg, dict):
            raise ValueError("'speculators_config' must be a dictionary. "
                             f"Got: {type(speculators_cfg).__name__}")

        proposal_methods = speculators_cfg.get("proposal_methods", [])
        if not proposal_methods:
            raise ValueError(
                "No proposal methods found in speculators_config. "
                "Expected: {'speculators_config': {'proposal_methods': "
                "[{'speculative_tokens': N}]}}. "
                "Check that your model config follows the speculators format.")

    def convert_speculators_to_vllm(
            self, num_lookahead_tokens: int) -> dict[str, Any]:
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
        speculators_model_type = self.config_dict["speculators_model_type"]
        transformer_config = self.config_dict["transformer_layer_config"]

        # Build base vLLM config
        vllm_config = {
            "model": transformer_config,
            "method":
            speculators_model_type,  # Use speculators_model_type as method
            "num_lookahead_tokens": num_lookahead_tokens,
        }
        return vllm_config, transformer_config

    def ensure_transformer_architectures(
            self, transformer_config: dict[str, Any]) -> None:
        """Ensure transformer config has required architecture field."""
        if "architectures" not in transformer_config:
            default_arch = "LlamaDecoderLayer"
            arch = self.config.get("transformer_layer_architecture",
                                   default_arch)
            if arch == "LlamaDecoderLayer":
                transformer_config["architectures"] = ["LlamaForCausalLM"]
            else:
                transformer_config["architectures"] = [arch]
        return transformer_config

    def preserve_additional_fields(self, vllm_config: dict[str, Any]) -> None:
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

        for key, value in self.config.items():
            if key not in handled_fields:
                vllm_config[key] = value
        return vllm_config
