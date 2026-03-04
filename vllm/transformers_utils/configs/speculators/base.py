# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
)

__all__ = ["SpeculatorsConfig"]


class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "SpeculatorsConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        vllm_config = cls.extract_transformers_pre_trained_config(config_dict)
        return cls(**vllm_config)

    @classmethod
    def extract_transformers_pre_trained_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract standard Transformers PreTrainedConfig config from speculators config.
        """
        speculators_model_type = config_dict.get("speculators_model_type")
        if speculators_model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Expected one of: {SUPPORTED_SPECULATORS_TYPES}. "
                "Please ensure you're loading a speculators-format model."
            )

        # Start with transformer layer configuration if present
        pre_trained_config = config_dict.get("transformer_layer_config", {})
        # Apply anything specific to the supported algorithm
        algo_updater = SUPPORTED_SPECULATORS_TYPES[speculators_model_type]
        algo_updater(config_dict=config_dict, pre_trained_config=pre_trained_config)
        return pre_trained_config

    @classmethod
    def extract_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract vLLM speculative config from speculators config."""
        # validate fields
        # TODO: @dsikka - use speculators pydantic model to validate
        cls.validate_speculators_config(config_dict=config_dict)
        # Convert from speculators config -> format that can be ingested by vLLM
        return cls.build_vllm_speculative_config(config_dict=config_dict)

    @classmethod
    def validate_speculators_config(cls, config_dict: dict[str, Any]) -> None:
        try:
            spec_config = config_dict["speculators_config"]
            methods = spec_config["proposal_methods"]
            first_method = methods[0]
            _ = first_method["speculative_tokens"]
            _ = spec_config["verifier"]["name_or_path"]
            _ = config_dict["speculators_model_type"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("Invalid speculators config structure") from e

        if "transformer_layer_config" not in config_dict:
            raise ValueError("Must provide transformer_layer_config")

        if not isinstance(config_dict["transformer_layer_config"], dict):
            raise TypeError(
                "'transformer_layer_config' must be a dictionary if provided"
            )

    @classmethod
    def build_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build vLLM-compatible speculative configuration from speculators format.

        This method extracts and transforms speculative configuration from the
        speculators format into the structure expected by vLLM.

        Args:
            config_dict: Configuration dictionary in speculators format

        Returns:
            Dictionary with vLLM-compatible speculative configuration
        """
        # Extract speculators configuration
        spec_config = config_dict["speculators_config"]

        # Currently we only support one proposal method
        proposal_methods = spec_config.get("proposal_methods")
        if not proposal_methods:
            raise ValueError("No proposal methods found in speculators config")

        first_method = proposal_methods[0]
        num_speculative_tokens = first_method.get("speculative_tokens")

        if num_speculative_tokens is None:
            raise ValueError(
                f"Missing 'speculative_tokens' in proposal method. Got: {first_method}"
            )

        # Build base vLLM speculative configuration
        return {
            "method": config_dict.get("speculators_model_type"),
            "num_speculative_tokens": num_speculative_tokens,
        }
