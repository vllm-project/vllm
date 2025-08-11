# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any, Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES)

__all__ = ["SpeculatorsConfig"]


class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "SpeculatorsConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path,
                                             **kwargs)

        speculators_model_type = config_dict.get("speculators_model_type")
        if speculators_model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Expected one of: {SUPPORTED_SPECULATORS_TYPES}. "
                "Please ensure you're loading a speculators-format model.")

        # validate fields
        # TODO: @dsikka - use speculators pydantic model to validate
        cls.validate_speculators_config(config_dict=config_dict)
        # Convert from speculators config -> format that can be ingested by vLLM
        vllm_config = cls.convert_speculators_to_vllm(config_dict=config_dict)
        # Apply anything specific to the supported algorithm
        algo_updater = SUPPORTED_SPECULATORS_TYPES[speculators_model_type]
        algo_updater(config_dict=config_dict, vllm_config=vllm_config)
        return cls(**vllm_config)

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
                "'transformer_layer_config' must be a dictionary if provided")

    @classmethod
    def convert_speculators_to_vllm(
            cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert speculators config format to vLLM format.
        
        This method handles the translation of field names and structure
        between speculators and vLLM formats.
        
        Returns:
            Dictionary with vLLM-compatible configuration
        """
        # Currently we only support one proposal method
        spec_config = config_dict["speculators_config"]
        first_method = spec_config.get("proposal_methods")[0]
        num_lookahead_tokens = first_method.get("speculative_tokens")

        if num_lookahead_tokens is None:
            raise ValueError(
                "Missing 'speculative_tokens' in proposal method. "
                f"Got: {first_method}")

        # Build base vLLM config
        vllm_config = {
            "method": config_dict.get("speculators_model_type"),
            "num_lookahead_tokens": num_lookahead_tokens,
            "target_model": spec_config.get("verifier")["name_or_path"]
        }
        vllm_config.update(config_dict["transformer_layer_config"])
        return vllm_config
