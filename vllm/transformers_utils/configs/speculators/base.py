# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import fields
from typing import Any

from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
)
from vllm.transformers_utils.utils import without_trust_remote_code

logger = init_logger(__name__)


class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    def __init__(self, **kwargs):
        # super().__init__ performs some validation before setting all kwargs as
        # attributes, so we set them first to be safe
        pre_trained_config_fields = {f.name for f in fields(PretrainedConfig)}
        super_kwargs = dict()
        for key, value in kwargs.items():
            if key == "model_type":
                continue  # model_type is set as a class variable, so skip it here
            elif key in pre_trained_config_fields:
                super_kwargs[key] = value
            else:
                setattr(self, key, value)
        super().__init__(**super_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "SpeculatorsConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(
            pretrained_model_name_or_path, **without_trust_remote_code(kwargs)
        )

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
        spec_config = config_dict.get("speculators_config")
        if not isinstance(spec_config, dict):
            raise ValueError("Invalid speculators config structure")
        methods = spec_config.get("proposal_methods")
        if not isinstance(methods, list) or not methods:
            raise ValueError(
                "speculators_config.proposal_methods must be a non-empty list"
            )
        # The schema is plural: every entry must at least be a mapping that
        # names its proposal_type. Fields specific to concrete method types
        # (e.g. speculative_tokens) are checked only on the selected method.
        for method in methods:
            if not isinstance(method, dict) or "proposal_type" not in method:
                raise ValueError(
                    "Each entry in proposal_methods must be a mapping with a "
                    f"'proposal_type'. Got: {method}"
                )
        try:
            _ = spec_config["verifier"]["name_or_path"]
            _ = config_dict["speculators_model_type"]
        except (KeyError, TypeError) as e:
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

        proposal_methods = spec_config.get("proposal_methods")
        if not proposal_methods:
            raise ValueError("No proposal methods found in speculators config")

        # The schema names the active method via default_proposal_method;
        # select it, falling back to the first entry for configs that
        # predate the field. vLLM drafts with a single method.
        default_method = spec_config.get("default_proposal_method")
        selected = next(
            (
                method
                for method in proposal_methods
                if method.get("proposal_type") == default_method
            ),
            proposal_methods[0],
        )
        if default_method is not None and selected.get("proposal_type") != (
            default_method
        ):
            logger.warning_once(
                "default_proposal_method '%s' matches no entry in "
                "proposal_methods; falling back to the first entry '%s'.",
                default_method,
                str(selected.get("proposal_type", "<unnamed>")),
            )
        if len(proposal_methods) > 1:
            ignored = ", ".join(
                str(method.get("proposal_type", "<unnamed>"))
                for method in proposal_methods
                if method is not selected
            )
            logger.warning_once(
                "Speculators config declares %d proposal methods; vLLM "
                "drafts with one. Selected '%s'; ignoring: %s.",
                len(proposal_methods),
                str(selected.get("proposal_type", "<unnamed>")),
                ignored,
            )

        num_speculative_tokens = selected.get("speculative_tokens")

        if num_speculative_tokens is None:
            raise ValueError(
                "Missing 'speculative_tokens' in the selected proposal "
                f"method. Got: {selected}"
            )

        # Build base vLLM speculative configuration
        result = {
            "method": config_dict.get("speculators_model_type"),
            "num_speculative_tokens": num_speculative_tokens,
        }
        if result["method"] == "peagle":
            result.update({"method": "eagle3", "parallel_drafting": True})
        elif result["method"] == "dflash2":
            result["method"] = "dflash"
        return result
