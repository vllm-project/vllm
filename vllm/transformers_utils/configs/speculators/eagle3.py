# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


class Eagle3SpeculatorsConfig(SpeculatorsConfig):

    def update_defaults(self, vllm_config: dict[str, Any]) -> None:
        """
        Apply Eagle-3 specific configuration transformations.
        
        Eagle-3 specific fields:
        - draft_vocab_size: Size of the draft model's vocabulary
        - target_hidden_size: Hidden size of the target model
        - norm_before_residual: Whether to apply norm before residual connection
        """
        # Copy Eagle-3 specific fields
        if self.config.get("draft_vocab_size") is not None:
            draft_vocab_size = self.config["draft_vocab_size"]
            vllm_config["draft_vocab_size"] = draft_vocab_size

        # Handle target_hidden_size
        if self.config.get("target_hidden_size") is not None:
            target_hidden_size = self.config["target_hidden_size"]
            vllm_config["target_hidden_size"] = target_hidden_size
        else:
            # Default to the draft model's hidden size
            # In practice, this should match the target model's hidden size
            vllm_config["target_hidden_size"] = vllm_config["model"].get(
                "hidden_size")

        if "norm_before_residual" in self.config:
            # Add to transformer config which becomes the model config
            vllm_config["model"]["norm_before_residual"] = self.config[
                "norm_before_residual"]

        # Eagle-3 uses a different architecture
        vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
