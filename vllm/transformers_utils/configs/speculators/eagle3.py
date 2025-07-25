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
        # The way we store hidden size and vocab size is confusing in out config
        # we store taarget_hidden_size and hidden_size

        # Copy Eagle-3 specific fields
        draft_vocab_size = self.config.get("draft_vocab_size", None)
        if draft_vocab_size is not None:
            vllm_config["draft_vocab_size"] = draft_vocab_size

        # Handle target_hidden_size - if different than the draft hidden size
        if self.config.get("target_hidden_size") is not None:
            vllm_config["target_hidden_size"] = self.config[
                "target_hidden_size"]

        # Norm before residual
        vllm_config["norm_before_residual"] = self.config.get(
            "norm_before_residual", True)

        # Eagle-3 uses a different architecture
        vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
