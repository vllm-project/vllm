# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.transformers_utils.configs.speculators import SpeculatorsConfig


class EagleSpeculatorsConfig(SpeculatorsConfig):

    def update_defualts(self, transformer_config: dict[str, Any],
                        vllm_config: dict[str, Any]) -> None:
        """
        Apply Eagle-1 specific configuration transformations.
        
        Eagle-1 specific fields:
        - fusion_bias → eagle_fc_bias
        - layernorms → add_para_norm (for HASH variant)
        - Uses truncated_vocab_size
        """
        # Handle HASH variant with additional layernorms
        if self.config.get("layernorms", False):
            transformer_config["add_para_norm"] = True
            # When using extra layernorms, ensure skip flags are set correctly
            # to maintain the expected architecture behavior
            transformer_config["skip_prenorm"] = False
            transformer_config["skip_output_norm"] = False

        if self.config.get("fusion_bias", False):
            # If fusion_bias is set, add it to the transformer config
            transformer_config["fusion_bias"] = True

        # Map Eagle-1 specific fields
        vocab_size = transformer_config.get("vocab_size")
        vllm_config["truncated_vocab_size"] = vocab_size
        vllm_config["architectures"] = ["EAGLEModel"]
