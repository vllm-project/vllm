# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `think_start_str` and `think_end_str` to the strings that delimit
    the reasoning block (e.g. `"<think>"` and `"</think>"`).  The
    corresponding token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    think_start_str: str = "<think>"
    """String that indicates the start of reasoning."""
    think_end_str: str = "</think>"
    """String that indicates the end of reasoning content."""

    think_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs derived from `think_start_str`. Set automatically by
    `initialize_token_ids`. Not intended to be configured directly."""
    think_end_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    """Token IDs derived from `think_end_str`. Set automatically by
    `initialize_token_ids`. Not intended to be configured directly."""

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        tokenizer = cached_tokenizer_from_config(model_config=model_config)

        # Convert reasoning strings to token IDs
        self.think_start_token_ids = tokenizer.encode(
            self.think_start_str, add_special_tokens=False
        )
        self.think_end_token_ids = tokenizer.encode(
            self.think_end_str, add_special_tokens=False
        )

        if not self.think_start_token_ids or not self.think_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"think_start_str='{self.think_start_str}', "
                f"think_end_str='{self.think_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
