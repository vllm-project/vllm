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

    think_start_str: str | None = None
    """String that indicates the start of reasoning."""
    think_end_str: str | None = None
    """String that indicates the end of reasoning content."""

    think_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs derived from `think_start_str`. Set automatically by
    `initialize_token_ids`. Not intended to be configured directly."""
    think_end_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    """Token IDs derived from `think_end_str`. Set automatically by
    `initialize_token_ids`. Not intended to be configured directly."""

    @property
    def is_thinking_enabled(self) -> bool:
        """Check if thinking boundaries are configured."""
        return (
            self.think_start_token_ids is not None
            and self.think_end_token_ids is not None
            and len(self.think_start_token_ids) > 0
            and len(self.think_end_token_ids) > 0
        )

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if self.think_start_str is not None and self.think_end_str is not None:
            tokenizer = cached_tokenizer_from_config(model_config=model_config)

            # Convert reasoning strings to token IDs
            self.think_start_token_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(self.think_start_str)
            )
            self.think_end_token_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(self.think_end_str)
            )
