# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig:
    """Configuration for reasoning models."""

    think_start_str: str | None = None
    """String that indicates the start of reasoning."""
    think_end_str: str | None = None
    """String that indicates the end of reasoning."""
    think_start_token_ids: list[int] | None = None
    """Token ID that indicates the start of reasoning."""
    think_end_token_ids: list[int] | None = None
    """Token ID that indicates the end of reasoning."""

    @property
    def is_thinking_enabled(self) -> bool:
        """Check if both start and end thinking token IDs
        are set to enable thinking token budget logic."""
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
