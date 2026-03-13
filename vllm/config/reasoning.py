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

    # NOTE: These parameters are temporary, the intent is to derive them
    # automatically from the reasoning parser in a future version.
    think_start_str: str = "<think>"
    """String that indicates the start of reasoning."""
    think_end_str: str = "</think>"
    """String that indicates the end of reasoning content."""

    _think_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `think_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _think_end_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    """Private backing field for `think_end_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    @property
    def think_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `think_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._think_start_token_ids

    @property
    def think_end_token_ids(self) -> list[int] | None:
        """Token IDs derived from `think_end_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._think_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._think_start_token_ids is not None
            and self._think_end_token_ids is not None
        ):
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)

        self._think_start_token_ids = tokenizer.encode(
            self.think_start_str, add_special_tokens=False
        )
        self._think_end_token_ids = tokenizer.encode(
            self.think_end_str, add_special_tokens=False
        )

        if not self._think_start_token_ids or not self._think_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"think_start_str='{self.think_start_str}', "
                f"think_end_str='{self.think_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
