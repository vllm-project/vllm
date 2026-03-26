# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `think_start_str` and `think_end_str` to the strings that delimit
    the reasoning block (e.g. `"<think>"` and `"</think>"`).  The
    corresponding token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    think_start_str: str = ""
    """String that indicates the start of reasoning."""
    think_end_str: str = ""
    """String that indicates the end of reasoning content."""

    _think_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `think_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _think_end_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    """Private backing field for `think_end_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

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

    def initialize_token_ids(self, model_config: ModelConfig) -> bool:
        """Initialize reasoning token IDs from strings using the tokenizer.

        Returns True if initialization was successful, False otherwise."""
        if (
            self._think_start_token_ids is not None
            and self._think_end_token_ids is not None
        ):
            self._enabled = True
            return True  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        think_start_str = self.think_start_str
        think_end_str = self.think_end_str
        if self.reasoning_parser and (not think_start_str or not think_end_str):
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not think_start_str:
                think_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not think_end_str:
                think_end_str = end_token

        if not think_start_str or not think_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return False
        self._think_start_token_ids = tokenizer.encode(
            think_start_str, add_special_tokens=False
        )
        self._think_end_token_ids = tokenizer.encode(
            think_end_str, add_special_tokens=False
        )

        if not self._think_start_token_ids or not self._think_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"think_start_str='{think_start_str}', "
                f"think_end_str='{think_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
        return True
