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

    Set `reasoning_start_str` and `reasoning_end_str` to the strings that delimit
    the reasoning block (e.g. `"<think>"` and `"</think>"`).  The
    corresponding token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String that indicates the end of reasoning content."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_end_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _parser_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `parser_reasoning_end_token_ids`. Set by
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
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_end_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_end_token_ids

    @property
    def parser_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs of the reasoning parser's intrinsic end-of-reasoning
        marker (e.g. `</think>` for Qwen3). May differ from
        `reasoning_end_token_ids` when the user supplies a longer custom
        `reasoning_end_str` that ends with the parser's natural marker.

        Consumers that need to know "has the parser already seen the end of
        reasoning?" should match on this sequence; consumers that need to
        *force* end-of-reasoning on budget exhaustion should still use the
        full `reasoning_end_token_ids`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._parser_reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        parser_end_str: str | None = None
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            parser_start_str = reasoning_parser.reasoning_start_str
            parser_end_str = reasoning_parser.reasoning_end_str
            if parser_start_str and not reasoning_start_str:
                reasoning_start_str = parser_start_str
            if parser_end_str and not reasoning_end_str:
                reasoning_end_str = parser_end_str

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )

        if not self._reasoning_start_token_ids or not self._reasoning_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )

        # If the reasoning parser advertises its own end-of-reasoning marker,
        # tokenize it and verify that the configured `reasoning_end_str`
        # contains it as a contiguous token subsequence. Without that,
        # the parser would never recognize the end of reasoning and on a
        # `thinking_token_budget` overrun
        if parser_end_str:
            parser_end_ids = tokenizer.encode(parser_end_str, add_special_tokens=False)
            if parser_end_ids:
                full = self._reasoning_end_token_ids
                sub = parser_end_ids
                contains = any(
                    full[i : i + len(sub)] == sub
                    for i in range(len(full) - len(sub) + 1)
                )
                if not contains:
                    raise ValueError(
                        f"ReasoningConfig: `reasoning_end_str` "
                        f"({reasoning_end_str!r}) does not contain the "
                        f"reasoning parser's intrinsic end marker "
                        f"({parser_end_str!r}) as a token subsequence. "
                        f"The parser would never recognize the configured "
                        f"end of reasoning on a `thinking_token_budget` "
                        f"overrun. Ensure `reasoning_end_str` ends "
                        f"with (or otherwise contains) {parser_end_str!r}."
                    )
                self._parser_reasoning_end_token_ids = parser_end_ids
        self._enabled = True
