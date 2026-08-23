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

    Set `reasoning_start_str` and `reasoning_end_str` to the strings used to
    enter and forcibly terminate reasoning. The end string may include a
    transition phrase before the parser's natural reasoning end marker. Token
    IDs are derived automatically by `initialize_token_ids`.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String forced when the thinking budget is exhausted."""

    loop_break_max_pattern_size: int = 0
    """Maximum N-gram pattern size for reasoning-scoped repetition detection.
    When > 0 (together with ``loop_break_min_count``), a request whose current
    reasoning section ends in a repeating token pattern is forced out of
    reasoning via ``reasoning_end_str`` — instead of looping until
    ``max_tokens`` or a ``thinking_token_budget`` — and then answers normally.
    Set to 0 (the default) to disable. Detection semantics match
    ``SamplingParams.repetition_detection``, which by contrast finishes the
    whole request."""

    loop_break_min_pattern_size: int = 0
    """Minimum N-gram pattern size for reasoning loop breaking. If 0, it
    defaults to 1. Must be <= ``loop_break_max_pattern_size``. Raising it does
    not exclude shorter cycles -- a cycle matches at every multiple of its own
    period -- so tune benign repeats out with ``loop_break_min_count``, which
    sets how many tokens must repeat before a break fires."""

    loop_break_min_count: int = 0
    """Number of consecutive repetitions of a pattern that triggers reasoning
    loop breaking. Must be >= 2 when enabled."""

    loop_break_min_reasoning_tokens: int = 256
    """Do not check a reasoning section for loops until it has generated at
    least this many tokens."""

    loop_break_check_interval: int = 16
    """Check for loops every this many newly accepted reasoning tokens."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for forced reasoning end token IDs."""
    _natural_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs that naturally terminate reasoning, as defined by the parser."""

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
        """Token IDs forced when the thinking budget is exhausted."""
        return self._reasoning_end_token_ids

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs that indicate the model naturally ended reasoning."""
        return self._natural_reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
            and self._natural_reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_reasoning_end_str = ""
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            natural_reasoning_end_str = end_token or ""

        if not natural_reasoning_end_str:
            natural_reasoning_end_str = reasoning_end_str

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
        self._natural_reasoning_end_token_ids = tokenizer.encode(
            natural_reasoning_end_str, add_special_tokens=False
        )

        if (
            not self._reasoning_start_token_ids
            or not self._reasoning_end_token_ids
            or not self._natural_reasoning_end_token_ids
        ):
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
