# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from mistral_common.tokens.tokenizers.base import SpecialTokens

from vllm.reasoning import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers.mistral import MistralTokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MistralReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Mistral models.

    The Mistral models uses `[THINK]`...`[/THINK]` tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.

    A valid reasoning trace should always start with a `[THINK]` token and end with
    a `[/THINK]` token.

    If `[THINK]` token is not generated, then this parser only returns content.
    """

    def __init__(self, tokenizer: MistralTokenizer, *args, **kwargs):
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("The tokenizer must be an instance of MistralTokenizer.")

        ReasoningParser.__init__(self, tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.start_token_id = tokenizer.tokenizer.get_special_token(self.start_token)
        self.end_token_id = tokenizer.tokenizer.get_special_token(self.end_token)

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "Mistral reasoning parser could not locate think start/end "
                "tokens in the tokenizer!"
            )

        self._turn_boundary_token_ids: frozenset[int] = (
            self._resolve_turn_boundary_token_ids(tokenizer)
        )

    def _resolve_turn_boundary_token_ids(
        self, tokenizer: MistralTokenizer
    ) -> frozenset[int]:
        """Resolve the token ids that close a turn preceding the current one.

        Real `[THINK]`/`[/THINK]` control tokens can only enter the prompt from
        a system message think chunk, a tool message think chunk, or a prior
        assistant turn.

        Args:
            tokenizer: The Mistral tokenizer used to resolve special tokens.

        Returns:
            The resolved boundary token ids, excluding the think tokens.
        """

        boundary_names = (
            SpecialTokens.end_system,
            SpecialTokens.end_tool_results,
            SpecialTokens.eos,
        )
        boundary_token_ids: set[int] = set()
        for name in boundary_names:
            try:
                boundary_token_ids.add(tokenizer.tokenizer.get_special_token(name))
            except ValueError:
                # Tokenizer version without this control token; skip it.
                continue
        return frozenset(boundary_token_ids)

    @cached_property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return SpecialTokens.begin_think

    @cached_property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return SpecialTokens.end_think

    def _current_turn_start_index(self, input_ids: Sequence[int]) -> int:
        """Return the index of the first token of the current assistant turn.

        Args:
            input_ids: The full token sequence to inspect.

        Returns:
            The start index of the current turn, or 0 if no boundary is found.
        """
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in self._turn_boundary_token_ids:
                return i + 1
        return 0

    def _reasoning_ended_in_current_turn(self, input_ids: Sequence[int]) -> bool:
        """Whether the current assistant turn's reasoning has ended.

        Args:
            input_ids: The full token sequence to inspect.

        Returns:
            True if reasoning ended in the current turn, False otherwise.
        """
        start_index = self._current_turn_start_index(input_ids)
        has_eot_token = False
        for id in reversed(input_ids[start_index:]):
            if id == self.start_token_id:
                # Reasoning ends only if a BOT token is found before a EOT token.
                return has_eot_token
            elif id == self.end_token_id:
                has_eot_token = True
        return False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._reasoning_ended_in_current_turn(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self.end_token_id in delta_ids:
            return True
        # Grammar's think? is optional — if [THINK] was never generated in the
        # current turn, reasoning was skipped entirely. Scope the check to the
        # current turn so a prompt-injected [THINK] example is not counted.
        start_index = self._current_turn_start_index(input_ids)
        return self.start_token_id not in input_ids[start_index:]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content
        """
        has_bot_token = False
        has_eot_token = False
        bot_token_index = -1
        eot_token_index = -1
        # One for loop instead of multiple lookups
        for i, token_id in enumerate(input_ids):
            # We filter that we have multiple BOT tokens which should not
            # happen for a well prompted trained model
            if token_id == self.start_token_id and not has_bot_token:
                has_bot_token = True
                bot_token_index = i
            elif token_id == self.end_token_id:
                has_eot_token = True
                eot_token_index = i
                break

        # 1. Only BOT has been outputted
        if has_bot_token and not has_eot_token:
            # Should be = [] if model is well prompted and trained.
            return input_ids[:bot_token_index]
        # 2. Neither BOT or EOT have been outputted
        elif not has_bot_token and not has_eot_token:
            return input_ids
        # 3. Both BOT and EOT have been outputted.
        elif has_bot_token and has_eot_token:
            return input_ids[:bot_token_index] + input_ids[eot_token_index + 1 :]
        # 4. Only EOT has been outputted => this should not have occurred for a model
        #    well prompted and trained.
        else:
            return input_ids[:eot_token_index] + input_ids[eot_token_index + 1 :]

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if not model_output:
            return (None, "")

        # Check if the start token is present in the model output, remove it
        # if it is present.
        prev_bot_token, bot_token, post_bot_token = model_output.partition(
            self.start_token
        )

        has_bot_token = bool(bot_token)
        # Valid EOT tokens should follow BOT token
        has_valid_eot_token = has_bot_token and self.end_token in post_bot_token

        # 1. If there is BOT token followed by EOT token
        if has_bot_token and has_valid_eot_token:
            prev_eot_token, _, post_eot_token = post_bot_token.partition(self.end_token)
            # If model is well prompted and trained prev_bot_token should be ""
            content = prev_bot_token + post_eot_token
            return prev_eot_token, content if content else None
        # 2. Only BOT token
        elif has_bot_token:
            # If model is well prompted and trained prev_bot_token should be ""
            return post_bot_token, prev_bot_token if prev_bot_token else None
        # 3. EOT token has been outputted without BOT or neither has been outputted
        else:
            has_non_valid_eot_token = self.end_token in prev_bot_token
            # 3.a EOT token has been outputted without BOT
            # If model is well prompted and trained `has_non_valid_eot_token` should
            # be `False` and the parser outputs all tokens as 'content'
            if has_non_valid_eot_token:
                prev_eot_token, _, post_eot_token = prev_bot_token.partition(
                    self.end_token
                )
                return None, prev_eot_token + post_eot_token
            # 3.b neither BOT or EOT have been outputted
            else:
                return None, prev_bot_token
