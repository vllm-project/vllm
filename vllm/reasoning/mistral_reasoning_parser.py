# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers.mistral import MistralTokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _partial_delimiter_len(text: str, *delimiters: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of any
    of ``delimiters``.

    Used while streaming to hold back a trailing fragment that might still grow
    into a ``[THINK]`` / ``[/THINK]`` marker (e.g. ``"...[/TH"``), so we never
    emit half a delimiter as content.
    """
    best = 0
    for delim in delimiters:
        for k in range(min(len(text), len(delim) - 1), best, -1):
            if text[-k:] == delim[:k]:
                best = k
                break
    return best


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

    @cached_property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        return SpecialTokens.begin_think

    @cached_property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        return SpecialTokens.end_think

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        has_eot_token = False

        for id in reversed(input_ids):
            if id == self.start_token_id:
                # Reasoning ends only if a BOT token is found before a EOT token.
                return has_eot_token
            elif id == self.end_token_id:
                has_eot_token = True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self.end_token_id in delta_ids:
            return True
        # Grammar's think? is optional — if [THINK] was never generated,
        # reasoning was skipped entirely.
        return self.start_token_id not in input_ids

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

    def _split_reasoning_content(self, text: str) -> tuple[str, str]:
        """Split ``text`` into ``(reasoning, content)`` on the literal
        ``[THINK]`` / ``[/THINK]`` delimiter strings.

        The delimiters are dropped. A stray ``[/THINK]`` outside a reasoning
        block is dropped too (matching :meth:`extract_reasoning`). A trailing
        partial delimiter is held back so streaming never emits half a marker.
        """
        start, end = self.start_token, self.end_token
        reasoning: list[str] = []
        content: list[str] = []
        in_think = False
        i = 0
        n = len(text)
        while i < n:
            if in_think:
                j = text.find(end, i)
                if j == -1:
                    rest = text[i:]
                    hold = _partial_delimiter_len(rest, end)
                    reasoning.append(rest[: len(rest) - hold] if hold else rest)
                    break
                reasoning.append(text[i:j])
                i = j + len(end)
                in_think = False
            else:
                start_idx = text.find(start, i)
                end_idx = text.find(end, i)
                if start_idx != -1 and (end_idx == -1 or start_idx < end_idx):
                    content.append(text[i:start_idx])
                    i = start_idx + len(start)
                    in_think = True
                elif end_idx != -1:
                    # Stray end-of-think with no matching begin — drop it.
                    content.append(text[i:end_idx])
                    i = end_idx + len(end)
                else:
                    rest = text[i:]
                    hold = _partial_delimiter_len(rest, start, end)
                    content.append(rest[: len(rest) - hold] if hold else rest)
                    break
        return "".join(reasoning), "".join(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Split reasoning from content by the literal ``[THINK]`` /
        ``[/THINK]`` strings rather than the special-token ids.

        The base implementation keys off the ``[THINK]`` / ``[/THINK]``
        special-token ids. Mistral models sometimes emit those delimiters as
        ordinary bracket *text* rather than the special tokens — most often a
        second reasoning block after a tool call, which is out-of-distribution
        for the prefilled opening ``[THINK]``. In that case the id-based path
        misses them and the raw ``[THINK]...[/THINK]`` leaks into the visible
        content. vLLM renders the special tokens as those same strings in the
        detokenized text, so scanning the text handles both the special-token
        and the literal-text forms uniformly — and keeps streaming consistent
        with the (already string-based) :meth:`extract_reasoning`.
        """
        prev_reasoning, prev_content = self._split_reasoning_content(previous_text)
        cur_reasoning, cur_content = self._split_reasoning_content(current_text)
        delta_reasoning = cur_reasoning[len(prev_reasoning) :]
        delta_content = cur_content[len(prev_content) :]
        if not delta_reasoning and not delta_content:
            return None
        return DeltaMessage(
            reasoning=delta_reasoning or None,
            content=delta_content or None,
        )

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
