# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.v1.token_buffer import rfind_tokens, token_slice

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


# How far into a prompt to look for the first turn's end.
FIRST_TURN_MAX_TOKENS = 1024


@functools.cache
def _colon_token_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    return frozenset(i for token, i in tokenizer.get_vocab().items() if ":" in token)


class GraniteReasoningParser(ReasoningParser):
    """Reasoning parser for IBM Granite.

    IBM granite models currently use "Here is my thought process:"
    and "Here is my response:" to separate its thinking / response outputs.

    is_reasoning_end reads the prompt to tell whether a thought section was
    asked for; a first turn that merely mentions a thought marker counts.
    When it was not, or the prompt already ends reasoning, extraction given
    that prompt through set_prompt_token_ids returns the output whole, as
    streaming does. Otherwise, or given no prompt, extraction reads the
    output alone, and markers count only at its start, after whitespace: a
    thought marker there opens reasoning that runs to the first response
    marker, and a response marker there is dropped. Any other output, a
    thought section after a preamble included, is content.

    The engine's gate reads token ids, so the grammar starts at the right
    token; a streaming frontend switches to content where the extraction
    does, through state that extract_reasoning_streaming sets, as K2
    Horizon's parser does (a general Parser hook would replace it).

    Structured output applies after the thought section the template asks
    for, or from the first token when thinking is off. With thinking on, a
    model that skips its thought section is not constrained: ending reasoning
    where the output leaves the marker prefix would put that token, the "{"
    of a direct JSON answer, outside the grammar, which then adds another.

    Known limits: a prompt that ends inside an open thought section (an
    assistant prefill) is not handled, and a stream that ends inside a
    response-marker prefix drops those held reasoning chars.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # NOTE: There have been some observed occurrences of quantized
        # instances of the current models using "Here's" instead of "Here is",
        # so to be safe, we match on both.
        self.valid_think_starts = [
            "Here's my thought process:",
            "Here is my thought process:",
        ]
        self.valid_response_starts = ["Here's my response:", "Here is my response:"]
        self.markers = (*self.valid_think_starts, *self.valid_response_starts)
        self.response_start_regex = re.compile(
            "|".join(map(re.escape, self.valid_response_starts))
        )

        # Every token of a response marker holds at least one of its
        # characters, so its length also bounds the tokens it spans.
        self.longest_response_start = max(
            len(response_start) for response_start in self.valid_response_starts
        )

        self.colon_token_ids = _colon_token_ids(tokenizer)
        # Set by extract_reasoning_streaming, which only the frontend calls.
        self._streamed_content: bool | None = None
        # Set from the prompt when the caller passes one.
        self._prompt_ends_reasoning = False

    def _asks_for_thoughts(self, prompt_token_ids: Sequence[int]) -> bool:
        # The template ignores `thinking` with a system message, tools or
        # documents, and offline chat passes no parser kwargs, so the first
        # turn says whether thoughts were asked for: its instruction, or a
        # user's own system prompt, names the thought marker.
        tokenizer = self.model_tokenizer
        eos = tokenizer.eos_token_id
        head = list(token_slice(prompt_token_ids, 0, FIRST_TURN_MAX_TOKENS))
        # Granite's BOS is its EOS, so a leading one does not end the turn.
        start = int(bool(head) and head[0] in (tokenizer.bos_token_id, eos))
        try:
            end = head.index(eos, start)
        except ValueError:
            end = len(head)
        text = tokenizer.decode(head[start:end])
        return any(think_start in text for think_start in self.valid_think_starts)

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        # The same decision the engine and the streaming frontend make, so
        # non-streaming extraction keeps the output whole when they do.
        self._prompt_ends_reasoning = self.is_reasoning_end(prompt_token_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._asks_for_thoughts(input_ids):
            return True
        # A response marker that ends the prompt (a prefill) ends reasoning.
        text = self.model_tokenizer.decode(self._turn_tail(input_ids, 0))
        return self.response_start_regex.search(text) is not None

    def _turn_tail(self, input_ids: Sequence[int], num_delta: int) -> list[int]:
        # The markers are text whose token ids depend on the character before
        # them, so read only the ids that can hold one ending in the last
        # `num_delta`, and none before the current turn: every chat turn ends
        # with EOS, and the system turn and earlier answers quote the markers.
        stop = len(input_ids)
        start = max(0, stop - num_delta - self.longest_response_start)
        tail = list(token_slice(input_ids, start, stop))
        return tail[rfind_tokens(tail, (self.model_tokenizer.eos_token_id,)) + 1 :]

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._streamed_content is not None:
            # The frontend's text can trail its token ids (a stop string holds
            # back the last characters), so its switch follows the extraction.
            return self._streamed_content
        delta_ids = list(delta_ids)
        # Every marker ends in ":", so only a delta holding one can end it.
        if self.colon_token_ids.isdisjoint(delta_ids):
            return False
        num_delta = len(delta_ids)
        tail = self._turn_tail(input_ids, num_delta)
        decode = self.model_tokenizer.decode
        search = self.response_start_regex.search
        return search(decode(tail)) is not None and (
            search(decode(tail[: max(0, len(tail) - num_delta)])) is None
        )

    def _split(self, text: str, final: bool) -> tuple[str | None, str | None]:
        """Split `text` into reasoning and content at the response marker.

        With `final` False, `text` is a prefix of the output, and text that
        a later token can still turn into a marker is left out of both parts.
        """
        stripped = text.lstrip()
        lead = len(text) - len(stripped)
        think = next(
            (t for t in self.valid_think_starts if stripped.startswith(t)), None
        )
        if think is None:
            response = next(
                (r for r in self.valid_response_starts if stripped.startswith(r)), None
            )
            if response is not None:
                # The marker opens the response, so it is not content either.
                content = text[:lead] + text[lead + len(response) :]
                return None, (content or None) if final else content
            if not final and self._is_marker_prefix(stripped):
                return None, None
            return None, text
        think_end = lead + len(think)
        response = self.response_start_regex.search(text, think_end)
        if response is None:
            held = 0
            if not final:
                # Imported here: only the frontend streams text, and the module
                # pulls the chat protocol into the engine core.
                from vllm.tool_parsers.utils import partial_tag_overlap

                held = max(
                    partial_tag_overlap(text, r) for r in self.valid_response_starts
                )
            # A thought section the response never followed is reasoning only.
            reasoning = text[think_end : max(think_end, len(text) - held)]
            return (reasoning or None) if final else reasoning, None
        reasoning = text[think_end : response.start()]
        content = text[response.end() :]
        if final:
            return reasoning or None, content or None
        return reasoning, content

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Called on the delta that ends the response marker; its content text
        # comes from extract_reasoning_streaming.
        # shortcut: drops the ids of content sharing that delta, split them
        # out if a consumer ever needs those ids.
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """Extract the reasoning content & content sections, respectively.
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Args:
            model_output (str): Output of the model to be parsed.
            request (ChatCompletionRequest): Request being processed.

        Returns:
            tuple[Optional[str], Optional[str]]: Tuple pair containing the
            reasoning content and non-reasoning content.

        """
        if self._prompt_ends_reasoning:
            return None, model_output
        return self._split(model_output, final=True)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract the reasoning content / content emitted by granite models;
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        NOTE: Granite models do not use a special token to start their reasoning
        and response sections; instead they have token sequences, e.g.,

                Here is my thought process: Foo Here is my response: Bar

        This increases the complexity of correctly handling streams, since we
        need to watch for specific sequences and correctly parse them without
        dropping content that is potentially overlapping & spanning multiple
        delta messages.

        Args:
            previous_text (str): Previous text outside of this delta message.
            current_text (str): Previous text + delta text.
            delta_text (str): Text to consider and parse content from.
            previous_token_ids (Sequence[int]): Token IDs of previous_text.
            current_token_ids (Sequence[int]): Token IDs of current_text.
            delta_token_ids (Sequence[int]): Token IDs of delta_text.

        Returns:
            Union[DeltaMessage, None]
                DeltaMessage with either reasoning content or content, or None.

        """
        previous = current_text[: max(0, len(current_text) - len(delta_text))]
        old_reasoning, old_content = self._split(previous, final=False)
        new_reasoning, new_content = self._split(current_text, final=False)
        self._streamed_content = new_content is not None
        reasoning = (new_reasoning or "")[len(old_reasoning or "") :]
        content = (new_content or "")[len(old_content or "") :]
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        """Called on the last delta of a stream that never reached the
        response marker; returns the content text held back so far. Held
        reasoning text cannot be returned here and stays unsent."""
        _, content = self._split(previous_text, final=True)
        _, streamed = self._split(previous_text, final=False)
        return (content or "")[len(streamed or "") :] or None

    def _is_marker_prefix(self, text: str) -> bool:
        """Whether `text` can still grow into a thought or response marker."""
        return any(marker.startswith(text) for marker in self.markers)
