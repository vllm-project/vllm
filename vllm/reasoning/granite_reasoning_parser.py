# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

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
# Tokens read from the start of the output or a prefill: whitespace and a marker.
TURN_HEAD_TOKENS = 64
# How a prompt leaves the model's turn (see _prefill_state).
PromptState = Literal["ended", "thought", "open"]
ROLE_START = "<|start_of_role|>"
ROLE_END = "<|end_of_role|>"


@functools.cache
def _vocab_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[frozenset[int], int | None, int | None]:
    """The ids of tokens holding a ":", and of the role-header tokens."""
    vocab = tokenizer.get_vocab()
    colon_ids = frozenset(i for t, i in vocab.items() if ":" in t)
    return colon_ids, vocab.get(ROLE_START), vocab.get(ROLE_END)


class GraniteReasoningParser(ReasoningParser):
    """Reasoning parser for IBM Granite.

    IBM granite models currently use "Here is my thought process:"
    and "Here is my response:" to separate its thinking / response outputs.

    Markers count only at the start of the output, after whitespace: a
    thought marker there opens reasoning that runs to the first response
    marker, a response marker there opens the content, and any other output
    is content. The engine's gate, which reads token ids, ends reasoning by
    the same rule, so the grammar starts where the content does; the engine
    passes the prompt through set_prompt_token_ids first, and without it the
    ids are taken as output only. A prompt that does not ask for thoughts,
    or already ends reasoning (text after the assistant role header, or a
    raw prompt ending with a response marker), makes the whole output
    content. An assistant prefill that leaves a thought section open, or a
    raw prompt ending with a thought marker, puts the output inside it:
    reasoning runs to the first response marker.

    With thinking on, a model that skips its thought section is not
    constrained: ending reasoning where the output leaves the marker prefix
    would put that token, the "{" of a direct JSON answer, outside the
    grammar.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # NOTE: There have been some observed occurrences of quantized
        # instances of the current models using "Here's" instead of "Here is",
        # so to be safe, we match on both.
        self.valid_think_starts = (
            "Here's my thought process:",
            "Here is my thought process:",
        )
        self.valid_response_starts = ("Here's my response:", "Here is my response:")
        self.markers = (*self.valid_think_starts, *self.valid_response_starts)
        self.response_start_regex = re.compile(
            "|".join(map(re.escape, self.valid_response_starts))
        )

        # Every token of a response marker holds at least one of its
        # characters, so its length also bounds the tokens it spans.
        self.longest_response_start = max(
            len(response_start) for response_start in self.valid_response_starts
        )

        self.colon_token_ids, self.role_start_id, self.role_end_id = _vocab_ids(
            tokenizer
        )
        # Set by extract_reasoning_streaming, which only the frontend calls.
        self._streamed_content: bool | None = None
        # Set from the prompt when the caller passes one.
        self._prompt_ends_reasoning = False
        self._output_in_thought = False
        self._output_start = 0

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
        state = self._prompt_state(prompt_token_ids)
        self._prompt_ends_reasoning = state == "ended"
        self._output_in_thought = state == "thought"
        # The engine's gate reads the prompt and output ids together.
        self._output_start = len(prompt_token_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._prompt_state(input_ids) == "ended"

    def _prompt_state(self, prompt_token_ids: Sequence[int]) -> PromptState:
        if not self._asks_for_thoughts(prompt_token_ids):
            return "ended"
        return self._prefill_state(prompt_token_ids)

    def _prefill_state(self, input_ids: Sequence[int]) -> PromptState:
        """How the prompt leaves the model's turn: "ended" when it holds the
        start of the answer, "thought" when it leaves a thought section open,
        else "open"."""
        decode = self.model_tokenizer.decode
        start = self._turn_start(input_ids)
        if start is None:
            # A raw prompt can quote the markers, so only its last words count.
            num = len(input_ids)
            tail = token_slice(input_ids, max(0, num - TURN_HEAD_TOKENS), num)
            text = decode(list(tail)).rstrip()
            if text.endswith(self.valid_response_starts):
                return "ended"
            return "thought" if text.endswith(self.valid_think_starts) else "open"
        eos = self.model_tokenizer.eos_token_id
        if rfind_tokens(input_ids, (eos,), start) >= 0 or not self._is_assistant_turn(
            input_ids, start
        ):
            # A closed turn or another role's text: the model writes its own
            # assistant turn.
            return "open"
        head = decode(list(token_slice(input_ids, start, start + TURN_HEAD_TOKENS)))
        head = head.lstrip()
        if not head:
            return "open"
        if not head.startswith(self.valid_think_starts):
            return "ended"
        prefill = decode(list(token_slice(input_ids, start, len(input_ids))))
        _, content = self._split(prefill, final=False, in_thought=False)
        return "thought" if content is None else "ended"

    def _turn_start(self, input_ids: Sequence[int]) -> int | None:
        """Where the last turn's text starts, after its role header, or None
        without one (a raw prompt)."""
        if self.role_end_id is None:
            return None
        start = rfind_tokens(input_ids, (self.role_end_id,)) + 1
        return start or None

    def _is_assistant_turn(self, input_ids: Sequence[int], start: int) -> bool:
        header = list(token_slice(input_ids, max(0, start - 8), start - 1))
        if self.role_start_id not in header:
            return False
        role = header[len(header) - header[::-1].index(self.role_start_id) :]
        return self.model_tokenizer.decode(role).strip() == "assistant"

    def _opens(self, input_ids: Sequence[int]) -> bool:
        # Extraction honors markers only at the start of the output, in text
        # detokenized without special tokens.
        if self._output_in_thought:
            return True
        start = self._output_start
        head = token_slice(input_ids, start, start + TURN_HEAD_TOKENS)
        text = self.model_tokenizer.decode(list(head), skip_special_tokens=True)
        return text.lstrip().startswith(self.markers)

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
        # Extraction reads the output alone, so a marker begun in the prompt's
        # prefill does not count.
        tail = tail[max(0, len(tail) - (len(input_ids) - self._output_start)) :]
        decode = self.model_tokenizer.decode
        search = self.response_start_regex.search
        return (
            search(decode(tail)) is not None
            and search(decode(tail[: max(0, len(tail) - num_delta)])) is None
            and self._opens(input_ids)
        )

    def _split(
        self, text: str, final: bool, in_thought: bool
    ) -> tuple[str | None, str | None]:
        """Split `text` into reasoning and content at the response marker.

        With `final` False, `text` is a prefix of the output, and text that
        a later token can still turn into a marker is left out of both parts.
        """
        # A prefill that opened the thought section puts the text inside it.
        think_end = 0 if in_thought else self._think_end(text)
        if think_end is None:
            stripped = text.lstrip()
            lead = len(text) - len(stripped)
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

    def _think_end(self, text: str) -> int | None:
        """The end of the thought marker that opens `text`, or None."""
        stripped = text.lstrip()
        think = next(
            (t for t in self.valid_think_starts if stripped.startswith(t)), None
        )
        return None if think is None else len(text) - len(stripped) + len(think)

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
        return self._split(model_output, True, self._output_in_thought)

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
        in_thought = self._output_in_thought
        old_reasoning, old_content = self._split(previous, False, in_thought)
        new_reasoning, new_content = self._split(current_text, False, in_thought)
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
        _, content = self._split(previous_text, True, self._output_in_thought)
        _, streamed = self._split(previous_text, False, self._output_in_thought)
        return (content or "")[len(streamed or "") :] or None

    def _is_marker_prefix(self, text: str) -> bool:
        """Whether `text` can still grow into a thought or response marker."""
        return any(marker.startswith(text) for marker in self.markers)
