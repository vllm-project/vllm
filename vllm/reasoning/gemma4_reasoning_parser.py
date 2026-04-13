# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Role label that Gemma4 emits at the start of the thinking channel.
# The model generates: <|channel>thought\n...reasoning...<channel|>
# This prefix must be stripped to expose only the actual reasoning content.
_THOUGHT_PREFIX = "thought\n"


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning/thinking
    content within its output. Thinking mode is activated by passing
    ``enable_thinking=True`` in the chat template kwargs, which injects a
    system turn containing <|think|> (token 98) to trigger chain-of-thought
    reasoning.

    Output pattern when thinking is enabled::

        <|channel>thought
        ...chain of thought reasoning...<channel|>
        Final answer text here.

    The ``thought\\n`` role label inside the channel delimiters is a
    structural artefact (analogous to ``user\\n`` in ``<|turn>user\\n...``).
    This parser strips it so that downstream consumers see only the
    actual reasoning text, consistent with the offline parser
    (``vllm.reasoning.gemma4_utils._strip_thought_label``).
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Instance state for streaming prefix stripping.
        # Tracks only the reasoning text received from the base parser,
        # independent of current_text (which may contain pre-reasoning
        # content and lacks special token text due to
        # skip_special_tokens=True).
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        """Disable special-token stripping to preserve boundary tokens."""
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<|channel>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        # Search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                # We're generating a tool call, so reasoning must be ended.
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                # We found a new turn or tool response token so don't consider
                # reasoning ended yet, since the model starts new reasoning
                # after these tokens.
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\n`` role label."""
        if self.start_token not in model_output and self.end_token not in model_output:
            # Default to content history if no tags are present
            # (or if they were stripped)
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract streaming reasoning, stripping ``thought\\n`` from the
        first reasoning delta(s).

        The ``thought\\n`` prefix may arrive as a single delta or split
        across multiple deltas (e.g. ``"thought"`` then ``"\\n"``). We
        buffer early reasoning tokens until we can determine whether the
        prefix is present, then emit the buffered content minus the
        prefix.

        Unlike the previous implementation which reconstructed accumulated
        reasoning from ``current_text``, this uses instance state
        (``_reasoning_text``) to track only the reasoning content returned
        by the base parser. This is necessary because
        ``skip_special_tokens=True`` (the vLLM default) causes the
        ``<|channel>`` delimiter to be invisible in ``current_text``,
        making it impossible to separate pre-reasoning content from
        reasoning content via string matching.
        """
        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if result is None:
            return None

        if result.reasoning is None:
            return result

        # Accumulate ONLY the reasoning text from base parser results.
        # This is immune to pre-reasoning content pollution.
        self._reasoning_text += result.reasoning

        # Once the prefix has been handled, all subsequent reasoning
        # deltas pass through unchanged.
        if self._prefix_stripped:
            return result

        # ---- Prefix stripping logic ----

        # Case 1: We've accumulated enough to confirm the prefix is
        # present. Strip it and pass through the remainder.
        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            # How much reasoning was accumulated before this delta?
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                # Prefix was already consumed by prior deltas; this
                # delta is entirely real content — pass through.
                self._prefix_stripped = True
                return result
            else:
                # Part or all of the prefix is in this delta.
                chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
                stripped = result.reasoning[chars_of_prefix_in_delta:]
                if stripped:
                    self._prefix_stripped = True
                    result.reasoning = stripped
                    return result
                else:
                    if len(self._reasoning_text) >= prefix_len:
                        self._prefix_stripped = True
                        result.reasoning = ""
                        return result
                    return None

        # Case 2: Accumulated text is a strict prefix of
        # _THOUGHT_PREFIX (e.g. we've only seen "thou" so far).
        # Buffer by suppressing — we can't yet tell if this will
        # become the full prefix or diverge.
        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        # Case 3: Accumulated text doesn't match the thought prefix
        # at all. This means prior deltas were buffered (suppressed
        # by Case 2) but the text diverged. Re-emit the full
        # accumulated text to avoid data loss.
        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
