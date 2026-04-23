# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
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
                # A <|tool_call> token ends reasoning only when it belongs to
                # the current generation turn. Scan further back to check:
                # - if we find a <|channel> start → active reasoning is
                #   ending (return True).
                # - if we find a turn boundary (<|turn> or <|tool_response>)
                #   before any channel start → this is a prior-turn tool call
                #   in the prompt, not the current generation (return False).
                # - if we find neither → the tool call is in the current
                #   context with no prior reasoning block (return True).
                # This avoids false positives from prior-turn tool calls in
                # multi-turn prompts, which is the root cause of #39885 on
                # the MoE A4B model variant where the generation prefix ends
                # with a priming <|tool_call> token and the backwards scan
                # finds the prior-turn <|tool_call> before hitting <|turn>.
                for j in range(i - 1, -1, -1):
                    if input_ids[j] == start_token_id:
                        # found the channel start — tool call ends reasoning
                        return True
                    if input_ids[j] in (
                        new_turn_token_id,
                        tool_response_token_id,
                    ):
                        # hit a turn boundary before finding channel start —
                        # this is a prior-turn tool call, not the current one
                        return False
                # no turn boundary found — tool call is in the current context
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                # We found a new turn or tool response token so don't consider
                # reasoning ended yet, since the model starts new reasoning
                # after these tokens.
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # During streaming we are already confirmed to be in the reasoning
        # phase (the caller tracks this). Any channel-end or tool-call token
        # in the current delta ends reasoning without needing further context.
        delta_list = list(delta_ids)
        return self.end_token_id in delta_list or self.tool_call_token_id in delta_list

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

    def _apply_prefix_stripping(
        self, result: DeltaMessage | None
    ) -> DeltaMessage | None:
        """Strip the ``thought\\n`` role label from streaming reasoning deltas.

        Accumulates reasoning text in ``_reasoning_text`` across calls and
        suppresses or trims the leading ``thought\\n`` label, which may
        arrive split across multiple deltas.
        """
        if result is None:
            return None
        if result.reasoning is None:
            return result

        self._reasoning_text += result.reasoning

        if self._prefix_stripped:
            return result

        # Case 1: accumulated text starts with the full prefix — strip it.
        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                # Prefix already consumed by prior deltas; pass through.
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
                    return None

        # Case 2: accumulated text is a strict prefix of _THOUGHT_PREFIX
        # (e.g. only "thou" so far). Buffer — can't tell yet if it diverges.
        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        # Case 3: text diverged from thought prefix. Re-emit everything
        # buffered so far to avoid data loss.
        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result

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

        On Gemma 4 MoE (A4B) models, the chat template does not emit a
        ``<|channel>`` start token when the previous message is a tool
        response — the model continues the same turn and outputs the
        ``thought\\n`` role label directly. The base-class implementation
        has no fallback for this case and routes every delta as
        ``content``, causing the reasoning leak (#39885).

        When neither ``<|channel>`` (start_token_id) appears in
        ``previous_token_ids`` nor ``delta_token_ids``, we mirror the
        sync ``extract_reasoning`` fallback: assume we are in the
        reasoning phase and treat each delta as reasoning until
        ``<channel|>`` (end_token_id) is seen.
        """
        start_seen = (
            self.start_token_id in previous_token_ids
            or self.start_token_id in delta_token_ids
        )

        if not start_seen:
            # Fallback path: no <|channel> token has been emitted.
            # Assume reasoning phase and route deltas accordingly.
            if self.end_token_id in previous_token_ids:
                # Past the end token — this is post-reasoning content.
                return DeltaMessage(content=delta_text)
            if self.end_token_id in delta_token_ids:
                # End token arrives in this delta; split at it.
                end_index = delta_text.find(self.end_token)
                if end_index == -1:
                    # Token present but text empty (e.g. special token
                    # rendered as empty string with skip_special_tokens=False)
                    # — nothing to emit, just mark the phase transition.
                    return None
                reasoning_part = delta_text[:end_index]
                content_part = delta_text[end_index + len(self.end_token) :]
                result = DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
                return self._apply_prefix_stripping(result)
            # Still inside reasoning with no start token seen yet.
            return self._apply_prefix_stripping(DeltaMessage(reasoning=delta_text))

        # Normal path: <|channel> token present — delegate to base class.
        base_result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        return self._apply_prefix_stripping(base_result)


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
