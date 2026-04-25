# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.utils import partial_tag_overlap

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: Qwen3.5 models may emit <tool_call> inside the thinking block
    without closing </think> first. <tool_call> is treated as an implicit
    end of reasoning, matching the approach in KimiK2ReasoningParser.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning tokens in the model OUTPUT.

        The Qwen3.5+ chat template places ``<think>`` in the prompt, so the
        output begins with reasoning tokens followed by ``</think>``.  The
        inherited depth-counter assumes ``<think>`` lives in the output and
        therefore returns 0 reasoning tokens for Qwen3.5 outputs.

        This override:
        - Treats the start of the output as the reasoning region by
          default (no leading ``<think>`` required).
        - If ``<think>`` IS present in the output (older 2507-era
          template), reasoning begins right after it.
        - Stops counting at the FIRST ``</think>`` or — for the Qwen3.5
          implicit-end case — the FIRST ``<tool_call>``.
        """
        end_token_id = self.end_token_id
        start_token_id = self.start_token_id
        tool_call_token_id = self._tool_call_token_id

        start_idx = 0
        if start_token_id in token_ids:
            for i, tid in enumerate(token_ids):
                if tid == start_token_id:
                    start_idx = i + 1
                    break

        count = 0
        for i in range(start_idx, len(token_ids)):
            tid = token_ids[i]
            if tid == end_token_id:
                return count
            if tool_call_token_id is not None and tid == tool_call_token_id:
                return count
            count += 1
        return count

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # Skip <tool_call> tokens that are paired with a subsequent
                # </tool_call> — these appear in system-prompt tool examples
                # and must not be mistaken for an implicit reasoning end.
                # Unpaired <tool_call> (model output) still signals the end.
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._tool_call_token_id is not None:
            return self._tool_call_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at the FIRST <tool_call> 
        # (implicit reasoning end).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = input_ids.index(self._tool_call_token_id)
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning or None, content or None

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return reasoning or None, content or None
        # Thinking enabled but no </think>: output was truncated.
        # Everything generated so far is reasoning.
        return model_output, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta.

        Since <think> is placed in the prompt by the chat template, all
        generated tokens before </think> are reasoning and tokens after
        are content.

        NOTE: When thinking is disabled, no think tokens appear in the
        generated output. The serving layer normally detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method, but we ALSO honour ``self.thinking_enabled``
        here so the parser is self-consistent with ``extract_reasoning``
        — direct callers (and any future regression of the bypass) get
        content rather than mis-categorised reasoning.
        """
        # Thinking explicitly disabled — every delta is content.  We still
        # split at </think> if (rarely) the model emits it in the output.
        if not self.thinking_enabled and self.end_token_id not in delta_token_ids:
            if not delta_text:
                return None
            return DeltaMessage(content=delta_text)

        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        
        # If thinking already ended, everything is content.
        if (self.end_token_id in previous_token_ids or 
            (self._tool_call_token_id is not None and 
             self._tool_call_token_id in previous_token_ids) or
            (bool(self._tool_call_tag) and 
             self._tool_call_tag in previous_text)):
            return DeltaMessage(content=delta_text)

        # Implicit reasoning end via <tool_call>.
        has_tool_call_id = (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in delta_token_ids
        )
        just_completed_tool_call_tag = (
            bool(self._tool_call_tag)
            and self._tool_call_tag in current_text
            and self._tool_call_tag not in previous_text
        )

        if has_tool_call_id or just_completed_tool_call_tag:
            if self._tool_call_tag and self._tool_call_tag in current_text:
                tag_start_idx = current_text.find(self._tool_call_tag)
                delta_start_idx = len(previous_text)
                
                if tag_start_idx >= delta_start_idx:
                    reasoning_len = tag_start_idx - delta_start_idx
                    reasoning = delta_text[:reasoning_len]
                    content = delta_text[reasoning_len:]
                else:
                    # Part of the tag was already emitted as reasoning.
                    # We MUST emit the full tag as content for the tool parser,
                    # but we avoid emitting it as reasoning in this delta.
                    reasoning = None
                    content = current_text[tag_start_idx:]
                
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )

        # To avoid leaking fragments of <tool_call> into reasoning,
        # withhold any suffix of current_text that could grow into the
        # tool_call tag.  Combine the bytes withheld by the previous
        # delta with this delta's (already-stripped) text and recompute
        # the overlap on the cumulative text.  When the continuation
        # reveals the withheld characters were NOT in fact part of
        # <tool_call> (e.g. the model emits "<tool_use>" instead), the
        # previously withheld bytes are now released and emitted as
        # reasoning.  The naive ``delta_text[:len(delta_text) - overlap]``
        # would silently drop them.
        prev_overlap = partial_tag_overlap(previous_text, self._tool_call_tag)
        withheld_from_prev = (
            previous_text[len(previous_text) - prev_overlap:]
            if prev_overlap > 0 else ""
        )
        combined = withheld_from_prev + delta_text
        curr_overlap = partial_tag_overlap(current_text, self._tool_call_tag)
        sendable_len = len(combined) - curr_overlap
        if sendable_len > 0:
            return DeltaMessage(reasoning=combined[:sendable_len])
        if curr_overlap > 0:
            # Still withholding a partial tag — emit nothing yet but keep
            # processing alive (None would signal "stop processing").
            return DeltaMessage()
        return None
