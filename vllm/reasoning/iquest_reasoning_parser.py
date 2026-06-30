# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class IquestReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the iQuest model family (Qwen3.5-style template).

    The iQuest model family uses <think>...</think> tokens to denote reasoning
    text. The chat template places <think> in the prompt so only </think>
    appears in the generated output. The model provides a strict switch to
    disable reasoning output via the 'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Some releases use an older chat template where the model generates
    <think> itself. This parser handles both styles: if <think> appears in the
    generated output it is stripped before extraction (non-streaming) or
    skipped (streaming).
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    # The template's "thinking disabled" tail (``<think>\n\n</think>\n\n``)
    # leaves only a couple of whitespace tokens after ``</think>``. More than
    # this many trailing tokens means real content follows (e.g. a prior
    # conversation turn), so reasoning has not ended for the upcoming output.
    _PROMPT_REASONING_END_TAIL = 8

    def is_prompt_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Tail-only reasoning-end detection for prompts.

        Unlike :meth:`is_reasoning_end`, which returns True for any ``</think>``
        in the sequence, this only reports an ended reasoning section when the
        ``</think>`` sits at the very tail of the prompt followed solely by
        whitespace — the ``enable_thinking=False`` template pattern. A prior
        turn's replayed ``</think>`` (followed by real content and a new
        generation boundary) is correctly treated as *not* ended, so the
        current turn's freshly generated reasoning is still parsed.
        """
        last_end = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.end_token_id:
                last_end = i
                break
        if last_end == -1:
            return False

        trailing = input_ids[last_end + 1 :]
        # A fresh, still-open think block after the last </think> means the
        # model is reasoning again (should not happen in a prompt, but be safe).
        if self.start_token_id in trailing:
            return False
        if len(trailing) > self._PROMPT_REASONING_END_TAIL:
            return False
        decoded = self.model_tokenizer.decode(list(trailing), skip_special_tokens=False)
        return decoded.strip() == ""

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is disabled (no </think> in output), returns
        (None, model_output) to indicate all output is content.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            # No end token means thinking is disabled or the model
            # did not produce reasoning. Treat everything as content.
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        # The chat template trains the model to emit "\n\n" right after
        # </think>. Strip the leading whitespace so it doesn't leak into the
        # content. (For slow tokenizers, spaces_between_special_tokens may also
        # prepend a space to the special token here, which this lstrip likewise
        # removes; fast tokenizers take the direct convert_tokens_to_string
        # path in detokenize_incrementally and never add that space.)
        content = content.lstrip()

        final_content = content or None
        return reasoning, final_content

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
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
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
        elif self.end_token_id in previous_token_ids:
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        else:
            # No end token yet: still in reasoning phase.
            return DeltaMessage(reasoning=delta_text)
