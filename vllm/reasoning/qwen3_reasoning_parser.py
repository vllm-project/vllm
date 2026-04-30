# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

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

        # Buffer used by extract_reasoning_streaming to hold back a tentative
        # <tool_call> until <function=…> lookahead can confirm it is genuine.
        # None when not buffering; a str (starting with "<tool_call>") when
        # we are waiting for the next delta to confirm.
        self._tool_call_pending: str | None = None

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    # Regex for detecting <function=…> or <function name="…"> in text.
    _RE_FUNCTION_OPEN = re.compile(r"<function[\s=]")

    # Number of tokens to decode after <tool_call> when checking whether a
    # genuine <function=…> tag follows.  The gap is always "\n<function=" in
    # the Qwen3 format, which is typically 3 tokens:
    #   \n          → 1 token
    #   <function   → 1 token (special token in the Qwen3 vocabulary)
    #   =           → 1 token
    # We use 8 to stay correct even when the tokenizer does not have
    # <function> as a single special token and encodes it character-by-character
    # (worst case: '<', 'f', 'u', 'n', 'c', 't', 'i', 'o', 'n', '=' = 10 chars,
    # but most sub-word tokenizers merge runs like this into 3-5 tokens).
    _FUNCTION_LOOKAHEAD_TOKENS: int = 8

    def _tool_call_is_genuine_end(
        self, input_ids: Sequence[int], tool_call_pos: int
    ) -> bool:
        """Return True iff <function=…> follows <tool_call> at tool_call_pos.

        Decodes a small window of tokens after the <tool_call> position and
        checks for a function-open tag (with optional leading whitespace).
        Returns False when no tokens follow or decoding fails.
        """
        after = list(
            input_ids[tool_call_pos + 1 : tool_call_pos + 1 + self._FUNCTION_LOOKAHEAD_TOKENS]
        )
        if not after:
            return False
        try:
            text = self.model_tokenizer.decode(after, skip_special_tokens=False)
            return bool(self._RE_FUNCTION_OPEN.match(text.lstrip()))
        except Exception:
            return False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                # Found <think> before </think> or <tool_call>
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # Skip paired occurrences (prompt few-shot examples).
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                # Lookahead: only treat as implicit end if <function=…> follows.
                # Reasoning text often mentions <tool_call> without following
                # it with a function tag; wait for confirmation.
                if not self._tool_call_is_genuine_end(input_ids, i):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # Materialise once to avoid double-consuming a one-pass iterable.
        delta_set = set(delta_ids)
        if self.end_token_id in delta_set:
            return True
        if self._tool_call_token_id is not None and self._tool_call_token_id in delta_set:
            # Require <function=…> to follow in the accumulated sequence.
            tc_id = self._tool_call_token_id
            ids = list(input_ids)
            try:
                pos = len(ids) - 1 - ids[::-1].index(tc_id)
                return self._tool_call_is_genuine_end(ids, pos)
            except ValueError:
                pass
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at <tool_call> (implicit reasoning end).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            )
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
            return reasoning, content or None

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        # The <tool_call> is only a genuine implicit end when followed
        # (modulo whitespace) by <function=…>.  Bare <tool_call> mentions in
        # reasoning text must not prematurely terminate reasoning.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            after_tc = model_output[tool_call_index + len(self._tool_call_tag):]
            if self._RE_FUNCTION_OPEN.match(after_tc.lstrip()):
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
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
        # Reset per-request state at the very first delta of a new generation
        # (previous_token_ids is empty).  Guards against _tool_call_pending
        # leaking from a previous request that ended while the buffer was set
        # (e.g. the stream was cut immediately after <tool_call>).
        if not previous_token_ids:
            self._tool_call_pending = None

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

        # Implicit reasoning end via <tool_call>.
        # Apply the same <function=…> lookahead used by is_reasoning_end so
        # that a bare <tool_call> token in reasoning text (e.g. a code
        # example) does not prematurely terminate reasoning.
        # We use current_token_ids (which already includes the delta) so the
        # check works whether <function=…> arrives in the same delta or the
        # next one.
        #
        # The challenge: <tool_call> often lands at the very end of a streaming
        # chunk, with <function=…> arriving only in the *next* chunk.  A
        # lookahead at that moment finds no tokens after <tool_call> and returns
        # False, causing the tag to be emitted as reasoning.  The *following*
        # chunk then confirms the lookahead, but the tool parser never sees the
        # required <tool_call> prefix, so it treats <function=…> as plain text.
        #
        # Solution: when the lookahead fails while <tool_call> is in the delta,
        # buffer the tag (and any trailing whitespace in that same delta) instead
        # of emitting it as reasoning.  Subsequent deltas are appended to the
        # buffer until lookahead confirms (→ emit as content) or non-whitespace
        # arrives that cannot lead to <function=…> (→ flush as reasoning).
        if self._tool_call_token_id is not None:
            tc_id = self._tool_call_token_id
            tc_in_delta = tc_id in delta_token_ids
            tc_in_prev = tc_id in previous_token_ids

            # --- Pending-buffer path: a previous delta left <tool_call>
            # buffered; re-evaluate lookahead with the new token context. ---
            if self._tool_call_pending is not None:
                ids = list(current_token_ids)
                try:
                    pos = len(ids) - 1 - ids[::-1].index(tc_id)
                    if self._tool_call_is_genuine_end(ids, pos):
                        # Lookahead confirmed: emit the full buffered text
                        # plus the current delta as content so the downstream
                        # tool parser receives the complete "<tool_call>\n
                        # <function=…>" sequence it needs.
                        content = self._tool_call_pending + delta_text
                        self._tool_call_pending = None
                        return DeltaMessage(content=content)
                except ValueError:
                    pass

                # Lookahead still unconfirmed.  Keep buffering only if the
                # text accumulated after <tool_call> is still a plausible
                # prefix of "<function" (the only valid tag that can follow
                # <tool_call>).  Anything else means it was not a genuine
                # tool-call opener, so flush immediately as reasoning.
                after_tc = self._tool_call_pending[len(self._tool_call_tag):]
                candidate = (after_tc + delta_text).lstrip()
                func_prefix = "<function"
                still_viable = (
                    not candidate                        # still pure whitespace
                    or func_prefix.startswith(candidate) # candidate is a prefix of "<function"
                    or candidate.startswith(func_prefix) # candidate already has the full prefix
                )
                if still_viable:
                    self._tool_call_pending += delta_text
                    return None
                # Accumulated text diverged from "<function": flush as reasoning.
                reasoning_buf = self._tool_call_pending + delta_text
                self._tool_call_pending = None
                return DeltaMessage(reasoning=reasoning_buf)

            # --- Normal path: <tool_call> arrives in this or a prior delta. ---
            if tc_in_delta or tc_in_prev:
                ids = list(current_token_ids)
                try:
                    pos = len(ids) - 1 - ids[::-1].index(tc_id)
                    if self._tool_call_is_genuine_end(ids, pos):
                        if tc_in_delta:
                            tool_index = delta_text.find(self._tool_call_tag)
                            if tool_index >= 0:
                                reasoning = delta_text[:tool_index]
                                content = delta_text[tool_index:]
                                return DeltaMessage(
                                    reasoning=reasoning if reasoning else None,
                                    content=content if content else None,
                                )
                        else:
                            # <tool_call> was confirmed in a prior delta;
                            # current delta is all content.
                            return DeltaMessage(content=delta_text)
                except ValueError:
                    pass

                # Lookahead returned False or <function=…> not yet arrived.
                if tc_in_delta and self.end_token_id not in previous_token_ids:
                    # Buffer <tool_call> (and any text that follows it in this
                    # delta) rather than emitting it as reasoning.  This lets
                    # the *next* delta's lookahead confirm or deny the call.
                    # Only buffer while still in reasoning mode: once </think>
                    # has passed the existing content-mode fallthrough below
                    # handles the chunk correctly without buffering.
                    tool_index = delta_text.find(self._tool_call_tag)
                    if tool_index >= 0:
                        pre = delta_text[:tool_index]
                        self._tool_call_pending = delta_text[tool_index:]
                        return DeltaMessage(reasoning=pre) if pre else None
                # tc_in_prev=True but lookahead failed, or in content mode:
                # fall through to content-mode emit below.

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
