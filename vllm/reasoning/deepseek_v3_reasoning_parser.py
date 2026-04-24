# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

# Phase 1: complete <tool_call>...</tool_call> blocks, or truncated open-to-EOS.
# Handles cases where the full open tag is present in reasoning_content.
_EMBEDDED_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>.*$",
    re.DOTALL,
)

# Phase 2: GLM uses token id 154829 for <tool_call>.  vLLM strips special
# tokens from text output, so the open tag is missing but the body remains.
# Pattern: optional-funcname  (<arg_key>...<arg_value>...)+ </tool_call>
# We require at least one <arg_key> tag so that bare closing fragments
# (Phase 3 territory) do not accidentally match here.
_BODY_WITHOUT_OPEN_RE = re.compile(
    r"^(?:[\w.-]+\n?)?"           # optional function name (no spaces, optional newline)
    r"(?:<arg_key>.*?</arg_key>"  # at least one complete key-value pair
    r"<arg_value>.*?</arg_value>)*"
    r"<arg_key>.*?</arg_key>"     # required: at least one <arg_key> must be present
    r"<arg_value>.*?"             # value (may be truncated)
    r"(?:</arg_value>)?"
    r"</tool_call>"               # required closing tag
    r".*$",
    re.DOTALL,
)

# Phase 3: closing-only fragment — the head of the call is already in content,
# only the tail landed in reasoning_content.
# Examples:
#   "</arg_value></tool_call>"
#   "1\n</arg_value></tool_call>"
#   "ls /\n</arg_value></tool_call>"
_CLOSING_FRAGMENT_RE = re.compile(
    r"^[^<]*</arg_value>\s*</tool_call>\s*$",
    re.DOTALL,
)


class DeepSeekV3ReasoningParser(ReasoningParser):
    """
    V3 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking` and `separate_reasoning`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        self._parser: ReasoningParser
        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )


class DeepSeekV3ReasoningWithThinkingParser(DeepSeekV3ReasoningParser):
    """
    DeepSeekV3ReasoningParser that defaults to thinking mode.

    Includes a three-phase fix for GLM-5.1-FP8 (and similar models) that emit
    tool-call XML inside <think> blocks at long context (~8k+ tokens).

    The base sequential pipeline captures such tool calls in reasoning_content
    before the tool parser can see them, causing finish_reason=stop instead of
    finish_reason=tool_calls.  Three patterns are handled:

    Phase 1 — Full embedded block:
        <tool_call>funcname<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        The complete open+body+close is inside reasoning.  Promoted as-is.

    Phase 2 — Body-without-open-tag:
        funcname<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        Token id 154829 (<tool_call>) is a special token stripped by vLLM.
        The body is reconstructed by prepending <tool_call>.

    Phase 3 — Closing fragment only:
        </arg_value></tool_call>   or   somevalue\n</arg_value></tool_call>
        The head of the call already landed in content (before </think>); only
        the tail leaked into reasoning.  The fragment is appended to content.

    Ref: https://github.com/vllm-project/vllm/pull/39055 (Qwen3 analog)
    Ref: https://amd-hub.atlassian.net/browse/AIVLLM-229
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True
            chat_kwargs["enable_thinking"] = True
            kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)

    @staticmethod
    def _split_embedded_tool_calls(
        reasoning: str | None,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Promote tool-call XML from reasoning_content into content.

        Handles patterns produced by GLM-5.1-FP8 at long context where the
        <tool_call> special token (id 154829) is stripped by vLLM and/or the
        model emits multiple concatenated calls inside a <think> block:

        Phase 1 — complete <tool_call>...</tool_call> blocks inside reasoning.
            Promoted verbatim.  Runs first and strips all complete blocks,
            leaving any partial tail in `cleaned`.

        Phase 2 — body-without-open-tag: after Phase 1, `cleaned` may contain
            the body of an additional call whose <tool_call> open token was
            stripped (e.g. "bash<arg_key>cmd</arg_key><arg_value>...</arg_value>
            </tool_call>").  Reconstructed by prepending <tool_call>.
            Runs on the Phase-1 remainder regardless of whether Phase 1 fired.

        Phase 3 — closing fragment only: the model stopped mid-output at the
            stop token, so reasoning contains only a tail fragment such as
            "\n271</arg_value></tool_call>".  The head may be in `content`
            (append) or nowhere (content=None — the fragment is promoted as a
            standalone truncated block so the tool parser can attempt recovery).
        """
        if not reasoning:
            return reasoning, content

        promoted: list[str] = []
        phase_counts = [0, 0, 0]  # Phase 1, 2, 3

        # ── Phase 1: strip all complete <tool_call>...</tool_call> blocks ──
        def _collect_phase1(m: re.Match) -> str:
            promoted.append(m.group(0))
            phase_counts[0] += 1
            return ""

        cleaned = _EMBEDDED_TOOL_CALL_RE.sub(_collect_phase1, reasoning).strip()

        # ── Phase 2: body-without-open-tag in the Phase-1 remainder ───────
        # Runs even if Phase 1 already found blocks, because a second call's
        # open token may have been stripped while the first call was complete.
        if cleaned and _BODY_WITHOUT_OPEN_RE.match(cleaned):
            reconstructed = "<tool_call>" + cleaned
            promoted.append(reconstructed)
            phase_counts[1] += 1
            cleaned = ""

        # ── Phase 3: closing fragment ──────────────────────────────────────
        # Handles two sub-cases:
        #   3a. content has an unclosed <tool_call> — append fragment to close it
        #   3b. content is None (model stopped inside <think>, no </think> emitted)
        #       — promote the fragment so the tool parser can attempt recovery
        if cleaned and _CLOSING_FRAGMENT_RE.match(cleaned):
            if content and "<tool_call>" in content:
                n_open = content.count("<tool_call>")
                n_close = content.count("</tool_call>")
                if n_open > n_close:
                    content = content + cleaned
                    phase_counts[2] += 1
                    cleaned = ""
                    logger.debug(
                        "glm45 parser: Phase 3a — appended closing fragment to "
                        "unclosed content call (AIVLLM-229)"
                    )
                    if not promoted:
                        return None, content
            elif not content:
                # 3b: no content at all — promote fragment so tool parser sees it
                promoted.append(cleaned)
                phase_counts[2] += 1
                cleaned = ""

        if not promoted:
            return reasoning, content

        promoted_text = "\n".join(promoted)
        merged_content = (
            promoted_text if not content else promoted_text + "\n" + content
        )

        logger.debug(
            "glm45 parser: promoted tool-call block(s) from reasoning_content "
            "into content — phase1=%d phase2=%d phase3=%d (AIVLLM-229)",
            phase_counts[0],
            phase_counts[1],
            phase_counts[2],
        )

        return cleaned or None, merged_content

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        reasoning, content = self._parser.extract_reasoning(model_output, request)
        logger.debug(
            "glm45 PRE-FIX | model_output=%r | reasoning=%r | content=%r",
            model_output[:300],
            reasoning[:300] if reasoning else reasoning,
            content[:300] if content else content,
        )
        result = self._split_embedded_tool_calls(reasoning, content)
        logger.debug(
            "glm45 POST-FIX | reasoning=%r | content=%r",
            result[0][:300] if result[0] else result[0],
            result[1][:300] if result[1] else result[1],
        )
        return result
