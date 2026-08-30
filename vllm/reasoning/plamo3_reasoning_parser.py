# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any

# Tag tokenization in the PLaMo-3 vocabulary
# Each "<|plamo:begin_NAME:plamo|>" or "<|plamo:end_NAME:plamo|>" tag is
# NOT a single token; it encodes as at least three tokens:
#   prefix token  +  name token(s)  +  suffix token
#   e.g. BEGIN_THINK_TAG = <|plamo:begin_ (256) + think (21279) + :plamo|> (258)
#        END_THINK_TAG   = <|plamo:end_  (257) + think (21279) + :plamo|> (258)
# The streaming parsers therefore hold back the trailing portion of the buffer
# whenever its tail is a prefix of such a tag (see compute_safe_until).
#
# The only exception is EOT_TAG ("<|plamo:tag|>"), which is a single token
# and therefore never appears partially in the buffer.

BEGIN_TOOL_REQUESTS_TAG = "<|plamo:begin_tool_requests:plamo|>"
END_TOOL_REQUESTS_TAG = "<|plamo:end_tool_requests:plamo|>"
BEGIN_TOOL_REQUEST_TAG = "<|plamo:begin_tool_request:plamo|>"
END_TOOL_REQUEST_TAG = "<|plamo:end_tool_request:plamo|>"
BEGIN_TOOL_NAME_TAG = "<|plamo:begin_tool_name:plamo|>"
END_TOOL_NAME_TAG = "<|plamo:end_tool_name:plamo|>"
BEGIN_TOOL_ARGS_TAG = (
    "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>"
)
END_TOOL_ARGS_TAG = "<|plamo:end_tool_arguments:plamo|>"
EOT_TAG = "<|plamo:tag|>"
BEGIN_THINK_TAG = "<|plamo:begin_think:plamo|>"
END_THINK_TAG = "<|plamo:end_think:plamo|>"

_ALL_SPECIAL_TAGS: list[str] = [
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_REQUESTS_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    END_TOOL_REQUEST_TAG,
    BEGIN_TOOL_NAME_TAG,
    END_TOOL_NAME_TAG,
    "<|plamo:begin_tool_arguments:plamo|>",
    "<|plamo:constrain|>",
    "<|plamo:msg|>",
    END_TOOL_ARGS_TAG,
    EOT_TAG,
    BEGIN_THINK_TAG,
    END_THINK_TAG,
]

_SPECIAL_TOKEN_PREFIX = "<|plamo:"


def strip_trailing_partial_marker(text: str) -> str:
    """Strip a trailing incomplete PLaMo-3 special-token fragment from text."""
    idx = text.rfind(_SPECIAL_TOKEN_PREFIX)
    if idx == -1:
        return text
    tail = text[idx:]
    for tag in _ALL_SPECIAL_TAGS:
        if tag.startswith(tail) and tail != tag:
            return text[:idx]
    return text


def strip_at_eot(text: str) -> str:
    return text.split(EOT_TAG, maxsplit=1)[0]


def compute_safe_until(buf: str, floor: int, tags: list[tuple[str, str]]) -> int:
    """Compute the maximum buffer index that is safe to flush.

    Holds back the tail of `buf` if it matches any tag prefix.
    Uses a fast `rfind` for long prefixes (Step 1) and a fallback
    `endswith` check for prefixes shorter than the anchor (Step 2).
    """
    buf_len = len(buf)
    max_hold = 0
    for tag, anchor in tags:
        # Anchor must be a prefix of tag for correct fallback slicing.
        assert len(anchor) <= len(tag) and tag.startswith(anchor), (
            f"anchor {anchor!r} must be a prefix of tag {tag!r}"
        )
        anchor_len = len(anchor)
        check_len = min(len(tag) - 1, buf_len)
        # Step 1: Fast search for partial tags >= anchor_len.
        if check_len >= anchor_len:
            search_end = buf_len
            search_start = buf_len - check_len
            while True:
                p = buf.rfind(anchor, search_start, search_end)
                if p == -1:
                    break
                k = buf_len - p
                if buf[p:] == tag[:k]:
                    if k > max_hold:
                        max_hold = k
                    break
                search_end = p
        # Step 2: Fallback for prefixes < anchor_len.
        # Skip if a longer match was already found.
        if max_hold < anchor_len:
            max_short = min(anchor_len - 1, buf_len)
            for k in range(max_short, 0, -1):
                if buf.endswith(tag[:k]):
                    if k > max_hold:
                        max_hold = k
                    break
    safe_until = max(buf_len - max_hold, floor)
    assert safe_until <= buf_len, f"floor={floor} exceeds buffer length={buf_len}"
    return safe_until


class ReasoningParserStreamPhase(Enum):
    BEFORE_REASONING = "before_reasoning"
    IN_REASONING = "in_reasoning"
    AFTER_REASONING = "after_reasoning"
    DONE = "done"


class Plamo3ReasoningParser(ReasoningParser):
    @property
    def reasoning_start_str(self) -> str:
        return BEGIN_THINK_TAG

    @property
    def reasoning_end_str(self) -> str:
        return END_THINK_TAG

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}

        self._identity_parser: IdentityReasoningParser | None = None
        if not chat_template_kwargs.get("enable_thinking", True):
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

        self._begin_think_token_ids: list[int] = list(
            tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
        )
        self._end_think_token_ids: list[int] = list(
            tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
        )
        if not self._begin_think_token_ids or not self._end_think_token_ids:
            raise ValueError(
                "PLaMo3 reasoning parser failed to tokenize think tags: "
                f"{BEGIN_THINK_TAG!r} -> {self._begin_think_token_ids}, "
                f"{END_THINK_TAG!r} -> {self._end_think_token_ids}."
            )
        # Streaming state.
        self._stream_phase: ReasoningParserStreamPhase = (
            ReasoningParserStreamPhase.BEFORE_REASONING
        )
        self._stream_emit_pos: int = 0
        self._identity_stream_terminated: bool = False
        # Some vLLM call paths pass only delta token IDs to extract_content_ids.
        # Retain the complete stream only for that extraction.
        self._stream_token_ids: list[int] = []

    @staticmethod
    def _tokens_match_at(
        input_ids: Sequence[int], token_ids: Sequence[int], offset: int
    ) -> bool:
        if offset < 0 or offset + len(token_ids) > len(input_ids):
            return False
        for i, token_id in enumerate(token_ids):
            if input_ids[offset + i] != token_id:
                return False
        return True

    def _find_seq(
        self,
        seq: Sequence[int],
        target: list[int],
        start: int = 0,
        reverse: bool = False,
    ) -> int:
        n = len(target)
        if not n or len(seq) < n:
            return -1
        search_range = (
            range(len(seq) - n, -1, -1) if reverse else range(start, len(seq) - n + 1)
        )
        for i in search_range:
            if self._tokens_match_at(seq, target, i):
                return i
        return -1

    def _effective_input_ids(self, input_ids: Sequence[int]) -> list[int]:
        return self._stream_token_ids or list(input_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)
        if not input_ids:
            return False
        if (
            last_end := self._find_seq(
                input_ids, self._end_think_token_ids, reverse=True
            )
        ) == -1:
            return False
        if (
            last_begin := self._find_seq(
                input_ids, self._begin_think_token_ids, reverse=True
            )
        ) == -1:
            return True
        return last_end > last_begin

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )
        # Scan window to check if END_THINK completes in this step.
        n = len(self._end_think_token_ids)
        delta_start = len(input_ids) - len(list(delta_ids))
        window_start = max(0, delta_start - (n - 1))
        return (
            self._find_seq(
                input_ids,
                self._end_think_token_ids,
                start=window_start,
            )
            != -1
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        ids = self._effective_input_ids(input_ids)
        end_start = self._find_seq(ids, self._end_think_token_ids)
        if end_start == -1:
            return []

        return ids[end_start + len(self._end_think_token_ids) :]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._identity_parser is not None:
            if self._identity_stream_terminated:
                return None
            if EOT_TAG in delta_text:
                self._identity_stream_terminated = True
                delta_text = strip_at_eot(delta_text)
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # Keep accumulated token IDs available because some vLLM call paths
        # pass only delta token IDs to extract_content_ids.
        self._stream_token_ids = list(current_token_ids)

        while True:
            if self._stream_phase == ReasoningParserStreamPhase.BEFORE_REASONING:
                if not current_text:
                    break
                if current_text.startswith(BEGIN_THINK_TAG):
                    self._stream_emit_pos = len(BEGIN_THINK_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.IN_REASONING
                    continue
                # Wait if BEGIN_THINK_TAG is partially generated.
                if current_text and BEGIN_THINK_TAG.startswith(current_text):
                    break
                # Treat everything as reasoning until END_THINK_TAG to cover the case
                # where BEGIN_THINK_TAG is included in the chat template.
                self._stream_phase = ReasoningParserStreamPhase.IN_REASONING
                continue

            if self._stream_phase == ReasoningParserStreamPhase.IN_REASONING:
                search_offset = self._stream_emit_pos
                text_end = len(current_text)
                end_tag_start = current_text.find(END_THINK_TAG, search_offset)
                end_tag_start = text_end if end_tag_start == -1 else end_tag_start
                eot_start = current_text.find(EOT_TAG, search_offset)
                eot_start = text_end if eot_start == -1 else eot_start
                # If EOT precedes END_THINK: emit up to EOT and finish.
                if eot_start < end_tag_start:
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                    if reasoning_delta := current_text[search_offset:eot_start]:
                        return DeltaMessage(reasoning=reasoning_delta)
                    break
                # If neither END_THINK_TAG nor EOT_TAG found: emit up to safe position.
                if end_tag_start == text_end:
                    safe_until = compute_safe_until(
                        current_text,
                        search_offset,
                        [(END_THINK_TAG, "<|plamo:end_")],
                    )
                    if safe_until > search_offset:
                        self._stream_emit_pos = safe_until
                        return DeltaMessage(
                            reasoning=current_text[search_offset:safe_until]
                        )
                    break
                # If END_THINK found: split into reasoning and content.
                if eot_start != text_end:
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                else:
                    self._stream_emit_pos = text_end
                    self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                end_tag_end = end_tag_start + len(END_THINK_TAG)
                reasoning_delta = current_text[search_offset:end_tag_start]
                content_delta = current_text[end_tag_end:eot_start]
                if reasoning_delta or content_delta:
                    return DeltaMessage(
                        reasoning=reasoning_delta or None,
                        content=content_delta or None,
                    )
                break

            if self._stream_phase == ReasoningParserStreamPhase.AFTER_REASONING:
                eot_start = current_text.find(EOT_TAG, self._stream_emit_pos)
                if eot_start != -1:
                    delta = current_text[self._stream_emit_pos : eot_start]
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                else:
                    delta = current_text[self._stream_emit_pos :]
                    self._stream_emit_pos = len(current_text)
                if delta:
                    return DeltaMessage(content=delta)
                break

            if self._stream_phase == ReasoningParserStreamPhase.DONE:
                break

        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._identity_parser is not None:
            reasoning, content = self._identity_parser.extract_reasoning(
                model_output, request
            )
            return reasoning, strip_at_eot(content) if content is not None else None

        model_output = strip_at_eot(model_output)
        begin_tag_end = (
            len(BEGIN_THINK_TAG) if model_output.startswith(BEGIN_THINK_TAG) else 0
        )
        end_tag_start = model_output.find(END_THINK_TAG, begin_tag_end)
        if end_tag_start == -1:
            reasoning = strip_trailing_partial_marker(model_output[begin_tag_end:])
            return reasoning or None, None

        end_tag_end = end_tag_start + len(END_THINK_TAG)
        reasoning = model_output[begin_tag_end:end_tag_start]
        content = strip_trailing_partial_marker(model_output[end_tag_end:])
        return reasoning, content
