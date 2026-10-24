# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterator, Sequence

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import is_complete_json, partial_tag_overlap
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


class Hermes2ProToolParser(ToolParser):
    structural_tag_model = "hermes"
    # Flush a held partial region when generation ends. This parser is not
    # engine-based, so the serving layer must opt in explicitly.
    flush_streaming_on_finish = True
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_regex = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
    )
    scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = tokenizer.tokenizer

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Streaming state: what has been sent to the client.
        self._sent_content_idx: int = 0
        # Index into the text after the last valid tool call that has already
        # been emitted as content. None until a later region stops being a call.
        self._suffix_sent_idx: int | None = None
        # Unclosed region whose body is not yet known to be a call or text.
        # ``_hold_from`` is the first index that stays buffered, including any
        # prose after the previous call.
        self._holding_partial_region: bool = False
        self._hold_from: int | None = None
        self._last_text: str = ""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[ToolCall] = []
        last_good_end: int | None = None
        malformed = False
        for _start, region_end, body, _closed in self._iter_tool_call_regions(
            model_output
        ):
            parsed = self._tool_call_from_body(body)
            if parsed is None:
                malformed = True
                break
            tool_calls.append(parsed)
            last_good_end = region_end

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        prefix_end = model_output.find(self.tool_call_start_token)
        if malformed and last_good_end is not None:
            content = model_output[:prefix_end] + model_output[last_good_end:]
        else:
            # Well-formed output keeps only the text before the first call.
            content = model_output[:prefix_end]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _iter_tool_call_regions(
        self, text: str
    ) -> Iterator[tuple[int, int, str, bool]]:
        """Yield ``(start, end, body, closed)`` for each tool-call region.

        ``body`` excludes a partial closing tag. ``end`` is the index after a
        closed region, or ``len(text)`` while the region is still open.
        """
        pos = 0
        start_token = self.tool_call_start_token
        end_token = self.tool_call_end_token
        while True:
            start = text.find(start_token, pos)
            if start == -1:
                return
            body_start = start + len(start_token)
            end = text.find(end_token, body_start)
            if end != -1:
                region_end = end + len(end_token)
                yield start, region_end, text[body_start:end], True
                pos = region_end
                continue
            raw = text[body_start:]
            overlap = partial_tag_overlap(raw, end_token)
            if overlap:
                raw = raw[:-overlap]
            yield start, len(text), raw, False
            return

    def _tool_call_from_body(self, body: str) -> ToolCall | None:
        """Parse one region. None means this region is not a tool call."""
        stripped = body.strip()
        if not stripped:
            return None
        try:
            function_call = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(function_call, dict)
            or not isinstance(function_call.get("name"), str)
            or "arguments" not in function_call
        ):
            return None
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_call["name"],
                arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
            ),
        )

    def _classify_stream_region(self, body: str, closed: bool, index: int) -> str:
        """Return ``tool``, ``text``, or ``pending`` for a streaming region.

        A region is text as soon as its body does not start with ``{``, or
        once it is closed without a parseable call. An empty unclosed tag
        stays pending so a later JSON payload can still arrive.
        """
        stripped = body.strip()
        if not stripped:
            return "text" if closed else "pending"
        if not stripped.startswith("{"):
            return "text"
        already_named = (
            index < len(self.prev_tool_call_arr)
            and "name" in self.prev_tool_call_arr[index]
        )
        complete = closed or is_complete_json(stripped)
        if complete:
            if self._tool_call_from_body(stripped) is not None or already_named:
                return "tool"
            return "text"
        if self._extract_tool_name(stripped) or already_named:
            return "tool"
        return "pending"

    def _take_suffix(self, current_text: str, start: int, end: int) -> str | None:
        """Return not-yet-sent text in ``current_text[start:end]``."""
        if self._suffix_sent_idx is None or self._suffix_sent_idx < start:
            self._suffix_sent_idx = start
        if end <= self._suffix_sent_idx:
            return None
        extra = current_text[self._suffix_sent_idx : end]
        self._suffix_sent_idx = end
        return extra or None

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial <tool_call> tag.
        """
        if self.tool_call_start_token not in current_text:
            overlap_length = partial_tag_overlap(
                current_text, self.tool_call_start_token
            )
            sendable_idx = len(current_text) - overlap_length
        else:
            sendable_idx = current_text.index(self.tool_call_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    @staticmethod
    def _extract_tool_name(tc_json: str) -> str | None:
        """Extract tool name, or None if the name isn't complete yet."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', tc_json)
        return match.group(1) if match else None

    @staticmethod
    def _extract_tool_args(tc_json: str, is_complete: bool) -> str | None:
        """Extract tool arguments from the tool call JSON.

        Given {"name": "f", "arguments": {"x": 1}}, returns '{"x": 1}'.
        When is_complete, strips the trailing '}' that closes the outer
        object (not the arguments). For partial JSON, returns as-is.
        """
        match = re.search(r'"arguments"\s*:\s*', tc_json)
        if not match:
            return None
        raw = tc_json[match.end() :]
        if is_complete:
            raw = raw.rstrip()
            if raw.endswith("}"):
                raw = raw[:-1].rstrip()
        return raw

    def _compute_args_diff(
        self, index: int, tc_json: str, is_complete: bool
    ) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        args = self._extract_tool_args(tc_json, is_complete)
        if args is None or len(args) <= len(self.streamed_args_for_tool[index]):
            return None
        diff = args[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args
        self.prev_tool_call_arr[index]["arguments"] = args
        return diff

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Incrementally stream tool call deltas from accumulated output.

        On each invocation, re-parses the full ``current_text`` to find
        ``<tool_call>`` regions, then diffs against previously sent state
        to emit only new content, tool names, or argument fragments.

        Returns a ``DeltaMessage`` containing either plain content (for
        text preceding any tool call) or one or more ``DeltaToolCall``
        entries, or ``None`` if there is nothing new to send yet."""
        try:
            self._last_text = current_text
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            tool_call_jsons: list[tuple[str, bool]] = []
            last_good_end: int | None = None
            self._holding_partial_region = False
            self._hold_from = None
            for index, (start, _region_end, body, closed) in enumerate(
                self._iter_tool_call_regions(current_text)
            ):
                kind = self._classify_stream_region(body, closed, index)
                if kind == "pending":
                    # Prose before this tag is part of the call when the call
                    # is valid, and part of the content when it is not. Hold
                    # both until the region is resolved.
                    self._holding_partial_region = True
                    self._hold_from = start if last_good_end is None else last_good_end
                    break
                if kind == "text":
                    self._holding_partial_region = False
                    self._hold_from = None
                    suffix_from = start if last_good_end is None else last_good_end
                    extra = self._take_suffix(
                        current_text, suffix_from, len(current_text)
                    )
                    if extra:
                        content = extra if not content else content + extra
                    break
                stripped = body.strip()
                is_complete = closed or is_complete_json(stripped)
                tool_call_jsons.append((stripped, is_complete))
                last_good_end = _region_end
                self._holding_partial_region = False
                self._hold_from = None
                # A region that was buffered and then became a call must not
                # be emitted again as content.
                if (
                    self._suffix_sent_idx is not None
                    and self._suffix_sent_idx < _region_end
                ):
                    self._suffix_sent_idx = _region_end
            tool_call_deltas: list[DeltaToolCall] = []

            for i, (tc_json, is_complete) in enumerate(tool_call_jsons):
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    name = self._extract_tool_name(tc_json)
                    if not name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = name
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tc_json, is_complete)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )

            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None

    def finish_streaming(self) -> DeltaMessage | None:
        """Emit text held back while an unclosed region was still unresolved."""
        if not self._holding_partial_region or self._hold_from is None:
            return None
        start = self._hold_from
        if self._suffix_sent_idx is not None:
            start = max(start, self._suffix_sent_idx)
        self._holding_partial_region = False
        self._hold_from = None
        if start >= len(self._last_text):
            return None
        extra = self._last_text[start:]
        self._suffix_sent_idx = len(self._last_text)
        if not extra:
            return None
        return DeltaMessage(content=extra)
