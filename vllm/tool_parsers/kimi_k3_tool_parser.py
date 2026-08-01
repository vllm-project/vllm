# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parser for the Kimi K3 (XTML) chat format.

This turns the generated XTML ``response`` and ``tools`` channels back into
OpenAI-compatible ``content`` and ``tool_calls``.

K3 assistant tool calls live in a nested ``tools`` channel::

    <|open|>tools<|sep|>
      <|open|>call tool="python" index="1"<|sep|>
        <|open|>argument key="code" type="string"<|sep|>print(1)<|close|>argument<|sep|>
        <|open|>argument key="opts" type="object"<|sep|>{"a":1}<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

where ``<|open|>``, ``<|close|>``, ``<|sep|>`` are dedicated special tokens. The plain
reply lives in a sibling ``response`` channel which is unwrapped into content.
``response`` always precedes ``tools`` (see the protocol channel grammar).

Argument decoding mirrors the template's type tagging (inverse encoding):
  * type="string"  -> value is the RAW text, no unescaping (template emits it raw)
  * other types    -> value is JSON-decoded (number/boolean/null/object/array)

Attribute values (``tool=``, ``index=``, ``key=``, ``type=``) ARE escaped on the
encode side (``&`` -> ``&amp;``, ``"`` -> ``&quot;``), so ``_attrs`` reverses
them on decode (``&quot;`` before ``&amp;`` -- reverse of encode order).

Known limitation: because string argument and response bodies are emitted raw,
a value that literally contains ``<|close|>argument<|sep|>`` or
``<|close|>response<|sep|>`` is indistinguishable from a real closing marker.
"""

import json
from collections.abc import Sequence

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

_O, _C, _S = r"<\|open\|>", r"<\|close\|>", r"<\|sep\|>"
_TEXT_UNTIL_SEP = r"(?:(?!" + _S + r").)*?"


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class KimiK3ToolParser(ToolParser):
    supports_required_and_named = False
    # Enables the vLLM-side XTML structural tag builder
    # (``get_kimi_k3_structural_tag`` in ``structural_tag_registry``). With
    # ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` on (default), ``_apply_structural_tag``
    # constrains generation to K3's ``<|open|>tools<|sep|>`` channel for
    # ``required`` (and ``auto`` when a tool sets ``strict``) instead of the
    # generic JSON guided decoding, which conflicts with the XTML format.
    structural_tag_model = "kimi_k3"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tools_open = "<|open|>tools<|sep|>"
        self.tools_close = "<|close|>tools<|sep|>"
        self.response_open = "<|open|>response<|sep|>"
        self.response_close = "<|close|>response<|sep|>"

        # Regexes operate on detokenized text. The XTML markers reach us as the
        # literal strings <|open|>/<|close|>/<|sep|>. adjust_request keeps them
        # from being stripped and suppresses spacing between adjacent token
        # pieces. As DEFENSE-IN-DEPTH we still tolerate optional whitespace
        # WITHIN a marker (e.g. "<|open|> tools <|sep|>") in case some serving
        # path leaves vLLM's added-token spacing on -- this `\s*` is a no-op on
        # clean input, so the normal byte-exact path is unaffected. We do NOT
        # strip body whitespace, so clean content stays byte-exact.
        # All bodies use non-greedy (.*?) so each block stops at its own first
        # closing marker -- see the module-level KNOWN LIMITATION about
        # literal-marker values.
        self._tools_open_re = re.compile(_O + r"\s*tools\s*" + _S)
        self._tools_close_re = re.compile(_C + r"\s*tools\s*" + _S)
        self._response_open_re = re.compile(_O + r"\s*response\s*" + _S)
        self._response_close_re = re.compile(_C + r"\s*response\s*" + _S)
        self._message_close_re = re.compile(_C + r"\s*message\s*" + _S)
        self._call_re = re.compile(
            _O
            + r"\s*call\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _S
            + r"(?P<body>.*?)"
            + _C
            + r"\s*call\s*"
            + _S,
            re.DOTALL,
        )
        self._arg_re = re.compile(
            _O
            + r"\s*argument\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _S
            + r"(?P<val>.*?)"
            + _C
            + r"\s*argument\s*"
            + _S,
            re.DOTALL,
        )
        # attr segment: key="value" (value already escaped on the encode side)
        self._attr_re = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')
        self._response_re = re.compile(
            _O + r"\s*response\s*" + _S + r"(?P<c>.*?)" + _C + r"\s*response\s*" + _S,
            re.DOTALL,
        )

        # streaming state
        self._sent_content_idx = 0
        self._sent_tool_call_count = 0

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        structured_outputs = getattr(request, "structured_outputs", None)
        has_structural_tag = (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        )
        if named and not has_structural_tag:
            # Without the XTML structural tag there is no way to force the
            # named call (the generic JSON guided-decoding path conflicts
            # with the XTML channel format).
            raise VLLMValidationError(
                "Named tool choice for Kimi K3 requires strict tool calling "
                "(VLLM_ENFORCE_STRICT_TOOL_CALLING) so the XTML structural "
                "tag can force the call. Otherwise use `tool_choice` set to "
                '"auto", "required", or "none".',
                parameter="tool_choice",
                value=request.tool_choice,
            )

        if request.tools and (request.tool_choice == "required" or named):
            # K3 emits tool calls in XTML. When strict tool calling is enabled,
            # DelegatingParser._apply_structural_tag has already attached the K3
            # XTML structural tag (structural_tag_model = "kimi_k3"). We return
            # early to skip the generic parent path, which would otherwise attach
            # JSON guided decoding when strict calling is off -- that JSON
            # constraint conflicts with the <|open|>tools<|sep|> channel.
            request.skip_special_tokens = False
            if hasattr(request, "spaces_between_special_tokens"):
                request.spaces_between_special_tokens = False
            return request

        request = super().adjust_request(request)
        # The XTML markers (<|open|>/<|close|>/<|sep|>,
        # <|open|>response<|sep|> ...) must
        # reach this parser as CONTIGUOUS literal text. Two request flags govern
        # that, and vLLM's detokenizer couples them:
        #   effective_spaces_between =
        #       skip_special_tokens OR spaces_between_special_tokens
        #   (vllm/v1/engine/detokenizer.py).
        # The detokenizer treats these control tokens as separate sub-texts and,
        # when the effective flag is True, joins them with spaces ->
        # "<|open|> response <|sep|>", which the regexes below would NOT match.
        # Forcing BOTH flags off is the only way to suppress that spacing. We
        # set both unconditionally: the response channel is unwrapped from these
        # markers even with no tools.
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def _attrs(self, s: str) -> dict[str, str]:
        return {
            m["k"]: m["v"].replace("&quot;", '"').replace("&amp;", "&")
            for m in self._attr_re.finditer(s)
        }

    def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
        """Decode one ``call`` block into a :class:`ToolCall`.

        ``attrs`` is the text between ``<|open|>call `` and ``<|sep|>`` (carries
        ``tool=`` / ``index=``); ``body`` is the sequence of ``argument`` blocks.
        Each argument is re-typed per its ``type=`` tag: strings pass through
        raw, everything else is JSON-decoded (falling back to raw text if the
        JSON is malformed, so a partial stream never raises). Returns ``None``
        when no tool name is present (an empty/garbage block is dropped).
        """
        call_attrs = self._attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        arguments: dict = {}
        for arg_match in self._arg_re.finditer(body):
            arg_attrs = self._attrs(arg_match["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg_match["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        if not tool_name:
            return None
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _strip_response_content(self, text: str) -> str | None:
        """Strip XTML response/message markers from generated response text.

        In chat serving, ``<|open|>response<|sep|>`` is often part of the prompt
        generation prefix, so the model output may only contain the body plus
        ``<|close|>response<|sep|>``. Handle both that consumed-prefix shape and a
        complete ``<|open|>response<|sep|>... <|close|>response<|sep|>`` wrapper.
        """
        m_open = self._response_open_re.search(text)
        if m_open is not None:
            m_close = self._response_close_re.search(text, m_open.end())
            if m_close is not None:
                text = text[m_open.end() : m_close.start()]
            else:
                text = text[m_open.end() :]
        else:
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        return text or None

    def _content(self, model_output: str, before: str) -> str | None:
        # prefer the unwrapped response channel; else the text before the tools
        m = self._response_re.search(model_output)
        if m is not None:
            return m["c"] or None
        return self._strip_response_content(before)

    def _extract_response_content(self, current_text: str) -> str | None:
        # Streaming response text is computed from the accumulated text. This is
        # what keeps split markers from leaking:
        #   <|open|> / response / <|sep|>Hi     -> emit only "Hi" after open closes
        #   Hi<|open|> / tools / <|sep|>...     -> emit "Hi", hold the tools marker
        #   Hi<|close|> / response / <|sep|>... -> emit "Hi", hold the close marker
        # Tool calls are even simpler: they are not emitted until a full
        # <|close|>call<|sep|> is present in current_text.
        m_open = self._response_open_re.search(current_text)
        # In the normal chat path, the response-open marker may be consumed as
        # the generation prefix. Then generated text starts directly with the
        # response body or with <|close|>response<|sep|> before a tools channel.
        body_start = m_open.end() if m_open is not None else 0
        m_tools = self._tools_open_re.search(current_text, body_start)
        m_rclose = self._response_close_re.search(current_text, body_start)
        tools_start = m_tools.start() if m_tools else -1
        response_end = m_rclose.start() if m_rclose else -1

        candidates = [i for i in (tools_start, response_end) if i != -1]
        if candidates:
            sendable_idx = min(candidates)
        else:
            overlap = max(
                _partial_tag_overlap(current_text, self.response_open),
                _partial_tag_overlap(current_text, self.response_close),
                _partial_tag_overlap(current_text, self.tools_open),
            )
            sendable_idx = len(current_text) - overlap

        if sendable_idx <= body_start:
            return None
        if self._sent_content_idx < body_start:
            self._sent_content_idx = body_start
        if sendable_idx <= self._sent_content_idx:
            return None

        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content or None

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        m_open = self._tools_open_re.search(model_output)
        if m_open is None:
            # no tools channel -> content is the response channel (unwrapped)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._content(model_output, model_output),
            )
        try:
            before = model_output[: m_open.start()]
            start = m_open.end()
            m_close = self._tools_close_re.search(model_output, start)
            section = (
                model_output[start:]
                if m_close is None
                else model_output[start : m_close.start()]
            )

            tool_calls = [
                tc
                for m in self._call_re.finditer(section)
                if (tc := self._decode_call(m["attrs"], m["body"])) is not None
            ]
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=self._content(model_output, before),
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=self._content(model_output, before),
            )
        except Exception:
            logger.exception("Error extracting K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

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
        # Conservative streaming: stream unwrapped response-channel text, then
        # buffer tool calls and emit each call once its block closes.
        content = self._extract_response_content(current_text)

        # tools channel is open: parse fully-closed calls we have not emitted yet
        m_tools = self._tools_open_re.search(current_text)
        if m_tools is None:
            return DeltaMessage(content=content) if content else None

        section = current_text[m_tools.end() :]
        calls = [
            tc
            for m in self._call_re.finditer(section)
            if (tc := self._decode_call(m["attrs"], m["body"])) is not None
        ]
        if len(calls) <= self._sent_tool_call_count:
            return DeltaMessage(content=content) if content else None
        new = calls[self._sent_tool_call_count :]

        deltas = [
            DeltaToolCall(
                index=self._sent_tool_call_count + i,
                id=tc.id,
                type="function",
                function=DeltaFunctionCall(
                    name=tc.function.name, arguments=tc.function.arguments
                ).model_dump(exclude_none=True),
            )
            for i, tc in enumerate(new)
        ]
        self._sent_tool_call_count = len(calls)
        return DeltaMessage(content=content, tool_calls=deltas)
