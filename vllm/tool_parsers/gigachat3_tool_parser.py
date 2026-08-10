# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

REGEX_FUNCTION_CALL = re.compile(
    r"(?:function call<\|role_sep\|>\n|<\|function_call\|>)(.*)",
    re.DOTALL,
)

REGEX_CONTENT_PATTERN = re.compile(
    r"^(.*?)(?:<\|message_sep\|>|<\|function_call\|>)",
    re.DOTALL,
)

NAME_REGEX = re.compile(
    r'"name"\s*:\s*"([^"]*)"',
    re.DOTALL,
)

ARGS_REGEX = re.compile(
    r'"arguments"\s*:\s*(.*)',
    re.DOTALL,
)

EOS_TOKEN = "</s>"
CONTENT_STOP_TOKENS = ("<|message_sep|>", "<|function_call|>")
CONTENT_HOLD_TOKENS = (*CONTENT_STOP_TOKENS, EOS_TOKEN)


class GigaChat3ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.tool_name_sent: bool = False
        self.tool_id: str | None = None
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        # Content scanner state: text before _content_consumed has been
        # emitted (or dropped as special tokens); once _content_done is set,
        # everything after belongs to the function call.
        self._content_consumed: int = 0
        self._content_done: bool = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        function_call = None
        content = None
        if model_output.rstrip().endswith("</s>"):
            model_output = model_output[: model_output.rfind("</s>")]
        m_func = REGEX_FUNCTION_CALL.search(model_output)
        if m_func:
            try:
                function_call = json.loads(m_func.group(1), strict=False)
                if (
                    isinstance(function_call, dict)
                    and "name" in function_call
                    and "arguments" in function_call
                ):
                    if not isinstance(function_call["arguments"], dict):
                        function_call = None
                else:
                    function_call = None
            except json.JSONDecodeError:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
        m_content = REGEX_CONTENT_PATTERN.search(model_output)
        content = m_content.group(1) if m_content else model_output
        if not function_call:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content if content else None,
            )
        name = function_call["name"]
        args = function_call["arguments"]
        if not isinstance(args, str):
            args = json.dumps(function_call["arguments"], ensure_ascii=False)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=args,
                    ),
                )
            ],
            content=content if content else None,
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
        func_name = None
        cur_args = None
        if not self._content_done:
            content = self._stream_content(current_text)
            if content:
                return DeltaMessage(content=content)
            if not self._content_done:
                return None
        m_func = REGEX_FUNCTION_CALL.search(current_text)
        if not m_func:
            return None
        json_tail = m_func.group(1).strip()
        name_match = NAME_REGEX.search(json_tail)
        if name_match:
            func_name = name_match.group(1)
        args_match = ARGS_REGEX.search(json_tail)
        if args_match:
            cur_args = args_match.group(1).strip()
            # The regex captures to the end of the payload, so cur_args may
            # end with text that is not part of the arguments yet: a partial
            # or complete EOS token, and the enclosing object's closing
            # brace. Hold those back until later text resolves them, so the
            # streamed prefix never has to shrink.
            if cur_args.endswith(EOS_TOKEN):
                cur_args = cur_args[: -len(EOS_TOKEN)]
            else:
                for k in range(len(EOS_TOKEN) - 1, 0, -1):
                    if cur_args.endswith(EOS_TOKEN[:k]):
                        cur_args = cur_args[:-k]
                        break
            cur_args = cur_args.rstrip()
            if cur_args.endswith("}"):
                cur_args = cur_args[:-1]
        if not self.prev_tool_call_arr:
            self.prev_tool_call_arr.append({})
        if not self.tool_name_sent:
            if not func_name:
                return None
            self.tool_name_sent = True
            self.tool_id = make_tool_call_id()
            self.prev_tool_call_arr[0]["name"] = func_name
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id=self.tool_id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=func_name,
                        ).model_dump(exclude_none=True),
                    )
                ],
            )
        if cur_args is None:
            return None
        prev_args = self.prev_tool_call_arr[0].get("arguments_str", "")
        if not prev_args:
            delta_args = cur_args
        elif cur_args.startswith(prev_args):
            delta_args = cur_args[len(prev_args) :]
        else:
            return None
        if not delta_args:
            return None
        self.prev_tool_call_arr[0]["arguments_str"] = cur_args
        try:
            args_dict = json.loads(cur_args, strict=False)
            self.prev_tool_call_arr[0]["arguments"] = args_dict
        except json.JSONDecodeError:
            self.prev_tool_call_arr[0]["arguments"] = {}
        if len(self.streamed_args_for_tool) <= 0:
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[0] = cur_args
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(
                        arguments=delta_args,
                    ).model_dump(exclude_none=True),
                )
            ],
        )

    def _stream_content(self, current_text: str) -> str:
        """Emit chat content up to the first special token.

        Works on the accumulated text rather than a single delta, so special
        tokens split across deltas are still recognized. A tail that could
        still become a special token is held back until disambiguated.
        """
        out: list[str] = []
        while not self._content_done and self._content_consumed < len(current_text):
            seg = current_text[self._content_consumed :]
            lt = seg.find("<")
            if lt == -1:
                out.append(seg)
                self._content_consumed += len(seg)
                break
            if lt > 0:
                out.append(seg[:lt])
                self._content_consumed += lt
                continue
            if seg.startswith(CONTENT_STOP_TOKENS):
                self._content_done = True
                break
            if seg.startswith(EOS_TOKEN):
                # Dropped from content, as in non-streaming extraction.
                self._content_consumed += len(EOS_TOKEN)
                continue
            if any(token.startswith(seg) for token in CONTENT_HOLD_TOKENS):
                break
            out.append("<")
            self._content_consumed += 1
        return "".join(out)
