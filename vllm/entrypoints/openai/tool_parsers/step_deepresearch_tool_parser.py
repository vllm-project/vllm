# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import contextlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

# XML Namespace and tag definitions
NAMESPACE = "steptml"
STEPTML_TOOLCALL_TAG = f"{NAMESPACE}:toolcall"
STEPTML_INVOKE_TAG = f"{NAMESPACE}:invoke"
STEPTML_PARAM_TAG = f"{NAMESPACE}:parameter"


def _parse_bool(x: Any) -> bool:
    """Parse boolean from string or other types."""
    if isinstance(x, str):
        return x.lower() == "true"
    return bool(x)


# Type alias for parsed invoke result
ParsedInvoke = tuple[Optional[str], Optional[dict[str, str]]]

# Type mapping for argument casting
TYPE_MAPPING: dict[str, Callable[[Any], Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": _parse_bool,
    "object": json.loads,
    "array": json.loads,
}


@ToolParserManager.register_module(["step_deepresearch"])
class StepDeepResearchToolParser(ToolParser):
    """
    Tool parser for steptml XML format used by Step models.

    This parser handles tool calls in the following XML format::

        <steptml:toolcall>
        <steptml:invoke name="function_name">
        <steptml:parameter name="param1">value1</steptml:parameter>
        <steptml:parameter name="param2">value2</steptml:parameter>
        </steptml:invoke>
        </steptml:toolcall>

    Features:
        - Supports multiple tool invocations in a single toolcall block
        - Streaming: sends function name immediately, arguments when complete
        - Automatic type casting based on tool schema definitions
        - Handles content before and after tool blocks
    """

    TOOLCALL_START = f"<{STEPTML_TOOLCALL_TAG}>"
    TOOLCALL_END = f"</{STEPTML_TOOLCALL_TAG}>"
    INVOKE_START = f"<{STEPTML_INVOKE_TAG}"
    INVOKE_END = f"</{STEPTML_INVOKE_TAG}>"
    SPECIAL_TOKENS = [TOOLCALL_START, TOOLCALL_END]

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.position = 0
        # State flags for streaming
        self.tool_block_started = False
        self.tool_block_finished = False
        # For streaming individual tool calls
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the request to ensure special tokens are not skipped."""
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _parse_steptml_invoke(
        text: str, ) -> tuple[Optional[str], Optional[dict[str, str]]]:
        """
        Parse a single steptml:invoke block using regex.

        Returns:
            (function_name, params_dict) or (None, None) if not found.
        """
        pattern = r'<steptml:invoke name="([^"]+)">'
        func_name_match = re.search(pattern, text)
        if not func_name_match:
            return None, None
        func_name = func_name_match.group(1)

        params: dict[str, str] = {}
        # Match parameters - handle single-line and multi-line content
        param_pattern = (r'<steptml:parameter name="([^"]+)">'
                         r"([\s\S]*?)"
                         r"</steptml:parameter>")
        param_matches = re.findall(param_pattern, text)
        for name, value in param_matches:
            params[name] = value.strip() if value else ""

        return func_name, params

    @staticmethod
    def _parse_all_invokes(
        xml_text: str,
    ) -> list[tuple[Optional[str], Optional[dict[str, str]]]]:
        """
        Parse all steptml:invoke blocks from a toolcall block.

        Returns:
            A list of (function_name, params_dict) tuples.
        """
        results: list[ParsedInvoke] = []
        # Find all invoke blocks
        invoke_pattern = (
            r'<steptml:invoke name="[^"]+"[\s\S]*?</steptml:invoke>')
        invoke_matches = re.findall(invoke_pattern, xml_text)

        for invoke_text in invoke_matches:
            func_name, params = StepDRToolParser._parse_steptml_invoke(
                invoke_text)
            if func_name:
                results.append((func_name, params))

        return results

    def _cast_arguments(
        self,
        func_name: str,
        params: dict[str, Any],
        request: ChatCompletionRequest,
    ) -> dict[str, Any]:
        """
        Cast argument types according to tool schema definition.

        Converts string values to their expected types (int, float, bool,
        array, object) based on the tool's parameter schema. For arrays,
        also handles flattening of quoted sub-items within string elements.
        """
        for tool in request.tools or []:
            if tool.function.name != func_name:
                continue
            schema = tool.function.parameters or {}
            properties = schema.get("properties", {})
            for key, value in list(params.items()):
                self._cast_single_param(params, properties, key, value)
            break
        return params

    def _cast_single_param(
        self,
        params: dict[str, Any],
        properties: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        """Cast a single parameter value based on its schema type."""
        prop = properties.get(key, {})
        typ = prop.get("type")
        if typ not in TYPE_MAPPING:
            return
        # Skip conversion if already correct type for object/array
        if typ == "object" and isinstance(value, dict):
            return
        if typ == "array" and isinstance(value, list):
            return
        if not isinstance(value, str):
            return

        type_constructor = TYPE_MAPPING[typ]
        try:
            parsed_val = type_constructor(value)
            if typ == "array" and isinstance(parsed_val, list):
                params[key] = self._flatten_array(parsed_val)
            else:
                params[key] = parsed_val
        except Exception:
            with contextlib.suppress(Exception):
                params[key] = ast.literal_eval(value)

    @staticmethod
    def _flatten_array(arr: list[Any]) -> list[Any]:
        """Flatten array elements that contain quoted sub-items."""
        new_list: list[Any] = []
        for elem in arr:
            if isinstance(elem, str):
                inner = re.findall(r'"([^"]+)"', elem)
                if inner:
                    new_list.extend(inner)
                else:
                    new_list.append(elem)
            else:
                new_list.append(elem)
        return new_list

    def _handle_before_tool_block(
            self, unprocessed_text: str,
            current_text: str) -> tuple[bool, Optional[DeltaMessage]]:
        """
        Handle state before tool block starts.

        Returns:
            (should_continue, delta_message)
        """
        if unprocessed_text.startswith(self.TOOLCALL_START):
            self.position += len(self.TOOLCALL_START)
            self.tool_block_started = True
            return True, None

        start_pos = unprocessed_text.find(self.TOOLCALL_START)
        if start_pos == -1:
            stripped = unprocessed_text.strip()
            if stripped and self.TOOLCALL_START.startswith(stripped):
                return False, None
            self.position = len(current_text)
            return False, DeltaMessage(content=unprocessed_text)

        content = unprocessed_text[:start_pos]
        self.position += len(content)
        if content:
            return False, DeltaMessage(content=content)
        return True, None

    def _handle_new_invoke_block(self,
                                 unprocessed_text: str) -> Optional[bool]:
        """
        Handle detection of new invoke block.

        Returns:
            True if should continue loop, False if should return None, None if
            should fall through to parsing.
        """
        tool_finished = self.current_tool_id != -1 and self.prev_tool_call_arr[
            self.current_tool_id].get("finished")

        if self.current_tool_id != -1 and not tool_finished:
            return None  # Fall through to parsing

        if unprocessed_text.startswith(self.INVOKE_START):
            if self.current_tool_id == -1:
                self.current_tool_id = 0
            else:
                self.current_tool_id += 1
            self.current_tool_name_sent = False
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            curr_tool = self.prev_tool_call_arr[self.current_tool_id]
            curr_tool["finished"] = False
            return None  # Fall through to parsing

        if self.INVOKE_START.startswith(unprocessed_text):
            return False  # Return None, partial invoke start

        if self.TOOLCALL_END.startswith(unprocessed_text):
            return False  # Return None, partial toolcall end

        return None

    def _check_partial_end_tag(self, text: str) -> bool:
        """Check if text ends with partial end tag."""
        for i in range(1, len(self.INVOKE_END)):
            if text.endswith(self.INVOKE_END[:i]):
                return True
        for i in range(1, len(self.TOOLCALL_END)):
            if text.endswith(self.TOOLCALL_END[:i]):
                return True
        return False

    def _parse_active_tool_call(
        self,
        unprocessed_text: str,
        request: ChatCompletionRequest,
    ) -> Optional[DeltaMessage]:
        """Parse an active tool call and return delta message if ready."""
        end_invoke_pos = unprocessed_text.find(self.INVOKE_END)
        if end_invoke_pos == -1:
            tool_body = unprocessed_text
        else:
            end_pos = end_invoke_pos + len(self.INVOKE_END)
            tool_body = unprocessed_text[:end_pos]

        if end_invoke_pos == -1 and self._check_partial_end_tag(
                unprocessed_text):
            return None

        func_name, arguments = self._parse_steptml_invoke(tool_body)
        if not func_name:
            return None

        tool_call_arr: dict[str, Any] = {
            "name": func_name,
            "parameters": arguments or {},
        }

        curr_tool = self.prev_tool_call_arr[self.current_tool_id]
        curr_tool.update(tool_call_arr)

        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            delta_tool_call = DeltaToolCall(
                index=self.current_tool_id,
                type="function",
                id=f"chatcmpl-tool-{random_uuid()}",
                function=DeltaFunctionCall(name=func_name),
            )
            return DeltaMessage(tool_calls=[delta_tool_call])

        if end_invoke_pos != -1:
            self.position += end_invoke_pos + len(self.INVOKE_END)
            curr_tool["finished"] = True
            params: dict[str, Any] = tool_call_arr.get("parameters", {})
            final_args = self._cast_arguments(func_name, params, request)
            final_args_json = json.dumps(final_args, ensure_ascii=False)
            delta_tool_call = DeltaToolCall(
                index=self.current_tool_id,
                function=DeltaFunctionCall(arguments=final_args_json),
            )
            return DeltaMessage(tool_calls=[delta_tool_call])

        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Extract tool calls from streaming output.

        Uses a state machine to track parsing progress:
            - Before tool block: output content as-is
            - Inside tool block: parse invoke blocks
            - After tool block: output remaining content

        Sends function name when parsed, arguments when invoke is complete.
        """
        while True:
            if self.position >= len(current_text):
                return None

            unprocessed_text = current_text[self.position:]

            if self.tool_block_finished:
                self.position = len(current_text)
                return DeltaMessage(content=unprocessed_text)

            if not self.tool_block_started:
                should_continue, delta = self._handle_before_tool_block(
                    unprocessed_text, current_text)
                if should_continue:
                    continue
                return delta

            # Inside tool block - skip whitespace
            offset = len(unprocessed_text) - len(unprocessed_text.lstrip())
            unprocessed_text = unprocessed_text.lstrip()
            self.position += offset

            if unprocessed_text.startswith(self.TOOLCALL_END):
                self.position += len(self.TOOLCALL_END)
                self.tool_block_finished = True
                self.current_tool_id = -1
                continue

            invoke_result = self._handle_new_invoke_block(unprocessed_text)
            if invoke_result is False:
                return None

            curr_tool = self.prev_tool_call_arr[self.current_tool_id]
            is_active = self.current_tool_id != -1 and not curr_tool.get(
                "finished", False)
            if is_active:
                result = self._parse_active_tool_call(unprocessed_text,
                                                      request)
                return result

            return None

    def _parse_tool_block(self, block: str,
                          request: ChatCompletionRequest) -> list[ToolCall]:
        """Parse a single tool block and return list of ToolCall objects."""
        tool_calls: list[ToolCall] = []
        parsed_tools = self._parse_all_invokes(block)
        for func_name, params in parsed_tools:
            if func_name and params is not None:
                casted_params = self._cast_arguments(func_name, params,
                                                     request)
                args_json = json.dumps(casted_params, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(function=FunctionCall(name=func_name,
                                                   arguments=args_json)))
        return tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from complete (non-streaming) model output.

        Parses the entire output, extracting all tool calls from toolcall
        blocks and returning any surrounding content separately.
        """
        if self.TOOLCALL_START not in model_output:
            final_content = model_output.strip() or model_output
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=final_content)

        # Split content by tool blocks
        start_escaped = re.escape(self.TOOLCALL_START)
        end_escaped = re.escape(self.TOOLCALL_END)
        pattern = rf"({start_escaped}.*?{end_escaped})"
        parts = re.split(pattern, model_output, flags=re.DOTALL)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in parts:
            is_tool_block = part.startswith(
                self.TOOLCALL_START) and part.endswith(self.TOOLCALL_END)
            if is_tool_block:
                tool_calls.extend(self._parse_tool_block(part, request))
            else:
                text_parts.append(part)

        final_content = "".join(text_parts).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=final_content if final_content else None,
            )

        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=model_output)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Note: Replace with your tokenizer path for testing
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_stepdr")
    parser = StepDRToolParser(tokenizer=tokenizer)

    # Mock request object with tool definitions
    request = ChatCompletionRequest(
        model="stepdr",
        messages=[{
            "role": "user",
            "content": "Search for information"
        }],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array"
                            },
                            "topk": {
                                "type": "integer"
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_url",
                    "description": "Open a URL and return content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string"
                            },
                        },
                    },
                },
            },
        ],
    )

    # Test Case 1: Non-streaming extraction with array parameter
    test_output_1 = ("<steptml:toolcall>\n"
                     '<steptml:invoke name="web_search">\n'
                     '<steptml:parameter name="query">'
                     '["AI news", "latest developments"]'
                     "</steptml:parameter>\n"
                     '<steptml:parameter name="topk">10</steptml:parameter>\n'
                     "</steptml:invoke>\n"
                     "</steptml:toolcall>")

    print("--- TEST 1: Non-streaming with array parameter ---")
    result = parser.extract_tool_calls(test_output_1, request)
    print(f"Tools called: {result.tools_called}")
    print(f"Content: {result.content}")
    for i, tc in enumerate(result.tool_calls):
        print(f"Tool {i}: {tc.function.name} - {tc.function.arguments}")

    print("\n" + "*" * 50)

    # Test Case 2: Streaming with content before/after tool block
    print("\n--- TEST 2: Streaming with content before/after ---")
    streaming_output = ("Let me search for that information...\n"
                        "<steptml:toolcall>\n"
                        '<steptml:invoke name="web_search">\n'
                        '<steptml:parameter name="query">'
                        '["machine learning", "deep learning"]'
                        "</steptml:parameter>\n"
                        "</steptml:invoke>\n"
                        '<steptml:invoke name="open_url">\n'
                        '<steptml:parameter name="url">'
                        "https://example.com/article"
                        "</steptml:parameter>\n"
                        "</steptml:invoke>\n"
                        "</steptml:toolcall>\n"
                        "Here is the summary.")

    ids = tokenizer.encode(streaming_output)
    stream_tokens = [tokenizer.decode(id) for id in ids]
    parser2 = StepDRToolParser(tokenizer=None)
    current_text = ""
    print("Streaming output:")
    for delta_text in stream_tokens:
        previous_text = current_text
        current_text += delta_text
        delta = parser2.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta:
            if delta.content:
                print(f"  Content: {repr(delta.content)}")
            if delta.tool_calls:
                for dtc in delta.tool_calls:
                    func = dtc.function
                    if func and func.name:
                        print(f"  ToolCall[{dtc.index}]: "
                              f"name={func.name}")
                    if func and func.arguments:
                        print(f"  ToolCall[{dtc.index}]: "
                              f"args={func.arguments}")

    # Test Case 3: Streaming with simple array
    print("\n--- TEST 3: Streaming with simple array ---")
    streaming_simple = ("<steptml:toolcall>\n"
                        '<steptml:invoke name="web_search">\n'
                        '<steptml:parameter name="query">'
                        '["python", "tutorial", "beginner", "examples"]'
                        "</steptml:parameter>\n"
                        "</steptml:invoke>\n"
                        "</steptml:toolcall>")
    ids = tokenizer.encode(streaming_simple)
    stream_tokens = [tokenizer.decode(id) for id in ids]
    parser3 = StepDRToolParser(tokenizer=None)
    current_text = ""
    print("Streaming output:")
    for delta_text in stream_tokens:
        previous_text = current_text
        current_text += delta_text
        delta = parser3.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta:
            if delta.content:
                print(f"  Content: {repr(delta.content)}")
            if delta.tool_calls:
                for dtc in delta.tool_calls:
                    func = dtc.function
                    if func and func.name:
                        print(f"  ToolCall[{dtc.index}]: "
                              f"name={func.name}")
                    if func and func.arguments:
                        print(f"  ToolCall[{dtc.index}]: "
                              f"args={func.arguments}")

    print("\n" + "*" * 50)
