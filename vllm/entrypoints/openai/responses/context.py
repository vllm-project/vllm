# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import copy
import json
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, Union

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.tool import Mcp
from openai_harmony import Author, Message, Role, TextContent

from vllm import envs
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.mcp.tool import Tool
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    FunctionCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import render_for_completion
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponseRawMessageAndToken,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    build_response_output_items,
    construct_tool_dicts,
)
from vllm.entrypoints.serve.utils.constants import MCP_PREFIX
from vllm.outputs import RequestOutput
from vllm.parser.abstract_parser import Parser
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from mcp.client import ClientSession

logger = logging.getLogger(__name__)

# This is currently needed as the tool type doesn't 1:1 match the
# tool namespace, which is what is used to look up the
# connection to the tool server
_TOOL_NAME_TO_TYPE_MAP = {
    "browser": "web_search_preview",
    "python": "code_interpreter",
    "container": "container",
}


def _map_tool_name_to_tool_type(tool_name: str) -> str:
    if tool_name not in _TOOL_NAME_TO_TYPE_MAP:
        available_tools = ", ".join(_TOOL_NAME_TO_TYPE_MAP.keys())
        raise ValueError(
            f"Built-in tool name '{tool_name}' not defined in mapping. "
            f"Available tools: {available_tools}"
        )
    return _TOOL_NAME_TO_TYPE_MAP[tool_name]


class TurnMetrics:
    """Tracks token and toolcall details for a single conversation turn."""

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        tool_output_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.tool_output_tokens = tool_output_tokens

    def reset(self) -> None:
        """Reset counters for a new turn."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.tool_output_tokens = 0

    def copy(self) -> "TurnMetrics":
        """Create a copy of this turn's token counts."""
        return TurnMetrics(
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.tool_output_tokens,
        )


class ConversationContext(ABC):
    response_parser: Parser | None = None

    @abstractmethod
    def append_output(self, output: RequestOutput) -> None:
        pass

    @abstractmethod
    def append_tool_output(self, output) -> None:
        pass

    @abstractmethod
    async def call_tool(self) -> list[Message]:
        pass

    @abstractmethod
    def need_builtin_tool_call(self) -> bool:
        pass

    @abstractmethod
    def render_for_completion(self) -> list[int]:
        pass

    @abstractmethod
    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ) -> None:
        pass

    @abstractmethod
    async def cleanup_session(self) -> None:
        raise NotImplementedError("Should not be called.")


def _create_json_parse_error_messages(
    last_msg: Message, e: json.JSONDecodeError
) -> list[Message]:
    """
    Creates an error message when json parse failed.
    """
    error_msg = (
        f"Error parsing tool arguments as JSON: {str(e)}. "
        "Please ensure the tool call arguments are valid JSON and try again."
    )
    content = TextContent(text=error_msg)
    author = Author(role=Role.TOOL, name=last_msg.recipient)
    return [
        Message(
            author=author,
            content=[content],
            recipient=Role.ASSISTANT,
            channel=last_msg.channel,
        )
    ]


class SimpleContext(ConversationContext):
    """This is a context that cannot handle MCP tool calls"""

    def __init__(
        self,
        *,
        response_parser: Parser | None = None,
        parser_cls: type[Parser] | None = None,
        tokenizer: TokenizerLike | None = None,
        request: ResponsesRequest | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.last_output = None
        self.response_parser = response_parser or (
            parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            if parser_cls is not None and tokenizer is not None and request is not None
            else None
        )

        # Accumulated final output for streaming mode
        self._accumulated_text: str = ""
        self._accumulated_token_ids: list[int] = []
        self._accumulated_logprobs: list = []

        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        # todo num_reasoning_tokens is not implemented yet.
        self.num_reasoning_tokens = 0
        # not implemented yet for SimpleContext
        self.all_turn_metrics: list[TurnMetrics] = []

        self.input_messages: list[ResponseRawMessageAndToken] = []
        self.kv_transfer_params: dict[str, Any] | None = None

    def append_output(self, output) -> None:
        self.last_output = output
        if not isinstance(output, RequestOutput):
            raise ValueError("SimpleContext only supports RequestOutput.")
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])
        if output.kv_transfer_params is not None:
            self.kv_transfer_params = output.kv_transfer_params

        # Accumulate text, token_ids, and logprobs for streaming mode
        delta_output = output.outputs[0]
        self._accumulated_text += delta_output.text
        self._accumulated_token_ids.extend(delta_output.token_ids)
        if delta_output.logprobs is not None:
            self._accumulated_logprobs.extend(delta_output.logprobs)

        if len(self.input_messages) == 0:
            output_prompt = output.prompt or ""
            output_prompt_token_ids = output.prompt_token_ids or []
            self.input_messages.append(
                ResponseRawMessageAndToken(
                    message=output_prompt,
                    tokens=output_prompt_token_ids,
                )
            )

    @property
    def output_messages(self) -> list[ResponseRawMessageAndToken]:
        """Return consolidated output as a single message.

        In streaming mode, text and tokens are accumulated across many deltas.
        This property returns them as a single entry rather than one per delta.
        """
        if not self._accumulated_text and not self._accumulated_token_ids:
            return []
        return [
            ResponseRawMessageAndToken(
                message=self._accumulated_text,
                tokens=list(self._accumulated_token_ids),
            )
        ]

    @property
    def final_output(self) -> RequestOutput | None:
        """Return the final output, with complete text/token_ids/logprobs."""
        if self.last_output is not None and self.last_output.outputs:
            assert isinstance(self.last_output, RequestOutput)
            final_output = copy.copy(self.last_output)
            # copy inner item to avoid modify last_output
            final_output.outputs = [replace(item) for item in self.last_output.outputs]
            final_output.outputs[0].text = self._accumulated_text
            final_output.outputs[0].token_ids = tuple(self._accumulated_token_ids)
            if self._accumulated_logprobs:
                final_output.outputs[0].logprobs = self._accumulated_logprobs
            return final_output
        return self.last_output

    def append_tool_output(self, output) -> None:
        raise NotImplementedError("Should not be called.")

    def need_builtin_tool_call(self) -> bool:
        return False

    async def call_tool(self) -> list[Message]:
        raise NotImplementedError("Should not be called.")

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")

    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ) -> None:
        pass

    async def cleanup_session(self) -> None:
        raise NotImplementedError("Should not be called.")


class ParsableContext(ConversationContext):
    def __init__(
        self,
        *,
        response_messages: list[ResponseInputOutputItem],
        tokenizer: TokenizerLike,
        parser_cls: type[Parser] | None,
        request: ResponsesRequest,
        available_tools: list[str] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        response_parser: Parser | None = None,
        enable_auto_tools: bool = False,
    ):
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        self.num_reasoning_tokens = 0
        # not implemented yet for ParsableContext
        self.all_turn_metrics: list[TurnMetrics] = []

        self.response_messages: list[ResponseInputOutputItem] = response_messages
        self.num_init_messages = len(response_messages)
        self.finish_reason: str | None = None
        self.enable_auto_tools = enable_auto_tools

        self.response_parser = response_parser
        self.parser_cls = parser_cls
        self.request = request

        self.available_tools = available_tools or []
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_tools: set[str] = set()

        self.tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        self.input_messages: list[ResponseRawMessageAndToken] = []
        self.output_messages: list[ResponseRawMessageAndToken] = []
        self._accumulated_token_ids: list[int] = []
        self.kv_transfer_params: dict[str, Any] | None = None

    def append_output(self, output: RequestOutput) -> None:
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])
        if output.kv_transfer_params is not None:
            self.kv_transfer_params = output.kv_transfer_params

        completion = output.outputs[0]
        self.finish_reason = completion.finish_reason

        if self.response_parser is not None:
            reasoning, content, tool_calls = self.response_parser.parse(
                completion.text,
                self.request,
                enable_auto_tools=self.enable_auto_tools,
                model_output_token_ids=completion.token_ids,
            )
            self.response_messages.extend(
                build_response_output_items(
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        elif completion.text:
            self.response_messages.append(
                ResponseOutputMessage(
                    type="message",
                    id=f"msg_{random_uuid()}",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            type="output_text",
                            text=completion.text,
                            logprobs=None,
                        )
                    ],
                )
            )

        self._accumulated_token_ids.extend(completion.token_ids or [])

        if self.request.enable_response_messages:
            output_prompt = output.prompt or ""
            output_prompt_token_ids = output.prompt_token_ids or []
            if len(self.input_messages) == 0:
                self.input_messages.append(
                    ResponseRawMessageAndToken(
                        message=output_prompt,
                        tokens=output_prompt_token_ids,
                    )
                )
            else:
                self.output_messages.append(
                    ResponseRawMessageAndToken(
                        message=output_prompt,
                        tokens=output_prompt_token_ids,
                    )
                )
            self.output_messages.append(
                ResponseRawMessageAndToken(
                    message=completion.text,
                    tokens=completion.token_ids,
                )
            )

    def append_tool_output(self, output: list[ResponseInputOutputItem]) -> None:
        self.response_messages.extend(output)

    def need_builtin_tool_call(self) -> bool:
        """Return true if the last message is a builtin tool call
        that the request has enabled."""
        last_message = self.response_messages[-1]
        if last_message.type != "function_call":
            return False
        if last_message.name in ("code_interpreter", "python"):
            return "python" in self.available_tools
        if last_message.name == "web_search_preview":
            return "browser" in self.available_tools
        if last_message.name.startswith("container"):
            return "container" in self.available_tools
        return False

    async def call_python_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: FunctionCall
    ) -> list[ResponseInputOutputItem]:
        self.called_tools.add("python")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result_parsable_context(self)
        args = json.loads(last_msg.arguments)
        param = {
            "code": args["code"],
        }
        result = await tool_session.call_tool("python", param)
        result_str = result.content[0].text

        message = ResponseFunctionToolCallOutputItem(
            id=f"mcpo_{random_uuid()}",
            type="function_call_output",
            call_id=f"call_{random_uuid()}",
            output=result_str,
            status="completed",
        )

        return [message]

    async def call_search_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: FunctionCall
    ) -> list[ResponseInputOutputItem]:
        self.called_tools.add("browser")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result_parsable_context(self)
        if envs.VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY:
            try:
                args = json.loads(last_msg.arguments)
            except json.JSONDecodeError as e:
                return _create_json_parse_error_messages(last_msg, e)
        else:
            args = json.loads(last_msg.arguments)
        result = await tool_session.call_tool("search", args)
        result_str = result.content[0].text

        message = ResponseFunctionToolCallOutputItem(
            id=f"fco_{random_uuid()}",
            type="function_call_output",
            call_id=f"call_{random_uuid()}",
            output=result_str,
            status="completed",
        )

        return [message]

    async def call_container_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        """
        Call container tool. Expect this to be run in a stateful docker
        with command line terminal.
        The official container tool would at least
        expect the following format:
        - for tool name: exec
            - args:
                {
                    "cmd":List[str] "command to execute",
                    "workdir":optional[str] "current working directory",
                    "env":optional[object/dict] "environment variables",
                    "session_name":optional[str] "session name",
                    "timeout":optional[int] "timeout in seconds",
                    "user":optional[str] "user name",
                }
        """
        self.called_tools.add("container")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result_parsable_context(self)
        # tool_name = last_msg.recipient.split(".")[1].split(" ")[0]
        if envs.VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY:
            try:
                args = json.loads(last_msg.arguments)
            except json.JSONDecodeError as e:
                return _create_json_parse_error_messages(last_msg, e)
        else:
            args = json.loads(last_msg.arguments)
        result = await tool_session.call_tool("exec", args)
        result_str = result.content[0].text

        message = ResponseFunctionToolCallOutputItem(
            id=f"fco_{random_uuid()}",
            type="function_call_output",
            call_id=f"call_{random_uuid()}",
            output=result_str,
            status="completed",
        )

        return [message]

    async def call_tool(self) -> list[ResponseInputOutputItem]:
        if not self.response_messages:
            return []
        last_msg = self.response_messages[-1]
        # change this to a mcp_ function call
        last_msg.id = f"{MCP_PREFIX}{random_uuid()}"
        self.response_messages[-1] = last_msg
        if last_msg.name == "code_interpreter":
            return await self.call_python_tool(self._tool_sessions["python"], last_msg)
        elif last_msg.name == "web_search_preview":
            return await self.call_search_tool(self._tool_sessions["browser"], last_msg)
        elif last_msg.name.startswith("container"):
            return await self.call_container_tool(
                self._tool_sessions["container"], last_msg
            )
        return []

    def make_response_output_items(self) -> list[ResponseOutputItem]:
        response_messages = self.response_messages[self.num_init_messages :]
        output_messages: list[ResponseOutputItem] = []
        for message in response_messages:
            if not isinstance(message, ResponseFunctionToolCallOutputItem):
                output_messages.append(message)
            else:
                if len(output_messages) == 0:
                    raise ValueError(
                        "Cannot have a FunctionToolCallOutput before FunctionToolCall."
                    )
                if isinstance(output_messages[-1], ResponseFunctionToolCall):
                    output_messages[-1] = McpCall(
                        id=f"{MCP_PREFIX}{random_uuid()}",
                        arguments=output_messages[-1].arguments,
                        name=output_messages[-1].name,
                        server_label=output_messages[-1].name,
                        type="mcp_call",
                        status="completed",
                        output=message.output,
                    )
        return output_messages

    def render_for_completion(self):
        raise NotImplementedError("Should not be called.")

    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ):
        if tool_server:
            for tool_name in self.available_tools:
                if tool_name in self._tool_sessions:
                    continue

                tool_type = _map_tool_name_to_tool_type(tool_name)
                headers = (
                    mcp_tools[tool_type].headers if tool_type in mcp_tools else None
                )
                tool_session = await exit_stack.enter_async_context(
                    tool_server.new_session(tool_name, request_id, headers)
                )
                self._tool_sessions[tool_name] = tool_session
                exit_stack.push_async_exit(self.cleanup_session)

    async def cleanup_session(self, *args, **kwargs) -> None:
        """Can be used as coro to used in __aexit__"""

        async def cleanup_tool_session(tool_session):
            if not isinstance(tool_session, Tool):
                logger.info(
                    "Cleaning up tool session for %s", tool_session._client_info
                )
                with contextlib.suppress(Exception):
                    await tool_session.call_tool("cleanup_session", {})

        await asyncio.gather(
            *(
                cleanup_tool_session(self._tool_sessions[tool])
                for tool in self.called_tools
            )
        )


class HarmonyContext(ConversationContext):
    def __init__(
        self,
        messages: list,
        available_tools: list[str],
        function_tool_names: frozenset[str],
        response_parser: Parser | None = None,
    ):
        from vllm.parser.harmony import HarmonyParser, Segment

        assert isinstance(response_parser, HarmonyParser)

        self._messages = messages
        self.response_parser: HarmonyParser = response_parser
        self.finish_reason: str | None = None
        self.available_tools = available_tools
        self.function_tool_names = function_tool_names
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_tools: set[str] = set()

        self.num_init_messages = len(messages)
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        self.num_reasoning_tokens = 0
        self.num_tool_output_tokens = 0

        self.last_append_segments: list[Segment] = []
        self.last_append_flush_status: bool = False

        # Turn tracking - replaces multiple individual tracking variables
        self.current_turn_metrics = TurnMetrics()
        # Track metrics for all turns
        self.all_turn_metrics: list[TurnMetrics] = []
        self.is_first_turn = True
        self.first_tok_of_message = True
        self.kv_transfer_params: dict[str, Any] | None = None

    def append_output(self, output: RequestOutput) -> None:
        if self.first_tok_of_message:
            self.finish_reason = None
            self._update_prefill_token_usage(output)

        output_token_ids = output.outputs[0].token_ids
        result = self.response_parser.process_chunk(output_token_ids)
        segments = result.segments
        self.num_reasoning_tokens += result.reasoning_token_count

        self.first_tok_of_message = output.finished
        self._update_decode_token_usage(output)
        if output.kv_transfer_params is not None:
            self.kv_transfer_params = output.kv_transfer_params

        if output.finished:
            self.finish_reason = output.outputs[0].finish_reason
            flushed_segments = self.response_parser.flush()
            if flushed_segments:
                segments.extend(flushed_segments)
            self.last_append_flush_status = len(flushed_segments) > 0
            self.all_turn_metrics.append(self.current_turn_metrics.copy())
            self.current_turn_metrics.reset()

        self.last_append_segments = segments
        self._messages.extend(
            segment.completed_message
            for segment in segments
            if segment.completed_message is not None
        )

    def append_tool_output(self, output: list[Message]) -> None:
        output_msgs = output
        self._messages.extend(output_msgs)

    def _update_prefill_token_usage(self, output: RequestOutput) -> None:
        """Update token usage statistics for the prefill phase of generation.

        The prefill phase processes the input prompt tokens. This method:
        1. Counts the prompt tokens for this turn
        2. Calculates tool output tokens for multi-turn conversations
        3. Updates cached token counts
        4. Tracks state for next turn calculations

        Tool output tokens are calculated as:
        current_prompt_tokens - last_turn_prompt_tokens -
        last_turn_output_tokens
        This represents tokens added between turns (typically tool responses).

        Args:
            output: The RequestOutput containing prompt token information
        """
        if output.prompt_token_ids is not None:
            this_turn_input_tokens = len(output.prompt_token_ids)
        else:
            this_turn_input_tokens = 0
            logger.error("RequestOutput appended contains no prompt_token_ids.")

        # Update current turn input tokens
        self.current_turn_metrics.input_tokens = this_turn_input_tokens
        self.num_prompt_tokens += this_turn_input_tokens

        # Calculate tool tokens (except on first turn)
        if self.is_first_turn:
            self.is_first_turn = False
        else:
            previous_turn = self.all_turn_metrics[-1]
            # start counting tool after first turn
            # tool tokens = this turn prefill - last turn prefill -
            # last turn decode
            this_turn_tool_tokens = (
                self.current_turn_metrics.input_tokens
                - previous_turn.input_tokens
                - previous_turn.output_tokens
            )

            # Handle negative tool token counts (shouldn't happen in normal
            # cases)
            if this_turn_tool_tokens < 0:
                logger.error(
                    "Negative tool output tokens calculated: %d "
                    "(current_input=%d, previous_input=%d, "
                    "previous_output=%d). Setting to 0.",
                    this_turn_tool_tokens,
                    self.current_turn_metrics.input_tokens,
                    previous_turn.input_tokens,
                    previous_turn.output_tokens,
                )
                this_turn_tool_tokens = 0

            self.num_tool_output_tokens += this_turn_tool_tokens
            self.current_turn_metrics.tool_output_tokens = this_turn_tool_tokens

        # Update cached tokens
        num_cached_token = output.num_cached_tokens
        if num_cached_token is not None:
            self.num_cached_tokens += num_cached_token
            self.current_turn_metrics.cached_input_tokens = num_cached_token

    def _update_decode_token_usage(self, output: RequestOutput) -> int:
        """Update token usage statistics for the decode phase of generation.

        The decode phase processes the generated output tokens. This method:
        1. Counts output tokens from all completion outputs
        2. Updates the total output token count
        3. Tracks tokens generated in the current turn

        In streaming mode, this is called for each token generated.
        In non-streaming mode, this is called once with all output tokens.

        Args:
            output: The RequestOutput containing generated token information

        Returns:
            int: Number of output tokens processed in this call
        """
        updated_output_token_count = 0
        if output.outputs:
            for completion_output in output.outputs:
                # only keep last round
                updated_output_token_count += len(completion_output.token_ids)
            self.num_output_tokens += updated_output_token_count
            self.current_turn_metrics.output_tokens += updated_output_token_count
        return updated_output_token_count

    @property
    def messages(self) -> list:
        return self._messages

    def need_builtin_tool_call(self) -> bool:
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is None:
            return False
        if recipient.startswith("browser."):
            return "browser" in self.available_tools
        if recipient.startswith("python"):
            return "python" in self.available_tools
        if recipient.startswith("container."):
            return "container" in self.available_tools
        return False

    async def call_tool(self) -> list[Message]:
        if not self.messages:
            return []
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is not None:
            if recipient.startswith("browser."):
                return await self.call_search_tool(
                    self._tool_sessions["browser"], last_msg
                )
            elif recipient.startswith("python"):
                return await self.call_python_tool(
                    self._tool_sessions["python"], last_msg
                )
            elif recipient.startswith("container."):
                return await self.call_container_tool(
                    self._tool_sessions["container"], last_msg
                )
        raise ValueError("No tool call found")

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)

    async def call_search_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        self.called_tools.add("browser")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        tool_name = last_msg.recipient.split(".")[1]
        if envs.VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY:
            try:
                args = json.loads(last_msg.content[0].text)
            except json.JSONDecodeError as e:
                return _create_json_parse_error_messages(last_msg, e)
        else:
            args = json.loads(last_msg.content[0].text)
        result = await tool_session.call_tool(tool_name, args)
        result_str = result.content[0].text
        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name=last_msg.recipient)
        return [
            Message(
                author=author,
                content=[content],
                recipient=Role.ASSISTANT,
                channel=last_msg.channel,
            )
        ]

    async def call_python_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        self.called_tools.add("python")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        param = {
            "code": last_msg.content[0].text,
        }
        result = await tool_session.call_tool("python", param)
        result_str = result.content[0].text

        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name="python")

        return [
            Message(
                author=author,
                content=[content],
                channel=last_msg.channel,
                recipient=Role.ASSISTANT,
            )
        ]

    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ):
        if tool_server:
            for tool_name in self.available_tools:
                if tool_name not in self._tool_sessions:
                    tool_type = _map_tool_name_to_tool_type(tool_name)
                    headers = (
                        mcp_tools[tool_type].headers if tool_type in mcp_tools else None
                    )
                    tool_session = await exit_stack.enter_async_context(
                        tool_server.new_session(tool_name, request_id, headers)
                    )
                    self._tool_sessions[tool_name] = tool_session
                    exit_stack.push_async_exit(self.cleanup_session)

    async def call_container_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        """
        Call container tool. Expect this to be run in a stateful docker
        with command line terminal.
        The official container tool would at least
        expect the following format:
        - for tool name: exec
            - args:
                {
                    "cmd":List[str] "command to execute",
                    "workdir":optional[str] "current working directory",
                    "env":optional[object/dict] "environment variables",
                    "session_name":optional[str] "session name",
                    "timeout":optional[int] "timeout in seconds",
                    "user":optional[str] "user name",
                }
        """
        self.called_tools.add("container")
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        tool_name = last_msg.recipient.split(".")[1].split(" ")[0]
        if envs.VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY:
            try:
                args = json.loads(last_msg.content[0].text)
            except json.JSONDecodeError as e:
                return _create_json_parse_error_messages(last_msg, e)
        else:
            args = json.loads(last_msg.content[0].text)
        result = await tool_session.call_tool(tool_name, args)
        result_str = result.content[0].text
        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name=last_msg.recipient)
        return [
            Message(
                author=author,
                content=[content],
                recipient=Role.ASSISTANT,
                channel=last_msg.channel,
            )
        ]

    async def cleanup_session(self, *args, **kwargs) -> None:
        """Can be used as coro to used in __aexit__"""

        async def cleanup_tool_session(tool_session):
            if not isinstance(tool_session, Tool):
                logger.info(
                    "Cleaning up tool session for %s", tool_session._client_info
                )
                with contextlib.suppress(Exception):
                    await tool_session.call_tool("cleanup_session", {})

        await asyncio.gather(
            *(
                cleanup_tool_session(self._tool_sessions[tool])
                for tool in self.called_tools
            )
        )
