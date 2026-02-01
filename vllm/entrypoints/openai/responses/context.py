# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import copy
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING, Union

from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.tool import Mcp
from openai_harmony import Author, Message, Role, StreamState, TextContent

from vllm import envs
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.mcp.tool import Tool
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    FunctionCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_encoding,
    get_streamable_parser_for_assistant,
    render_for_completion,
)
from vllm.entrypoints.openai.parser.responses_parser import (
    get_responses_parser_for_simple_context,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponseRawMessageAndToken,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts
from vllm.outputs import RequestOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
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

    def __init__(self):
        self.last_output = None

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
        self.all_turn_metrics = []

        self.input_messages: list[ResponseRawMessageAndToken] = []
        self.output_messages: list[ResponseRawMessageAndToken] = []

    def append_output(self, output) -> None:
        self.last_output = output
        if not isinstance(output, RequestOutput):
            raise ValueError("SimpleContext only supports RequestOutput.")
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])

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
        self.output_messages.append(
            ResponseRawMessageAndToken(
                message=delta_output.text,
                tokens=delta_output.token_ids,
            )
        )

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
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser] | None,
        request: ResponsesRequest,
        available_tools: list[str] | None,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ):
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        # TODO: num_reasoning_tokens is not implemented yet.
        self.num_reasoning_tokens = 0
        # not implemented yet for ParsableContext
        self.all_turn_metrics: list[TurnMetrics] = []

        if reasoning_parser_cls is None:
            raise ValueError("reasoning_parser_cls must be provided.")

        self.parser = get_responses_parser_for_simple_context(
            tokenizer=tokenizer,
            reasoning_parser_cls=reasoning_parser_cls,
            response_messages=response_messages,
            request=request,
            tool_parser_cls=tool_parser_cls,
        )
        self.tool_parser_cls = tool_parser_cls
        self.request = request

        self.available_tools = available_tools or []
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_tools: set[str] = set()

        self.tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format

        self.input_messages: list[ResponseRawMessageAndToken] = []
        self.output_messages: list[ResponseRawMessageAndToken] = []
        self._accumulated_token_ids: list[int] = []

    def append_output(self, output: RequestOutput) -> None:
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])
        self.parser.process(output.outputs[0])
        output_token_ids = output.outputs[0].token_ids or []
        self._accumulated_token_ids.extend(output_token_ids)

        # only store if enable_response_messages is True, save memory
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
                    message=output.outputs[0].text,
                    tokens=output.outputs[0].token_ids,
                )
            )

    def append_tool_output(self, output: list[ResponseInputOutputItem]) -> None:
        self.parser.response_messages.extend(output)

    def need_builtin_tool_call(self) -> bool:
        """Return true if the last message is a MCP tool call"""
        last_message = self.parser.response_messages[-1]
        # TODO(qandrew): figure out which tools are MCP tools
        if last_message.type == "function_call":  # noqa: SIM102
            if last_message.name in (
                "code_interpreter",
                "python",
                "web_search_preview",
            ) or last_message.name.startswith("container"):
                return True

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
        if not self.parser.response_messages:
            return []
        last_msg = self.parser.response_messages[-1]
        # change this to a mcp_ function call
        last_msg.id = f"{MCP_PREFIX}{random_uuid()}"
        self.parser.response_messages[-1] = last_msg
        if last_msg.name == "code_interpreter":
            return await self.call_python_tool(self._tool_sessions["python"], last_msg)
        elif last_msg.name == "web_search_preview":
            return await self.call_search_tool(self._tool_sessions["browser"], last_msg)
        elif last_msg.name.startswith("container"):
            return await self.call_container_tool(
                self._tool_sessions["container"], last_msg
            )
        return []

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
    ):
        self._messages = messages
        self.finish_reason: str | None = None
        self.available_tools = available_tools
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_tools: set[str] = set()

        self.parser = get_streamable_parser_for_assistant()
        self.num_init_messages = len(messages)
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        self.num_reasoning_tokens = 0
        self.num_tool_output_tokens = 0

        # Turn tracking - replaces multiple individual tracking variables
        self.current_turn_metrics = TurnMetrics()
        # Track metrics for all turns
        self.all_turn_metrics: list[TurnMetrics] = []
        self.is_first_turn = True
        self.first_tok_of_message = True  # For streaming support

    def _update_num_reasoning_tokens(self):
        # Count all analysis and commentary channels as reasoning tokens
        if self.parser.current_channel in {"analysis", "commentary"}:
            self.num_reasoning_tokens += 1

    def append_output(self, output: RequestOutput) -> None:
        output_token_ids = output.outputs[0].token_ids
        self.parser = get_streamable_parser_for_assistant()
        for token_id in output_token_ids:
            self.parser.process(token_id)
            # Check if the current token is part of reasoning content
            self._update_num_reasoning_tokens()
        self._update_prefill_token_usage(output)
        self._update_decode_token_usage(output)
        # Append current turn to all turn list for next turn's calculations
        self.all_turn_metrics.append(self.current_turn_metrics.copy())
        self.current_turn_metrics.reset()
        # append_output is called only once before tool calling
        # in non-streaming case
        # so we can append all the parser messages to _messages
        output_msgs = self.parser.messages
        # The responses finish reason is set in the last message
        self.finish_reason = output.outputs[0].finish_reason
        self._messages.extend(output_msgs)

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
        return recipient is not None and (
            recipient.startswith("browser.")
            or recipient.startswith("python")
            or recipient.startswith("container.")
        )

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


class StreamingHarmonyContext(HarmonyContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()
        self.encoding = get_encoding()
        self.last_tok = None
        self.first_tok_of_message = True
        self.last_content_delta = None

    @property
    def messages(self) -> list:
        return self._messages

    def append_output(self, output: RequestOutput) -> None:
        # append_output is called for each output token in streaming case,
        # so we only want to add the prompt tokens once for each message.
        self.last_content_delta = None
        if self.first_tok_of_message:
            self._update_prefill_token_usage(output)
        # Reset self.first_tok_of_message if needed:
        # if the current token is the last one of the current message
        # (finished=True), then the next token processed will mark the
        # beginning of a new message
        self.first_tok_of_message = output.finished
        last_delta_text = ""
        for tok in output.outputs[0].token_ids:
            self.parser.process(tok)
            last_delta_text += self.parser.last_content_delta or ""
        if last_delta_text:
            self.last_content_delta = last_delta_text
        self._update_decode_token_usage(output)

        # For streaming, update previous turn when message is complete
        if output.finished:
            self.all_turn_metrics.append(self.current_turn_metrics.copy())
            self.current_turn_metrics.reset()
        # Check if the current token is part of reasoning content
        self._update_num_reasoning_tokens()
        self.last_tok = tok
        if len(self._messages) - self.num_init_messages < len(self.parser.messages):
            self._messages.extend(
                self.parser.messages[len(self._messages) - self.num_init_messages :]
            )

    def append_tool_output(self, output: list[Message]) -> None:
        # Handle the case of tool output in direct message format
        assert len(output) == 1, "Tool output should be a single message"
        msg = output[0]
        # Sometimes the recipient is not set for tool messages,
        # so we set it to "assistant"
        if msg.author.role == Role.TOOL and msg.recipient is None:
            msg.recipient = "assistant"
        toks = self.encoding.render(msg)
        for tok in toks:
            self.parser.process(tok)
        self.last_tok = toks[-1]
        # TODO: add tool_output messages to self._messages

    def is_expecting_start(self) -> bool:
        return self.parser.state == StreamState.EXPECT_START

    def is_assistant_action_turn(self) -> bool:
        return self.last_tok in self.encoding.stop_tokens_for_assistant_actions()

    def render_for_completion(self) -> list[int]:
        # now this list of tokens as next turn's starting tokens
        # `<|start|>assistant`,
        # we need to process them in parser.
        rendered_tokens = super().render_for_completion()

        last_n = -1
        to_process = []
        while rendered_tokens[last_n] != self.last_tok:
            to_process.append(rendered_tokens[last_n])
            last_n -= 1
        for tok in reversed(to_process):
            self.parser.process(tok)

        return rendered_tokens
