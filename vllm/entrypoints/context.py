# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from openai.types.responses.tool import Mcp
from openai_harmony import Author, Message, Role, StreamState, TextContent

from vllm.entrypoints.harmony_utils import (
    get_encoding,
    get_streamable_parser_for_assistant,
    render_for_completion,
)
from vllm.entrypoints.mcp.mcp_utils import call_mcp_tool
from vllm.entrypoints.tool import Tool
from vllm.entrypoints.tool_server import ToolServer
from vllm.outputs import RequestOutput

if TYPE_CHECKING:
    from mcp.client import ClientSession

logger = logging.getLogger(__name__)


class TurnTokens:
    """Tracks token counts for a single conversation turn."""

    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def reset(self):
        """Reset counters for a new turn."""
        self.input_tokens = 0
        self.output_tokens = 0

    def copy(self):
        """Create a copy of this turn's token counts."""
        return TurnTokens(self.input_tokens, self.output_tokens)


class ConversationContext(ABC):
    @abstractmethod
    def append_output(self, output) -> None:
        pass

    @abstractmethod
    async def call_tool(self) -> list[Message]:
        pass

    @abstractmethod
    def need_server_side_tool_call(self) -> bool:
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


class SimpleContext(ConversationContext):
    def __init__(self):
        self.last_output = None
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        # todo num_reasoning_tokens is not implemented yet.
        self.num_reasoning_tokens = 0

    def append_output(self, output) -> None:
        self.last_output = output
        if not isinstance(output, RequestOutput):
            raise ValueError("SimpleContext only supports RequestOutput.")
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])

    def need_server_side_tool_call(self) -> bool:
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


class HarmonyContext(ConversationContext):
    def __init__(
        self,
        messages: list,
        enabled_tool_namespaces: list[str],
    ):
        """Initialize HarmonyContext for managing conversation state.

        Args:
            messages: Initial conversation messages
            enabled_tool_namespaces: List of all enabled tool namespaces
                (includes both elevated and custom MCP tools)
        """
        self._messages = messages
        self.finish_reason: str | None = None
        self.available_tools = enabled_tool_namespaces
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_namespaces: set[str] = set()

        self.parser = get_streamable_parser_for_assistant()
        self.num_init_messages = len(messages)
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        self.num_reasoning_tokens = 0
        self.num_tool_output_tokens = 0

        # Turn tracking - replaces multiple individual tracking variables
        self.current_turn = TurnTokens()
        self.previous_turn = TurnTokens()
        self.is_first_turn = True
        self.first_tok_of_message = True  # For streaming support

    def _update_num_reasoning_tokens(self):
        # Count all analysis and commentary channels as reasoning tokens
        if self.parser.current_channel in {"analysis", "commentary"}:
            self.num_reasoning_tokens += 1

    def append_output(self, output: RequestOutput | list[Message]) -> None:
        if isinstance(output, RequestOutput):
            output_token_ids = output.outputs[0].token_ids
            self.parser = get_streamable_parser_for_assistant()
            for token_id in output_token_ids:
                self.parser.process(token_id)
                # Check if the current token is part of reasoning content
                self._update_num_reasoning_tokens()
            self._update_prefill_token_usage(output)
            # Reset current turn output tokens for this turn
            self.current_turn.output_tokens = 0
            self._update_decode_token_usage(output)
            # Move current turn to previous turn for next turn's calculations
            self.previous_turn = self.current_turn.copy()
            # append_output is called only once before tool calling
            # in non-streaming case
            # so we can append all the parser messages to _messages
            output_msgs = self.parser.messages
            # The responses finish reason is set in the last message
            self.finish_reason = output.outputs[0].finish_reason
        else:
            # Tool output.
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
        self.current_turn.input_tokens = this_turn_input_tokens
        self.num_prompt_tokens += this_turn_input_tokens

        # Calculate tool tokens (except on first turn)
        if self.is_first_turn:
            self.is_first_turn = False
        else:
            # start counting tool after first turn
            # tool tokens = this turn prefill - last turn prefill -
            # last turn decode
            this_turn_tool_tokens = (
                self.current_turn.input_tokens
                - self.previous_turn.input_tokens
                - self.previous_turn.output_tokens
            )

            # Handle negative tool token counts (shouldn't happen in normal
            # cases)
            if this_turn_tool_tokens < 0:
                logger.error(
                    "Negative tool output tokens calculated: %d "
                    "(current_input=%d, previous_input=%d, "
                    "previous_output=%d). Setting to 0.",
                    this_turn_tool_tokens,
                    self.current_turn.input_tokens,
                    self.previous_turn.input_tokens,
                    self.previous_turn.output_tokens,
                )
                this_turn_tool_tokens = 0

            self.num_tool_output_tokens += this_turn_tool_tokens

        # Update cached tokens
        if output.num_cached_tokens is not None:
            self.num_cached_tokens += output.num_cached_tokens

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
            self.current_turn.output_tokens += updated_output_token_count
        return updated_output_token_count

    @property
    def messages(self) -> list:
        return self._messages

    def _resolve_namespace(self, recipient: str) -> str:
        """Map recipient to tool namespace.

        Most tools use recipient prefix as namespace
        (e.g., "browser.search" → "browser").
        Exception: "python" → "code_interpreter" for gpt-oss specifically.

        Args:
            recipient: The recipient string from the message

        Returns:
            The namespace string
        """
        if recipient.startswith("python"):
            return "code_interpreter"
        return recipient.split(".")[0] if "." in recipient else recipient

    def _resolve_tool_name(self, recipient: str) -> str:
        """Map recipient to tool name.

        Most tools use recipient suffix as tool_name
        (e.g., "browser.search" → "search").
        Exception: "python" → "python" for gpt-oss specifically.

        Args:
            recipient: The recipient string from the message

        Returns:
            The tool_name string
        """
        if recipient.startswith("python"):
            return "python"
        return recipient.split(".")[-1] if "." in recipient else recipient

    def need_server_side_tool_call(self) -> bool:
        """Check if the last message requires a server-side tool call.

        Returns:
            True if recipient is set, not a client-side function,
                                        and namespace is available
            False otherwise
        """
        if not self.messages:
            return False

        last_msg = self.messages[-1]
        recipient = last_msg.recipient

        if not recipient:
            return False

        # Client-side function tools are handled by client
        if recipient.startswith("functions."):
            return False

        # Validate that the namespace is actually available
        namespace = self._resolve_namespace(recipient)
        if namespace not in self.available_tools:
            logger.warning(
                "Model requested unknown tool namespace: %s (from recipient: %s). "
                "Available: %s. Ignoring tool call.",
                namespace,
                recipient,
                self.available_tools,
            )
            return False

        return True

    async def call_tool(self) -> list[Message]:
        if not self.messages:
            return []
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is not None:
            namespace = self._resolve_namespace(recipient)
            if namespace not in self._tool_sessions:
                available = list(self._tool_sessions.keys())
                raise ValueError(
                    f"Tool session for namespace '{namespace}' not found. "
                    f"Model requested recipient '{recipient}' but the "
                    f"Available namespaces are: {available}"
                )
            tool_session = self._tool_sessions[namespace]
            if isinstance(tool_session, Tool):
                return await tool_session.get_result(self)

            tool_name = self._resolve_tool_name(recipient)
            # Using str here to do str -> json error handling
            # in one spot in call_mcp_tool
            tool_args_str = ""
            # code_interpreter is special as the model outputs code not json
            if namespace == "code_interpreter":
                tool_args_str = json.dumps(
                    {
                        "code": last_msg.content[0].text,
                    }
                )
            else:
                tool_args_str = last_msg.content[0].text

            self.called_namespaces.add(namespace)
            tool_output_str = await call_mcp_tool(
                tool_session, tool_name, tool_args_str
            )
            return [
                Message(
                    author=Author(role=Role.TOOL, name=recipient),
                    content=[TextContent(text=tool_output_str)],
                    recipient=Role.ASSISTANT,
                    channel=last_msg.channel,
                )
            ]
        raise ValueError("No tool call found")

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)

    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ):
        if tool_server:
            for namespace in self.available_tools:
                if namespace not in self._tool_sessions:
                    headers = (
                        mcp_tools[namespace].headers if namespace in mcp_tools else None
                    )
                    tool_session = await exit_stack.enter_async_context(
                        tool_server.new_session(namespace, request_id, headers)
                    )
                    self._tool_sessions[namespace] = tool_session
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
                for tool in self.called_namespaces
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

    @property
    def messages(self) -> list:
        return self._messages

    def append_output(self, output: RequestOutput | list[Message]) -> None:
        if isinstance(output, RequestOutput):
            # append_output is called for each output token in streaming case,
            # so we only want to add the prompt tokens once for each message.
            if self.first_tok_of_message:
                self._update_prefill_token_usage(output)
                self.current_turn.output_tokens = 0
            # Reset self.first_tok_of_message if needed:
            # if the current token is the last one of the current message
            # (finished=True), then the next token processed will mark the
            # beginning of a new message
            self.first_tok_of_message = output.finished
            for tok in output.outputs[0].token_ids:
                self.parser.process(tok)
            self._update_decode_token_usage(output)

            # For streaming, update previous turn when message is complete
            if output.finished:
                self.previous_turn = self.current_turn.copy()
            # Check if the current token is part of reasoning content
            self._update_num_reasoning_tokens()
            self.last_tok = tok
            if len(self._messages) - self.num_init_messages < len(self.parser.messages):
                self._messages.extend(
                    self.parser.messages[len(self._messages) - self.num_init_messages :]
                )
        else:
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
        # `<|start|>assistant``,
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
