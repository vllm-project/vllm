# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Callable, TYPE_CHECKING, Optional, Union, Tuple

from openai.types.responses import (ResponseOutputItem,
                                    ResponseOutputMessage, ResponseOutputText,
                                    ResponseReasoningItem)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent)
from openai_harmony import Author, Message, Role, StreamState, TextContent

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.harmony_utils import (
    convert_harmony_message_to_output_item, get_encoding,
    get_streamable_parser_for_assistant,
    parse_remaining_state_into_output_items,
    render_for_completion,)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ResponsesRequest
from vllm.entrypoints.tool import Tool
from vllm.entrypoints.tool_server import ToolServer
from vllm.entrypoints.openai.logprobs_utils import create_response_logprobs
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid
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
    def need_builtin_tool_call(self) -> bool:
        pass

    @abstractmethod
    def render_for_completion(self) -> list[int]:
        pass

    @abstractmethod
    async def init_tool_sessions(self, tool_server: Optional[ToolServer],
                                 exit_stack: AsyncExitStack) -> None:
        pass

    @abstractmethod
    def get_input_messages(self) -> list[ChatCompletionMessageParam]:
        pass

    @abstractmethod
    def get_output_and_output_messages(self, request: ResponsesRequest) -> Tuple[list[ResponseOutputItem], list[ChatCompletionMessageParam]]:
        pass

# Standard functionality for ChatCompletion based models
class ChatContext(ConversationContext):
    def __init__(self, input_messages: list[ChatCompletionMessageParam],
                    tokenizer: AnyTokenizer,
                    reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]],
                    request_logger: Optional[RequestLogger] = None):
        self.request_logger = request_logger
        self.last_output = None
        self.tokenizer = tokenizer
        self._input_messages = input_messages.copy()
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        # todo num_reasoning_tokens is not implemented yet.
        self.num_reasoning_tokens = 0
        self.reasoning_parser = reasoning_parser
        # TODO: Tool parser for tool calling for other models

    def append_output(self, output) -> None:
        self.last_output = output
        if not isinstance(output, RequestOutput):
            raise ValueError("SimpleContext only supports RequestOutput.")
        self.num_prompt_tokens = len(output.prompt_token_ids or [])
        self.num_cached_tokens = output.num_cached_tokens or 0
        self.num_output_tokens += len(output.outputs[0].token_ids or [])

    def need_builtin_tool_call(self) -> bool:
        return False

    async def call_tool(self) -> list[Message]:
        raise NotImplementedError("Should not be called.")

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")

    async def init_tool_sessions(self, tool_server: Optional[ToolServer],
                                 exit_stack: AsyncExitStack) -> None:
        pass

    def get_input_messages(self) -> list[ChatCompletionMessageParam]:
        return self._input_messages

    # TODO: Ideally this class only deals with messages, but since we don't have
    # a way to represent reasoning as messages yet
    # This is very important as once Responses specific concepts are removed
    # then this can be used to handle tool calling for completions API as well
    def get_output_and_output_messages(self,
                            request: ResponsesRequest) -> Tuple[
                                list[ResponseOutputItem],
                                list[ChatCompletionMessageParam]]:

        if self.last_output is None:
            return [], []
        assert len(self.last_output.outputs) == 1
        final_output = self.last_output.outputs[0]
        output_items = []
        output_messages = []
        if self.reasoning_parser:
            try:
                reasoning_parser = self.reasoning_parser(self.tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                raise e
            # TODO: Figure out how to get number of reasoning tokens here
            # without tokenizing again
            reasoning_content, content = (
                reasoning_parser.extract_reasoning_content(final_output.text,
                                                           request=request))
        else:
            reasoning_content = None
            self.num_reasoning_tokens = 0
            content = final_output.text

        # Log complete response if output logging is provided
        # matches previous functionality
        if self.request_logger is not None:
            output_text = ""
            if content:
                output_text = content
            elif reasoning_content:
                output_text = f"[reasoning: {reasoning_content}]"

            if output_text:
                self.request_logger.log_outputs(
                    request_id=request.request_id,
                    outputs=output_text,
                    output_token_ids=final_output.token_ids,
                    finish_reason=final_output.finish_reason,
                    is_streaming=False,
                    delta=False,
                )


        if reasoning_content:
            # TODO: Make a ResponseOutputItem but skip a reasoning message
            # since there is no direct match in OpenAI spec and
            # functionality drops them between API requests at the moment
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning_content,
                                                 type="reasoning_text")
                ],
                status=None,  # NOTE: Only the last output item has status.
            )

            output_items.append(reasoning_item)
        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=create_response_logprobs(
                    token_ids=final_output.token_ids,
                    logprobs=final_output.logprobs,
                    tokenizer=self.tokenizer,
                    top_logprobs=request.top_logprobs,
                ) if request.is_include_output_logprobs() else None,
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
            # It is always an assistant message, which is a typed_dict
            output_messages.append({
                        "role": "assistant",
                        "content": content,
                    })

        return output_items, output_messages

class HarmonyContext(ConversationContext):

    def __init__(
        self,
        messages: list,
        available_tools: list[str],
    ):
        self._messages = messages
        self._input_messages = messages.copy()
        self.available_tools = available_tools
        self._tool_sessions: dict[str, Union[ClientSession, Tool]] = {}

        self.parser = get_streamable_parser_for_assistant()
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

    def append_output(self, output) -> None:
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
            output_msgs = self.parser.messages
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
            logger.error(
                "RequestOutput appended contains no prompt_token_ids.")

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
            this_turn_tool_tokens = (self.current_turn.input_tokens -
                                     self.previous_turn.input_tokens -
                                     self.previous_turn.output_tokens)

            # Handle negative tool token counts (shouldn't happen in normal
            # cases)
            if this_turn_tool_tokens < 0:
                logger.error(
                    "Negative tool output tokens calculated: %d "
                    "(current_input=%d, previous_input=%d, "
                    "previous_output=%d). Setting to 0.",
                    this_turn_tool_tokens, self.current_turn.input_tokens,
                    self.previous_turn.input_tokens,
                    self.previous_turn.output_tokens)
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

    def need_builtin_tool_call(self) -> bool:
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        return recipient is not None and (recipient.startswith("browser.")
                                          or recipient.startswith("python"))

    async def call_tool(self) -> list[Message]:
        if not self.messages:
            return []
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is not None:
            if recipient.startswith("browser."):
                return await self.call_search_tool(
                    self._tool_sessions["browser"], last_msg)
            elif recipient.startswith("python"):
                return await self.call_python_tool(
                    self._tool_sessions["python"], last_msg)
        raise ValueError("No tool call found")

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)

    async def call_search_tool(self, tool_session: Union["ClientSession",
                                                         Tool],
                               last_msg: Message) -> list[Message]:
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        tool_name = last_msg.recipient.split(".")[1]
        args = json.loads(last_msg.content[0].text)
        result = await tool_session.call_tool(tool_name, args)
        result_str = result.content[0].text
        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name=last_msg.recipient)
        return [
            Message(author=author, content=[content], recipient=Role.ASSISTANT)
        ]

    async def call_python_tool(self, tool_session: Union["ClientSession",
                                                         Tool],
                               last_msg: Message) -> list[Message]:
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
            Message(author=author,
                    content=[content],
                    channel=last_msg.channel,
                    recipient=Role.ASSISTANT)
        ]

    async def init_tool_sessions(self, tool_server: Optional[ToolServer],
                                 exit_stack: AsyncExitStack) -> None:
        if tool_server:
            for tool_name in self.available_tools:
                if tool_name not in self._tool_sessions:
                    self._tool_sessions[
                        tool_name] = await exit_stack.enter_async_context(
                            tool_server.new_session(tool_name))

    def get_input_messages(self) -> list[ChatCompletionMessageParam]:
        return self._input_messages

    def get_output_and_output_messages(self, request: ResponsesRequest) -> Tuple[list[ResponseOutputItem], list[ChatCompletionMessageParam]]:
        output_items = []
        output_messages = self.messages[len(self._input_messages):]
        for msg in output_messages:
            output_items.extend(convert_harmony_message_to_output_item(msg))
        # Handle the generation stopped in the middle (if any).
        # TODO: This will not result in any messages, so the next API
        # request will not see these outputs, should this be kept?
        last_items = parse_remaining_state_into_output_items(self.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items, output_messages


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
        return self.parser.messages

    def append_output(self, output) -> None:
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

    def is_expecting_start(self) -> bool:
        return self.parser.state == StreamState.EXPECT_START

    def is_assistant_action_turn(self) -> bool:
        return self.last_tok in self.encoding.stop_tokens_for_assistant_actions(
        )

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
