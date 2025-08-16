# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from openai_harmony import Author, Message, Role, StreamState, TextContent

from vllm.entrypoints.harmony_utils import (
    get_encoding, get_streamable_parser_for_assistant, render_for_completion)
from vllm.entrypoints.tool import Tool
from vllm.outputs import RequestOutput

if TYPE_CHECKING:
    from mcp.client import ClientSession

logger = logging.getLogger(__name__)


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


class SimpleContext(ConversationContext):

    def __init__(self):
        self.last_output = None

    def append_output(self, output) -> None:
        self.last_output = output

    def need_builtin_tool_call(self) -> bool:
        return False

    async def call_tool(self) -> list[Message]:
        raise NotImplementedError("Should not be called.")

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")


class HarmonyContext(ConversationContext):

    def __init__(
        self,
        messages: list,
        tool_sessions: dict[str, Tool],
    ):
        self._messages = messages
        self.tool_sessions = tool_sessions

        self.parser = get_streamable_parser_for_assistant()
        self.num_init_messages = len(messages)
        # TODO(woosuk): Implement the following fields.
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
        self.num_reasoning_tokens = 0

    def append_output(self, output) -> None:
        if isinstance(output, RequestOutput):
            output_token_ids = output.outputs[0].token_ids
            self.parser = get_streamable_parser_for_assistant()
            for token_id in output_token_ids:
                self.parser.process(token_id)
            output_msgs = self.parser.messages
        else:
            # Tool output.
            output_msgs = output
        self._messages.extend(output_msgs)

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
                    self.tool_sessions["browser"], last_msg)
            elif recipient.startswith("python"):
                return await self.call_python_tool(
                    self.tool_sessions["python"], last_msg)
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


class StreamingHarmonyContext(HarmonyContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()
        self.encoding = get_encoding()
        self.last_tok = None

    @property
    def messages(self) -> list:
        return self.parser.messages

    def append_output(self, output) -> None:
        if isinstance(output, RequestOutput):
            tok = output.outputs[0].token_ids[0]
            self.parser.process(tok)
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
