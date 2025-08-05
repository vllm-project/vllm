# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from openai_harmony import Role, StreamState

from vllm.entrypoints.harmony_utils import (
    get_encoding, get_streamable_parser_for_assistant,
    parse_output_into_messages, render_for_completion)
from vllm.outputs import RequestOutput

if TYPE_CHECKING:
    # Avoid circular import.
    from vllm.entrypoints.tool import Tool


class ConversationContext(ABC):

    @abstractmethod
    def append_output(self, output) -> None:
        pass

    @abstractmethod
    def get_tool_call(self) -> Optional["Tool"]:
        pass

    @abstractmethod
    def render_for_completion(self) -> list[int]:
        pass


class SimpleContext(ConversationContext):

    def __init__(self):
        self.last_output = None

    def append_output(self, output) -> None:
        self.last_output = output

    def get_tool_call(self) -> Optional["Tool"]:
        # Doesn't support builtin tool calls.
        return None

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")


class HarmonyContext(ConversationContext):

    def __init__(
        self,
        messages: list,
        browser_tool,
        python_tool,
    ):
        self._messages = messages
        self.browser_tool = browser_tool
        self.python_tool = python_tool

        self.num_init_messages = len(messages)
        # TODO
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
        self.num_reasoning_tokens = 0

    def append_output(self, output) -> None:
        if isinstance(output, RequestOutput):
            output_token_ids = output.outputs[0].token_ids
            parser = parse_output_into_messages(output_token_ids)
            output_msgs = parser.messages
        else:
            # Tool output.
            output_msgs = output
        self._messages.extend(output_msgs)

    @property
    def messages(self) -> list:
        return self._messages

    def get_tool_call(self) -> Optional["Tool"]:
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is not None:
            if recipient.startswith("browser."):
                return self.browser_tool
            elif recipient.startswith("python"):
                return self.python_tool
        return None

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)


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
