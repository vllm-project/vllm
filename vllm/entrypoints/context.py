# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from vllm.entrypoints.harmony_utils import (parse_output_into_messages,
                                            render_for_completion)
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
        self.messages = messages
        self.browser_tool = browser_tool
        self.python_tool = python_tool

        self.num_init_messages = len(messages)
        # TODO
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0

    def append_output(self, output) -> None:
        # TODO: Support streaming.
        if isinstance(output, RequestOutput):
            output_token_ids = output.outputs[0].token_ids
            output_msgs = parse_output_into_messages(output_token_ids)
        else:
            output_msgs = output
        self.messages.extend(output_msgs)

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
