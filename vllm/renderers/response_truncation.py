# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Search whole-message suffixes without changing the stored conversation."""

from collections.abc import Sequence
from typing import Generic, TypeVar

from openai_harmony import Message

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.exceptions import VLLMValidationError

_M = TypeVar("_M", bound=ChatCompletionMessageParam | Message)


def _role(message: ChatCompletionMessageParam | Message) -> str:
    return message.author.role if isinstance(message, Message) else message["role"]


def _cutoffs(messages: Sequence[ChatCompletionMessageParam | Message]) -> list[int]:
    """Keep instructions and never split a known tool call/result pair."""
    calls: dict[str, int] = {}
    dependencies: list[tuple[int, int]] = []
    first_harmony_call: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, Message):
            if message.author.role == "assistant" and message.recipient:
                if first_harmony_call is None:
                    first_harmony_call = index
            elif message.author.role == "tool" and first_harmony_call is not None:
                dependencies.append((first_harmony_call, index))
            elif message.author.role in ("user", "assistant"):
                # Keep parallel calls/results together; Harmony has no call IDs.
                first_harmony_call = None
        else:
            for call in message.get("tool_calls") or ():
                calls[call["id"]] = index
            call_id = message.get("tool_call_id")
            call_index = calls.get(call_id) if call_id is not None else None
            if message["role"] == "tool" and call_index is not None:
                dependencies.append((call_index, index))

    # A cut k removes indices < k except protected instructions. The final
    # message is always retained. Intervals exclude cuts orphaning a tool result.
    blocked = [0] * (len(messages) + 1)
    for start, end in dependencies:
        blocked[start + 1] += 1
        blocked[end + 1] -= 1
    cutoffs = [0]
    active = 0
    for index in range(1, len(messages)):
        active += blocked[index]
        if active == 0 and _role(messages[index - 1]) not in ("system", "developer"):
            cutoffs.append(index)
    return cutoffs


class ResponseTruncationSearch(Generic[_M]):
    """Probe full input, then the smallest safe suffix, then bisect.

    Binary search assumes approximately monotonic rendered lengths. Custom
    templates can violate this: callers must retain an actually validated
    result, rather than infer that an untested candidate fits.
    """

    def __init__(self, messages: Sequence[_M]):
        self.messages = messages
        self.cutoffs = _cutoffs(messages)
        self.index = 0
        self.low = 0
        self.high = len(self.cutoffs) - 1
        self.probing_minimum = False
        self.done = False

    def candidate(self) -> list[_M]:
        cutoff = self.cutoffs[self.index]
        return [
            message
            for index, message in enumerate(self.messages)
            if index >= cutoff or _role(message) in ("system", "developer")
        ]

    def record(self, fits: bool) -> None:
        if self.index == 0 and not self.probing_minimum:
            if fits:
                self.done = True
                return
            self.index = self.high
            self.probing_minimum = True
            if self.high:
                return
        if self.probing_minimum:
            if not fits:
                raise VLLMValidationError(
                    "The instructions and latest message (including required tool "
                    "context) cannot fit in the available input token budget. "
                    "Shorten them or reduce max_output_tokens.",
                    parameter="input_tokens",
                )
            self.probing_minimum = False
        elif fits:
            self.high = self.index
        else:
            self.low = self.index
        if self.high - self.low <= 1:
            self.done = True
        else:
            self.index = (self.low + self.high) // 2


def is_context_overflow(exc: VLLMValidationError) -> bool:
    return exc.parameter in ("input_tokens", "input_text")
