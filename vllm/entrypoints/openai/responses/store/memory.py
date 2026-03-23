# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
from vllm.entrypoints.openai.responses.store.base import ResponsesStore


class InMemoryResponsesStore(ResponsesStore):
    """In-memory store using plain dicts. This is the default backend
    and matches the original behavior. Stored data is lost on restart.

    Note: messages are kept as their original objects (which may be
    OpenAIHarmonyMessage instances), so no deserialization is needed.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._responses: dict[str, ResponsesResponse] = {}
        self._messages: dict[str, list[ChatCompletionMessageParam]] = {}

    async def get_response(self, response_id: str) -> ResponsesResponse | None:
        return self._responses.get(response_id)

    async def put_response(self, response_id: str, response: ResponsesResponse) -> None:
        self._responses[response_id] = response

    async def get_messages(
        self, response_id: str
    ) -> list[ChatCompletionMessageParam] | None:
        return self._messages.get(response_id)

    async def put_messages(
        self,
        response_id: str,
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        self._messages[response_id] = messages
