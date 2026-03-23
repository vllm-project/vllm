# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse


class ResponsesStore(ABC):
    """Abstract base class for Responses API storage backends.

    Provides async get/put operations for responses and messages.
    The event_store (used for background streaming coordination) is
    intentionally not part of this interface — it uses asyncio.Event
    and is inherently local to the instance.
    """

    @abstractmethod
    async def get_response(self, response_id: str) -> "ResponsesResponse | None":
        """Retrieve a stored response by ID, or None if not found."""
        ...

    @abstractmethod
    async def put_response(
        self, response_id: str, response: "ResponsesResponse"
    ) -> None:
        """Store or update a response."""
        ...

    @abstractmethod
    async def get_messages(
        self, response_id: str
    ) -> "list[ChatCompletionMessageParam] | None":
        """Retrieve stored messages for a response, or None if not found."""
        ...

    @abstractmethod
    async def put_messages(
        self,
        response_id: str,
        messages: "list[ChatCompletionMessageParam]",
    ) -> None:
        """Store messages associated with a response."""
        ...

    async def close(self) -> None:
        """Optional cleanup hook for backends that hold resources."""
        pass
