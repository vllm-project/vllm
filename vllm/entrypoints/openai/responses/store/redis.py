# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesResponse,
    serialize_message,
)
from vllm.entrypoints.openai.responses.store.base import ResponsesStore
from vllm.logger import init_logger

logger = init_logger(__name__)

_RESPONSE_PREFIX = "vllm:responses:response:"
_MESSAGES_PREFIX = "vllm:responses:messages:"


class RedisResponsesStore(ResponsesStore):
    """Redis-backed store for the Responses API.

    Requires the ``redis`` package with async support
    (``pip install redis``). All instances sharing the same Redis
    server see writes immediately.
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "The redis package is required for the Redis responses "
                "store backend. Install it with: pip install redis"
            ) from e

        self._redis = aioredis.from_url(redis_url)
        logger.info(
            "RedisResponsesStore connected to %s",
            redis_url.split("@")[-1],  # avoid logging credentials
        )

    async def get_response(self, response_id: str) -> ResponsesResponse | None:
        data = await self._redis.get(f"{_RESPONSE_PREFIX}{response_id}")
        if data is None:
            return None
        return ResponsesResponse.model_validate_json(data)

    async def put_response(self, response_id: str, response: ResponsesResponse) -> None:
        data = response.model_dump_json()
        await self._redis.set(f"{_RESPONSE_PREFIX}{response_id}", data)

    async def get_messages(
        self, response_id: str
    ) -> list[ChatCompletionMessageParam] | None:
        data = await self._redis.get(f"{_MESSAGES_PREFIX}{response_id}")
        if data is None:
            return None
        return json.loads(data)

    async def put_messages(
        self,
        response_id: str,
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        data = json.dumps([serialize_message(m) for m in messages], ensure_ascii=False)
        await self._redis.set(f"{_MESSAGES_PREFIX}{response_id}", data)

    async def close(self) -> None:
        await self._redis.aclose()
