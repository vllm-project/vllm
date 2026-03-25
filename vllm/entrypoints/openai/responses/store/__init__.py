# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs
from vllm.entrypoints.openai.responses.store.base import ResponsesStore
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["ResponsesStore", "create_responses_store"]


def create_responses_store(*, use_harmony: bool = False) -> ResponsesStore:
    """Create a ResponsesStore based on environment configuration.

    Args:
        use_harmony: When True, messages retrieved from the store are
            deserialized as OpenAIHarmonyMessage objects. This is needed
            for harmony (gpt-oss) models where downstream code expects
            Message instances with attributes like ``.channel``.
    """
    backend = envs.VLLM_RESPONSES_STORE_BACKEND

    if backend == "memory":
        from vllm.entrypoints.openai.responses.store.memory import (
            InMemoryResponsesStore,
        )

        return InMemoryResponsesStore(use_harmony=use_harmony)

    if backend == "file":
        from vllm.entrypoints.openai.responses.store.file import (
            FileResponsesStore,
        )

        path = envs.VLLM_RESPONSES_STORE_PATH
        if not path:
            raise ValueError(
                "VLLM_RESPONSES_STORE_PATH must be set when using the "
                "'file' responses store backend."
            )
        return FileResponsesStore(path, use_harmony=use_harmony)

    if backend == "redis":
        from vllm.entrypoints.openai.responses.store.redis import (
            RedisResponsesStore,
        )

        url = envs.VLLM_RESPONSES_STORE_REDIS_URL
        if not url:
            raise ValueError(
                "VLLM_RESPONSES_STORE_REDIS_URL must be set when using "
                "the 'redis' responses store backend."
            )
        return RedisResponsesStore(url, use_harmony=use_harmony)

    raise ValueError(f"Unknown responses store backend: {backend!r}")
