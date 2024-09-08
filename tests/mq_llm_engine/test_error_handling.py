import asyncio
import pytest
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)

from vllm.engine.arg_utils import AsyncEngineArgs

@pytest.mark.asyncio(scope="module")
async def test_bad_startup():
    bad_engine_args = AsyncEngineArgs(model="Qwen/Qwen2-0.5B-Instruct",
                                      tensor_parallel_size=1234)

    async with asyncio.timeout(60.):
        async with build_async_engine_client_from_engine_args(
            bad_engine_args, False) as llm:
            assert llm is None




