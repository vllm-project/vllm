# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import random
import time
from typing import Callable

import openai
import pytest
import pytest_asyncio
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        "--load-format",
        "dummy",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=["completion", "chat"],
    argnames=["create_func_gen", "content_body"],
    argvalues=[
        (lambda x: x.completions.create, {
            "prompt": " ".join(['A'] * 10_000)
        }),
        (lambda x: x.chat.completions.create, {
            "messages": [{
                "role": "user",
                "content": " ".join(['A'] * 10_000)
            }]
        }),
    ],
)
async def test_with_and_without_truncate(
    server: RemoteOpenAIServer,
    client: openai.AsyncOpenAI,
    create_func_gen: Callable,
    content_body: dict,
):
    create_func = create_func_gen(client)
    body = {"model": MODEL_NAME, **content_body, "max_tokens": 10}

    num_requests = 10
    truncate_prompt_tokens = ([1000] * (num_requests // 2) + [None] *
                              (num_requests - num_requests // 2))
    random.shuffle(truncate_prompt_tokens)

    bodies = [{
        **body, "extra_body": {
            'truncate_prompt_tokens': t
        }
    } for t in truncate_prompt_tokens]

    async def get_status_code(**kwargs):
        try:
            await create_func(**kwargs)
            return 200
        except openai.APIStatusError as e:
            return e.status_code

    responses = await asyncio.gather(*[get_status_code(**b) for b in bodies])
    assert 500 not in responses


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=["single completion", "multiple completions", "chat"],
    argnames=["create_func_gen", "content_body"],
    argvalues=[
        (lambda x: x.completions.create, {
            "prompt": " ".join(['A'] * 300_000)
        }),
        (lambda x: x.completions.create, {
            "prompt": [" ".join(['A'] * 300_000)] * 2
        }),
        (lambda x: x.chat.completions.create, {
            "messages": [{
                "role": "user",
                "content": " ".join(['A'] * 300_000)
            }]
        }),
    ],
)
async def test_healthcheck_response_time(
    server: RemoteOpenAIServer,
    client: openai.AsyncOpenAI,
    create_func_gen: Callable,
    content_body: dict,
):
    num_requests = 50

    create_func = create_func_gen(client)
    body = {"model": MODEL_NAME, **content_body, "max_tokens": 10}

    def get_response_time(url):
        start_time = time.monotonic()
        res = requests.get(url)
        end_time = time.monotonic()
        assert res.status_code == 200
        return end_time - start_time

    no_load_response_time = get_response_time(server.url_for("health"))
    tasks = [
        asyncio.create_task(create_func(**body)) for _ in range(num_requests)
    ]
    await asyncio.sleep(1)  # give the tasks a chance to start running
    load_response_time = get_response_time(server.url_for("health"))

    with contextlib.suppress(openai.APIStatusError):
        await asyncio.gather(*tasks)

    assert load_response_time < 100 * no_load_response_time
    assert load_response_time < 0.1
