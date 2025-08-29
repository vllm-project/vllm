# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests

from vllm.version import __version__ as VLLM_VERSION

MODEL_NAME = "microsoft/DialoGPT-small"


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_show_version(server):
    response = requests.get(server.url_for("version"))
    response.raise_for_status()

    assert response.json() == {"version": VLLM_VERSION}


@pytest.mark.asyncio
async def test_check_health(server):
    response = requests.get(server.url_for("health"))

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_request_cancellation(server):
    # clunky test: send an ungodly amount of load in with short timeouts
    # then ensure that it still responds quickly afterwards

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client(timeout=0.5)
    tasks = []
    # Request about 2 million tokens
    for _ in range(200):
        task = asyncio.create_task(
            client.chat.completions.create(messages=chat_input,
                                           model=MODEL_NAME,
                                           max_tokens=10000,
                                           extra_body={"min_tokens": 10000}))
        tasks.append(task)

    done, pending = await asyncio.wait(tasks,
                                       return_when=asyncio.ALL_COMPLETED)

    # Make sure all requests were sent to the server and timed out
    # (We don't want to hide other errors like 400s that would invalidate this
    # test)
    assert len(pending) == 0
    for d in done:
        with pytest.raises(openai.APITimeoutError):
            d.result()

    # If the server had not cancelled all the other requests, then it would not
    # be able to respond to this one within the timeout
    client = server.get_async_client(timeout=5)
    response = await client.chat.completions.create(messages=chat_input,
                                                    model=MODEL_NAME,
                                                    max_tokens=10)

    assert len(response.choices) == 1


@pytest.mark.asyncio
async def test_request_wrong_content_type(server):

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client()

    with pytest.raises(openai.APIStatusError):
        await client.chat.completions.create(
            messages=chat_input,
            model=MODEL_NAME,
            max_tokens=10000,
            extra_headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })


@pytest.mark.asyncio
async def test_server_load(server):
    # Check initial server load
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    initial_load = response.json().get("server_load")
    print(f"Initial server load: {initial_load}")
    assert initial_load == 0, f"Expected initial \
        server_load to be 0, but got {initial_load}"

    def make_long_completion_request():
        return requests.post(
            server.url_for("v1/chat/completions"),
            headers={"Content-Type": "application/json"},
            json={
                "model":
                MODEL_NAME,
                "messages": [{
                    "role":
                    "user",
                    "content":
                    "Give me a very long story with many details"
                }],
                "max_tokens":
                1000,
                "temperature":
                0,
                "stream":
                True,
            },
            stream=True,
        )

    # Start the completion request in a background thread.
    completion_future = asyncio.create_task(
        asyncio.to_thread(make_long_completion_request))

    await asyncio.sleep(0.5)

    # Check server load while the completion request is running.
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 1

    # Wait for the completion request to finish.
    await completion_future
    await asyncio.sleep(0.1)

    # Check server load after the completion request has finished.
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 0
