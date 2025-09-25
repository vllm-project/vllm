# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests

from vllm.version import __version__ as VLLM_VERSION


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
async def test_request_cancellation(server, model_name):
    # clunky test: send an ungodly amount of load in with short timeouts
    # then ensure that it still responds quickly afterwards

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client(timeout=0.5)
    tasks = []
    for _ in range(20):
        task = asyncio.create_task(
            client.chat.completions.create(messages=chat_input,
                                           model=model_name,
                                           max_tokens=1000,
                                           extra_body={"min_tokens": 1000}))
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
                                                    model=model_name,
                                                    max_tokens=10)

    assert len(response.choices) == 1


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Server always returns 400 for wrong content type")
async def test_request_wrong_content_type(server, model_name):

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client()

    with pytest.raises(openai.APIStatusError):
        await client.chat.completions.create(
            messages=chat_input,
            model=model_name,
            max_tokens=1000,
            extra_headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })


@pytest.mark.asyncio
async def test_server_load(server):
    # Check initial server load
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 0

    def make_long_completion_request():
        return requests.post(
            server.url_for("v1/completions"),
            headers={"Content-Type": "application/json"},
            json={
                "prompt": "Give me a long story",
                "max_tokens": 1000,
                "temperature": 0,
            },
        )

    # Start the completion request in a background thread.
    completion_future = asyncio.create_task(
        asyncio.to_thread(make_long_completion_request))

    # Give a short delay to ensure the request has started.
    await asyncio.sleep(0.1)

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
