# SPDX-License-Identifier: Apache-2.0

import asyncio
from http import HTTPStatus
from typing import List

import openai
import pytest
import pytest_asyncio
import requests

from vllm.version import __version__ as VLLM_VERSION

from ...utils import RemoteOpenAIServer

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope='module')
def server_args(request: pytest.FixtureRequest) -> List[str]:
    """ Provide extra arguments to the server via indirect parametrization

    Usage:

    >>> @pytest.mark.parametrize(
    >>>     "server_args",
    >>>     [
    >>>         ["--disable-frontend-multiprocessing"],
    >>>         [
    >>>             "--model=NousResearch/Hermes-3-Llama-3.1-70B",
    >>>             "--enable-auto-tool-choice",
    >>>         ],
    >>>     ],
    >>>     indirect=True,
    >>> )
    >>> def test_foo(server, client):
    >>>     ...

    This will run `test_foo` twice with servers with:
    - `--disable-frontend-multiprocessing`
    - `--model=NousResearch/Hermes-3-Llama-3.1-70B --enable-auto-tool-choice`.

    """
    if not hasattr(request, "param"):
        return []

    val = request.param

    if isinstance(val, str):
        return [val]

    return request.param


@pytest.fixture(scope="module")
def server(server_args):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        *server_args,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.parametrize(
    "server_args",
    [
        pytest.param([], id="default-frontend-multiprocessing"),
        pytest.param(["--disable-frontend-multiprocessing"],
                     id="disable-frontend-multiprocessing")
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_show_version(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("version"))
    response.raise_for_status()

    assert response.json() == {"version": VLLM_VERSION}


@pytest.mark.parametrize(
    "server_args",
    [
        pytest.param([], id="default-frontend-multiprocessing"),
        pytest.param(["--disable-frontend-multiprocessing"],
                     id="disable-frontend-multiprocessing")
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_check_health(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))

    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server_args",
    [
        pytest.param(["--max-model-len", "10100"],
                     id="default-frontend-multiprocessing"),
        pytest.param(
            ["--disable-frontend-multiprocessing", "--max-model-len", "10100"],
            id="disable-frontend-multiprocessing")
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_request_cancellation(server: RemoteOpenAIServer):
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
async def test_request_wrong_content_type(server: RemoteOpenAIServer):

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
