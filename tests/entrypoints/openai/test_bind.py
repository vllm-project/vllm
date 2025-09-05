# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import socket
from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests

from vllm.utils import is_valid_ipv4_address, is_valid_ipv6_address
from vllm.version import __version__ as VLLM_VERSION

from ...utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope='function')
def server_args(request: pytest.FixtureRequest) -> list[str]:
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


@pytest.fixture(scope="function")
def server(server_args):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        *server_args,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest.mark.parametrize(
    ("server_args", "expected_addrs", "unexpected_addrs"),
    [
        pytest.param([], ["127.0.0.1", "::1"], [], id="default"),
        pytest.param(["--host=0.0.0.0"], ["127.0.0.1"], ["::1"],
                     id="0-0-0-0-ipv4"),
        pytest.param(["--host=127.0.0.1"], ["127.0.0.1"], ["::1"],
                     id="127-0-0-1-ipv4"),
        pytest.param(["--host=::1"], ["::1"], ["127.0.0.1"], id="_-_-1-ipv4"),
        pytest.param(["--host=::"], ["127.0.0.1", "::1"], [], id="_-_-all"),
        pytest.param(["--host=localhost"], ["127.0.0.1", "::1"], [],
                     id="localhost"),
    ],
    indirect=["server_args"],
)
@pytest.mark.asyncio
async def test_bind_ipv4_ipv6(server: RemoteOpenAIServer,
                              expected_addrs: list[str],
                              unexpected_addrs: list[str]):
    # if the test system lacks IPv4 or IPv6, move addresses of those types
    # to unexpected_addrs
    has_ipv4, has_ipv6 = False, False
    for family, _, _, _, _ in socket.getaddrinfo(None,
                                                 server.port,
                                                 type=socket.SOCK_STREAM,
                                                 flags=socket.AI_PASSIVE):
        if family == socket.AF_INET:
            has_ipv4 = True
        if family == socket.AF_INET6:
            has_ipv6 = True
    for addr in expected_addrs:
        if (not has_ipv6 and is_valid_ipv6_address(addr)) or (
                not has_ipv4 and is_valid_ipv4_address(addr)):
            expected_addrs.remove(addr)
            unexpected_addrs.append(addr)

    for addr in expected_addrs:
        response = requests.get(server.url_for_host(addr, "health"), timeout=1)
        assert response.status_code == HTTPStatus.OK

    for addr in unexpected_addrs:
        with pytest.raises(requests.ConnectionError):
            _response = requests.get(server.url_for_host(addr, "health"),
                                     timeout=1)
