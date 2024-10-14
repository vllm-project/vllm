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
async def test_show_version(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/version")
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
async def test_check_health(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/health")

    assert response.status_code == HTTPStatus.OK
