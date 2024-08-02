from http import HTTPStatus

import openai
import pytest
import requests

from vllm.version import __version__ as VLLM_VERSION

from ...utils import RemoteOpenAIServer

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


@pytest.mark.asyncio
async def test_show_version(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/version")
    response.raise_for_status()

    assert response.json() == {"version": VLLM_VERSION}


@pytest.mark.asyncio
async def test_check_health(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/health")

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_log_metrics(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/metrics")

    assert response.status_code == HTTPStatus.OK

@pytest.mark.asyncio
def test_get_liveness(client: openai.AsyncOpenAI):
    """Test the technical route /liveness"""
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/liveness")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"alive": "ok"}

@pytest.mark.asyncio
def test_get_readiness_ko(client: openai.AsyncOpenAI):
    """Test the technical route /readiness when the model is not loaded"""
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/readiness")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"ready": "ko"}

@pytest.mark.asyncio
def test_get_readiness_ok(client: openai.AsyncOpenAI):
    """Test the technical route /readiness when the model is fully loaded"""
    base_url = str(client.base_url)[:-3].strip("/")

    response = requests.get(base_url + "/readiness")
    
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"ready": "ok"}
