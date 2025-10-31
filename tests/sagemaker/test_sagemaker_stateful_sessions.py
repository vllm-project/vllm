# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests

from ..utils import RemoteOpenAIServer

# any model with a chat template should work here
HEADER_SAGEMAKER_CLOSED_SESSION_ID = "X-Amzn-SageMaker-Closed-Session-Id"
HEADER_SAGEMAKER_SESSION_ID = "X-Amzn-SageMaker-Session-Id"
HEADER_SAGEMAKER_NEW_SESSION_ID = "X-Amzn-SageMaker-New-Session-Id"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
CLOSE_BADREQUEST_CASES = [
    (
        "nonexistent_session_id",
        {"session_id": "nonexistent-session-id"},
        {},
        "session not found",
    ),
    ("malformed_close_request", {}, {"extra-field": "extra-field-data"}, None),
]


@pytest.fixture(scope="module")
def server():  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_create_session_badrequest(server: RemoteOpenAIServer):
    bad_response = requests.post(
        server.url_for("invocations"),
        json={"requestType": "NEW_SESSION", "extra-field": "extra-field-data"},
    )

    assert bad_response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_name,session_id_change,request_body_change,expected_error",
    CLOSE_BADREQUEST_CASES,
)
async def test_close_session_badrequest(
    server: RemoteOpenAIServer,
    test_name: str,
    session_id_change: dict[str, str],
    request_body_change: dict[str, str],
    expected_error: str | None,
):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(url, json={"requestType": "NEW_SESSION"})
    create_response.raise_for_status()
    valid_session_id, expiration = create_response.headers.get(
        HEADER_SAGEMAKER_NEW_SESSION_ID, ""
    ).split(";")
    assert valid_session_id

    close_request_json = {"requestType": "CLOSE"}
    if request_body_change:
        close_request_json.update(request_body_change)
    bad_session_id = session_id_change.get("session_id")
    bad_close_response = requests.post(
        url,
        headers={HEADER_SAGEMAKER_SESSION_ID: bad_session_id or valid_session_id},
        json=close_request_json,
    )

    # clean up created session, should succeed
    clean_up_response = requests.post(
        url,
        headers={HEADER_SAGEMAKER_SESSION_ID: valid_session_id},
        json={"requestType": "CLOSE"},
    )
    clean_up_response.raise_for_status()

    assert bad_close_response.status_code == 400
    if expected_error:
        assert expected_error in bad_close_response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_close_session_invalidrequest(
    server: RemoteOpenAIServer, client: openai.AsyncOpenAI
):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(url, json={"requestType": "NEW_SESSION"})
    create_response.raise_for_status()
    valid_session_id, expiration = create_response.headers.get(
        HEADER_SAGEMAKER_NEW_SESSION_ID, ""
    ).split(";")
    assert valid_session_id

    close_request_json = {"requestType": "CLOSE"}
    invalid_close_response = requests.post(
        url,
        # no headers to specify session_id
        json=close_request_json,
    )

    # clean up created session, should succeed
    clean_up_response = requests.post(
        url,
        headers={HEADER_SAGEMAKER_SESSION_ID: valid_session_id},
        json={"requestType": "CLOSE"},
    )
    clean_up_response.raise_for_status()

    assert invalid_close_response.status_code == 424
    assert "invalid session_id" in invalid_close_response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_session(server: RemoteOpenAIServer):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(url, json={"requestType": "NEW_SESSION"})
    create_response.raise_for_status()
    valid_session_id, expiration = create_response.headers.get(
        HEADER_SAGEMAKER_NEW_SESSION_ID, ""
    ).split(";")
    assert valid_session_id

    # test invocation with session id

    request_args = {
        "model": MODEL_NAME,
        "prompt": "what is 1+1?",
        "max_completion_tokens": 5,
        "temperature": 0.0,
        "logprobs": False,
    }

    invocation_response = requests.post(
        server.url_for("invocations"),
        headers={HEADER_SAGEMAKER_SESSION_ID: valid_session_id},
        json=request_args,
    )
    invocation_response.raise_for_status()

    # close created session, should succeed
    close_response = requests.post(
        url,
        headers={HEADER_SAGEMAKER_SESSION_ID: valid_session_id},
        json={"requestType": "CLOSE"},
    )
    close_response.raise_for_status()

    assert (
        close_response.headers.get(HEADER_SAGEMAKER_CLOSED_SESSION_ID)
        == valid_session_id
    )
