# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that `/invocations` reports why it rejected a request body, the same
way the endpoint of the matching request type does."""

import pytest
import requests

from tests.utils import RemoteOpenAIServer

from .conftest import MODEL_NAME_SMOLLM

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

CHAT_BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
COMPLETION_BODY = {"prompt": "hi", "max_tokens": 1}
RERANK_BODY = {"query": "q", "documents": ["d"]}


@pytest.fixture
def embedding_server():
    args = ["--runner", "pooling", "--max-model-len", "128", "--enforce-eager"]
    with RemoteOpenAIServer(EMBEDDING_MODEL_NAME, args) as remote_server:
        yield remote_server


def _invocation_error(server: RemoteOpenAIServer, body: dict) -> dict:
    response = requests.post(server.url_for("invocations"), json=body)
    assert response.status_code == 400, response.text
    return response.json()["error"]


class TestGenerateInvocationValidation:
    @pytest.fixture(scope="class")
    def generate_server(self):
        args = ["--max-model-len", "2048", "--enforce-eager", "--max-num-seqs", "4"]
        with RemoteOpenAIServer(MODEL_NAME_SMOLLM, args) as remote_server:
            yield remote_server

    @pytest.mark.parametrize(
        ("body", "endpoint"),
        [(CHAT_BODY, "v1/chat/completions"), (COMPLETION_BODY, "v1/completions")],
        ids=["chat", "completion"],
    )
    def test_invalid_field_is_reported_like_the_matching_endpoint(
        self, generate_server, body, endpoint
    ):
        body = {**body, "temperature": "hot"}

        error = _invocation_error(generate_server, body)

        assert error["param"] == "body.temperature"
        expected = requests.post(generate_server.url_for(endpoint), json=body)
        assert error == expected.json()["error"]

    def test_completion_request_rejected_by_chat_validator_is_served(
        self, generate_server
    ):
        """A chat-only validator error does not block a valid completion body."""
        body = {**COMPLETION_BODY, "logprobs": 0, "logprob_token_ids": [1]}

        response = requests.post(generate_server.url_for("invocations"), json=body)

        assert response.status_code == 200, response.text
        assert response.json()["object"] == "text_completion"

    def test_request_type_validator_error_is_reported(self, generate_server):
        body = {**CHAT_BODY, "stream_options": {"include_usage": True}}

        error = _invocation_error(generate_server, body)

        assert error["param"] == "stream_options"


def test_invalid_field_of_lower_priority_type_is_reported(embedding_server):
    """A rerank body is not reported with the errors of the embedding request
    type, which is tried first."""
    error = _invocation_error(embedding_server, {**RERANK_BODY, "top_n": "x"})

    assert error["param"] == "body.top_n"
