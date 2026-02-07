# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the Score API with CausalLM models.

Tests verify the full HTTP request/response flow using RemoteOpenAIServer.
"""

import pytest
import requests

from ....utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = "/shared/public/elr-models/Qwen/Qwen3-0.6B/e6de91484c29aa9480d55605af694f39b081c455/"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--enforce-eager",
        "--max-num-seqs", "32",
    ]
    with RemoteOpenAIServer(MODEL_PATH, args) as remote_server:
        yield remote_server


class TestScoreAPIWithCausalLM:
    """End-to-end tests for Score API with CausalLM models."""

    @pytest.mark.asyncio
    async def test_basic_score_and_response_structure(self, server: RemoteOpenAIServer):
        """Test basic score request and verify response structure."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Is Paris the capital of France? Answer Yes or No: ",
                "documents": ["Paris is beautiful.", "London is rainy."],
                "label_token_ids": [9454, 2753],
            },
        )
        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Verify response structure
        assert data["id"].startswith("score-")
        assert data["object"] == "list"
        assert "model" in data
        assert "usage" in data
        assert len(data["data"]) == 2

        # Verify each result
        for i, result in enumerate(data["data"]):
            assert result["index"] == i
            assert result["object"] == "score"
            assert 0.0 <= result["score"] <= 1.0

        # Verify usage tracking
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_multiple_documents(self, server: RemoteOpenAIServer):
        """Test score request with multiple documents."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Is this city a capital? ",
                "documents": ["Paris", "London", "Berlin", "New York", "Tokyo"],
                "label_token_ids": [9454, 2753],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 5

    @pytest.mark.asyncio
    async def test_validation_missing_label_token_ids(self, server: RemoteOpenAIServer):
        """Test that missing label_token_ids returns an error for CausalLM."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query",
                "documents": ["doc1", "doc2"],
            },
        )
        assert response.status_code == 400
        assert "label_token_ids" in response.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label_token_ids,expected_status",
        [
            ([100], 422),                      # Wrong count (1 instead of 2)
            ([100, 200, 300], 422),             # Wrong count (3 instead of 2)
            ([9999999999, 9999999998], 400),    # Out of vocab range
        ],
        ids=["single_token", "three_tokens", "invalid_token_ids"],
    )
    async def test_validation_errors(self, server: RemoteOpenAIServer, label_token_ids, expected_status):
        """Test validation errors for various invalid inputs."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query",
                "documents": ["item1"],
                "label_token_ids": label_token_ids,
            },
        )
        assert response.status_code == expected_status

    @pytest.mark.asyncio
    async def test_score_consistency(self, server: RemoteOpenAIServer):
        """Test that scores are deterministic across identical requests."""
        request_body = {
            "model": MODEL_NAME,
            "queries": "Is this consistent? ",
            "documents": ["Yes it is."],
            "label_token_ids": [100, 200],
        }

        r1 = requests.post(server.url_for("v1/score"), json=request_body)
        r2 = requests.post(server.url_for("v1/score"), json=request_body)

        assert r1.status_code == 200 and r2.status_code == 200
        assert abs(r1.json()["data"][0]["score"] - r2.json()["data"][0]["score"]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
