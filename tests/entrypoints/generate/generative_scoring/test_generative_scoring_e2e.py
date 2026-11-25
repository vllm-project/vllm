# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the Generative Scoring API.

Tests verify the full HTTP request/response flow using RemoteOpenAIServer.
"""

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server(request: pytest.FixtureRequest):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
    ]
    args.extend(getattr(request, "param", []))
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


class TestGenerativeScoringAPI:
    """End-to-end tests for the Generative Scoring API."""

    @pytest.mark.asyncio
    async def test_basic_score_and_response_structure(self, server: RemoteOpenAIServer):
        """Test basic generative scoring request and verify response structure."""
        response = requests.post(
            server.url_for("generative_scoring"),
            json={
                "model": MODEL_NAME,
                "query": "Is Paris the capital of France? Answer Yes or No: ",
                "items": ["Paris is beautiful.", "London is rainy."],
                "label_token_ids": [9454, 2753],
            },
        )
        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Verify response structure
        assert data["id"].startswith("generative-scoring-")
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
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    @pytest.mark.asyncio
    async def test_multiple_items(self, server: RemoteOpenAIServer):
        """Test generative scoring request with multiple items."""
        response = requests.post(
            server.url_for("generative_scoring"),
            json={
                "model": MODEL_NAME,
                "query": "Is this city a capital? ",
                "items": ["Paris", "London", "Berlin", "New York", "Tokyo"],
                "label_token_ids": [9454, 2753],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 5

    @pytest.mark.asyncio
    async def test_validation_missing_label_token_ids(self, server: RemoteOpenAIServer):
        """Test that missing label_token_ids returns a validation error."""
        response = requests.post(
            server.url_for("generative_scoring"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": ["item1", "item2"],
            },
        )
        # Missing required field returns 400 (manual JSON parsing)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_validation_empty_items(self, server: RemoteOpenAIServer):
        """Test that empty items returns an error."""
        response = requests.post(
            server.url_for("generative_scoring"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": [],
                "label_token_ids": [100, 200],
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label_token_ids,expected_status",
        [
            ([9999999999, 9999999998], 400),  # Out of vocab range
        ],
        ids=["invalid_token_ids"],
    )
    async def test_validation_errors(
        self, server: RemoteOpenAIServer, label_token_ids, expected_status
    ):
        """Test validation errors for various invalid inputs."""
        response = requests.post(
            server.url_for("generative_scoring"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": ["item1"],
                "label_token_ids": label_token_ids,
            },
        )
        assert response.status_code == expected_status

    @pytest.mark.asyncio
    async def test_score_consistency(self, server: RemoteOpenAIServer):
        """Test that scores are deterministic across identical requests."""
        request_body = {
            "model": MODEL_NAME,
            "query": "Is this consistent? ",
            "items": ["Yes it is."],
            "label_token_ids": [100, 200],
        }

        r1 = requests.post(server.url_for("generative_scoring"), json=request_body)
        r2 = requests.post(server.url_for("generative_scoring"), json=request_body)

        assert r1.status_code == 200 and r2.status_code == 200
        r1_score = r1.json()["data"][0]["score"]
        r2_score = r2.json()["data"][0]["score"]
        assert abs(r1_score - r2_score) < 1e-6

    @pytest.mark.parametrize("item_first", [False, True])
    @pytest.mark.parametrize("server", [["--no-enable-prefix-caching"]], indirect=True)
    def test_long_text_matches_token_ids(self, server: RemoteOpenAIServer, item_first):
        tokenizer = get_tokenizer(MODEL_NAME)
        text = " " * (511 * tokenizer.max_chars_per_token) + "hello world"
        query, item = ("short query", text) if item_first else (text, "short item")
        request = {
            "model": MODEL_NAME,
            "query": query,
            "items": [item],
            "item_first": item_first,
            "label_token_ids": [100, 200],
        }
        text_response = requests.post(
            server.url_for("generative_scoring"), json=request, timeout=60
        )
        token_response = requests.post(
            server.url_for("generative_scoring"),
            json={
                **request,
                "query": tokenizer.encode(query, add_special_tokens=True),
                "items": [tokenizer.encode(item, add_special_tokens=False)],
            },
            timeout=60,
        )
        assert text_response.status_code == 200, text_response.text
        assert token_response.status_code == 200, token_response.text
        text_data, token_data = text_response.json(), token_response.json()
        assert text_data["usage"] == token_data["usage"]
        assert text_data["data"][0]["score"] == pytest.approx(
            token_data["data"][0]["score"], abs=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
