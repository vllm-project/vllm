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
BOS_MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def server(request):
    model_name = getattr(request, "param", MODEL_NAME)
    args = [
        "--dtype",
        "float32" if model_name == BOS_MODEL_NAME else "bfloat16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "1" if model_name == BOS_MODEL_NAME else "32",
    ]
    if model_name == BOS_MODEL_NAME:
        args += ["--no-enable-prefix-caching", "--kv-cache-memory-bytes", "268435456"]
    with RemoteOpenAIServer(model_name, args) as remote_server:
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


@pytest.mark.parametrize("server", [BOS_MODEL_NAME], indirect=True)
class TestGenerativeScoringSpecialTokens:
    @pytest.mark.parametrize(
        "item",
        ["Document: Paris is the capital of France.", []],
        ids=["text", "empty-ids"],
    )
    def test_item_first_matches_explicit_opt_prompt(self, server, item):
        """Score the same sequence as a prompt with explicit OPT token IDs."""
        tokenizer = get_tokenizer(BOS_MODEL_NAME)
        query = "\nQuestion: Is this city the capital of France?\nAnswer:"
        item_ids = tokenizer.encode(item, add_special_tokens=False) if item else []
        expected_ids = [
            tokenizer.bos_token_id,
            *item_ids,
            *tokenizer.encode(query, add_special_tokens=False),
        ]
        common = {
            "model": BOS_MODEL_NAME,
            "label_token_ids": [
                tokenizer.encode(label, add_special_tokens=False)[0]
                for label in (" Yes", " No")
            ],
        }
        actual = requests.post(
            server.url_for("generative_scoring"),
            json=common | {"query": query, "items": [item], "item_first": True},
        )
        reference = requests.post(
            server.url_for("generative_scoring"),
            json=common | {"query": expected_ids, "items": [[]]},
        )

        assert actual.status_code == reference.status_code == 200
        actual_result = actual.json()
        reference_result = reference.json()
        assert actual_result["usage"] == reference_result["usage"]
        assert actual_result["data"][0]["score"] == pytest.approx(
            reference_result["data"][0]["score"], abs=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
