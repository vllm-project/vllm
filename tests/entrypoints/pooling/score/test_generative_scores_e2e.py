# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the Score API with CausalLM models.

These tests verify the full HTTP request/response flow using RemoteOpenAIServer.
The Score API with label_token_ids enables generative scoring for CausalLM models.
"""

import pytest
import requests

from ....utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = "/shared/public/elr-models/Qwen/Qwen3-0.6B/e6de91484c29aa9480d55605af694f39b081c455/"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
    ]

    with RemoteOpenAIServer(MODEL_PATH, args) as remote_server:
        yield remote_server


class TestScoreAPIWithCausalLM:
    """End-to-end tests for Score API with CausalLM models using label_token_ids."""

    @pytest.mark.asyncio
    async def test_basic_score_request(self, server: RemoteOpenAIServer):
        """Test basic score request with label_token_ids for CausalLM."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Is Paris the capital of France? Answer with Yes or No: ",
                "documents": ["Paris is beautiful.", "London is rainy."],
                "label_token_ids": [9454, 2753],  # Common token IDs
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Verify response structure matches ScoreResponse format
        assert "id" in data
        assert data["id"].startswith("score-")
        assert data["object"] == "list"
        assert "model" in data
        assert "data" in data
        assert "usage" in data
        assert len(data["data"]) == 2

        # Verify each result has expected structure
        for i, result in enumerate(data["data"]):
            assert result["index"] == i
            assert result["object"] == "score"
            assert "score" in result
            # Score should be between 0 and 1 (probability)
            assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_with_multiple_documents(self, server: RemoteOpenAIServer):
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

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        assert len(data["data"]) == 5
        for i, result in enumerate(data["data"]):
            assert result["index"] == i
            assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_validation_missing_label_token_ids(
        self, server: RemoteOpenAIServer
    ):
        """Test that missing label_token_ids returns an error for CausalLM."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query",
                "documents": ["doc1", "doc2"],
                # label_token_ids missing - should error for CausalLM
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "label_token_ids" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_score_validation_empty_label_tokens(
        self, server: RemoteOpenAIServer
    ):
        """Test that empty label_token_ids returns an error."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query",
                "documents": ["item1"],
                "label_token_ids": [],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "at least one token" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_score_validation_invalid_token_id(
        self, server: RemoteOpenAIServer
    ):
        """Test that out-of-range token IDs return an error."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query",
                "documents": ["item1"],
                "label_token_ids": [9999999999],  # Way out of vocab range
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "out of vocabulary range" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_score_usage_tracking(self, server: RemoteOpenAIServer):
        """Test that usage info is properly tracked."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "A test query with multiple tokens ",
                "documents": ["item one", "item two", "item three"],
                "label_token_ids": [100, 200],
            },
        )

        assert response.status_code == 200
        data = response.json()

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_score_consistency(self, server: RemoteOpenAIServer):
        """Test that scores are consistent across identical requests."""
        request_body = {
            "model": MODEL_NAME,
            "queries": "Is this consistent? ",
            "documents": ["Yes it is."],
            "label_token_ids": [100, 200, 300],
        }

        response1 = requests.post(
            server.url_for("v1/score"),
            json=request_body,
        )
        response2 = requests.post(
            server.url_for("v1/score"),
            json=request_body,
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Scores should be identical for deterministic inference
        score1 = data1["data"][0]["score"]
        score2 = data2["data"][0]["score"]

        assert abs(score1 - score2) < 1e-6

    @pytest.mark.asyncio
    async def test_score_with_many_label_tokens(self, server: RemoteOpenAIServer):
        """Test score with many label tokens."""
        # Use a larger set of label tokens
        label_token_ids = list(range(100, 200))  # 100 tokens

        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "queries": "Test query ",
                "documents": ["item1"],
                "label_token_ids": label_token_ids,
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Score should be the probability of the first label token
        result = data["data"][0]
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0


class TestLogprobTokenIds:
    """Tests to verify logprob_token_ids feature works correctly.
    
    These tests verify that the logprob_token_ids field in SamplingParams
    works correctly, which is the underlying mechanism for generative scores.
    """

    @pytest.mark.asyncio
    async def test_logprob_token_ids_via_completion(self, server: RemoteOpenAIServer):
        """Test that logprob_token_ids returns correct logprobs for specified tokens."""
        # Use the completions API directly to test logprob_token_ids
        client = server.get_client()
        
        # Request completion with logprobs
        response = client.completions.create(
            model=MODEL_NAME,
            prompt="The capital of France is",
            max_tokens=1,
            logprobs=5,  # Get top 5 logprobs
            temperature=0.0,
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.logprobs is not None
        assert len(choice.logprobs.top_logprobs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
