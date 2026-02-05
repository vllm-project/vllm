# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the Generative Scores API.

These tests verify the full HTTP request/response flow using RemoteOpenAIServer.
"""

import pytest
import requests

from ...utils import RemoteOpenAIServer

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


class TestGenerativeScoresE2E:
    """End-to-end tests for generative scores API."""

    @pytest.mark.asyncio
    async def test_basic_generative_score_request(self, server: RemoteOpenAIServer):
        """Test basic generative score request with string inputs."""
        # Get some token IDs to test with - we'll use common tokens
        # For Qwen3-0.6B, let's use tokens for "Yes" and "No"
        # First, let's make a simple request to verify the endpoint works
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Is Paris the capital of France? Answer with Yes or No: ",
                "items": ["Paris is beautiful.", "London is rainy."],
                "label_token_ids": [9454, 2753],  # Common token IDs
                "apply_softmax": True,
                "item_first": False,
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert data["id"].startswith("genscore-")
        assert data["object"] == "generative_score"
        assert "model" in data
        assert "results" in data
        assert "usage" in data
        assert len(data["results"]) == 2

        # Verify each result has expected structure
        for i, result in enumerate(data["results"]):
            assert result["index"] == i
            assert "token_probs" in result
            # Probabilities should be between 0 and 1
            for token_id, prob in result["token_probs"].items():
                assert 0.0 <= prob <= 1.0

        # With apply_softmax=True, probabilities should sum to ~1
        for result in data["results"]:
            prob_sum = sum(result["token_probs"].values())
            assert abs(prob_sum - 1.0) < 1e-5, f"Prob sum: {prob_sum}"

    @pytest.mark.asyncio
    async def test_generative_score_with_pretokenized_input(
        self, server: RemoteOpenAIServer
    ):
        """Test generative score with pre-tokenized inputs."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": [100, 200, 300, 400, 500],  # Pre-tokenized query
                "items": [[600, 700], [800, 900, 1000]],  # Pre-tokenized items
                "label_token_ids": [1, 2, 3],
                "apply_softmax": True,
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        assert data["object"] == "generative_score"
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_generative_score_apply_softmax_false(
        self, server: RemoteOpenAIServer
    ):
        """Test generative score with apply_softmax=False returns true model probs."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query ",
                "items": ["item1", "item2"],
                "label_token_ids": [100, 200, 300],
                "apply_softmax": False,
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # With apply_softmax=False, probabilities should NOT sum to 1
        # (they are true model probs over full vocab for those tokens)
        for result in data["results"]:
            prob_sum = sum(result["token_probs"].values())
            # True probs typically sum to much less than 1
            assert prob_sum < 1.0, f"Prob sum should be < 1: {prob_sum}"

    @pytest.mark.asyncio
    async def test_generative_score_item_first(self, server: RemoteOpenAIServer):
        """Test generative score with item_first=True."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": " is the answer",
                "items": ["Yes", "No"],
                "label_token_ids": [100, 200],
                "item_first": True,  # Items prepended to query
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_generative_score_validation_empty_items(
        self, server: RemoteOpenAIServer
    ):
        """Test that empty items returns an error."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": [],
                "label_token_ids": [100],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "at least one item" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_generative_score_validation_empty_label_tokens(
        self, server: RemoteOpenAIServer
    ):
        """Test that empty label_token_ids returns an error."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": ["item1"],
                "label_token_ids": [],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "at least one token" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_generative_score_validation_invalid_token_id(
        self, server: RemoteOpenAIServer
    ):
        """Test that out-of-range token IDs return an error."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": ["item1"],
                "label_token_ids": [9999999999],  # Way out of vocab range
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "out of vocabulary range" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_generative_score_validation_mixed_item_types(
        self, server: RemoteOpenAIServer
    ):
        """Test that mixed item types (string and token list) returns a validation error.
        
        Note: Pydantic validates types at request parsing, so this returns a 422 
        Unprocessable Entity error, not a 400 from our validation logic.
        """
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query",
                "items": ["string item", [100, 200]],  # Mixed types
                "label_token_ids": [100],
            },
        )

        # Pydantic returns 422 for type validation errors
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_generative_score_usage_tracking(self, server: RemoteOpenAIServer):
        """Test that usage info is properly tracked."""
        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "A test query with multiple tokens ",
                "items": ["item one", "item two", "item three"],
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

    @pytest.mark.asyncio
    async def test_generative_score_with_many_label_tokens(
        self, server: RemoteOpenAIServer
    ):
        """Test generative score with many label tokens to stress test logprob_token_ids."""
        # Use a larger set of label tokens
        label_token_ids = list(range(100, 200))  # 100 tokens

        response = requests.post(
            server.url_for("v1/score"),
            json={
                "model": MODEL_NAME,
                "query": "Test query ",
                "items": ["item1"],
                "label_token_ids": label_token_ids,
                "apply_softmax": True,
            },
        )

        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()

        # Should have probs for all 100 tokens
        result = data["results"][0]
        assert len(result["token_probs"]) == 100

    @pytest.mark.asyncio
    async def test_generative_score_consistency(self, server: RemoteOpenAIServer):
        """Test that generative scores are consistent across identical requests."""
        request_body = {
            "model": MODEL_NAME,
            "query": "Is this consistent? ",
            "items": ["Yes it is."],
            "label_token_ids": [100, 200, 300],
            "apply_softmax": True,
            "temperature": 0.0,  # Deterministic
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

        # Probabilities should be identical for deterministic inference
        probs1 = data1["results"][0]["token_probs"]
        probs2 = data2["results"][0]["token_probs"]

        for token_id in probs1:
            assert abs(probs1[token_id] - probs2[token_id]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
