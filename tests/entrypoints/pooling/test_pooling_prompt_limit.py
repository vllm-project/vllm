# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that VLLM_MAX_COMPLETION_PROMPTS is enforced on pooling/scoring routes.

The fan-out bound was originally only applied to /v1/completions
(CVE-2026-73559). These tests verify it is now also enforced on
/score, /rerank, /pooling, and /classify request models.
"""

import pytest

from vllm.exceptions import VLLMValidationError


def _clear_envs_cache():
    from vllm import envs

    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


class TestScoringPromptLimit:
    """Scoring and rerank request models reject oversized lists."""

    def test_score_documents_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import (
            ScoreQueriesDocumentsRequest,
        )

        with pytest.raises(VLLMValidationError, match="documents list length"):
            ScoreQueriesDocumentsRequest(model="m", queries=["q"], documents=["d"] * 10)

    def test_score_documents_within_limit(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import (
            ScoreQueriesDocumentsRequest,
        )

        req = ScoreQueriesDocumentsRequest(
            model="m", queries=["q"], documents=["d"] * 5
        )
        assert req.documents == ["d"] * 5

    def test_score_items_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import (
            ScoreQueriesItemsRequest,
        )

        with pytest.raises(VLLMValidationError, match="items list length"):
            ScoreQueriesItemsRequest(model="m", queries=["q"], items=["d"] * 10)

    def test_score_text_2_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import (
            ScoreTextRequest,
        )

        with pytest.raises(VLLMValidationError, match="text_2 list length"):
            ScoreTextRequest(model="m", text_1="q", text_2=["d"] * 10)

    def test_rerank_documents_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import RerankRequest

        with pytest.raises(VLLMValidationError, match="documents list length"):
            RerankRequest(model="m", query="q", documents=["d"] * 10)

    def test_rerank_within_limit(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import RerankRequest

        req = RerankRequest(model="m", query="q", documents=["d"] * 5)
        assert len(req.documents) == 5

    def test_scalar_input_unaffected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.scoring.protocol import RerankRequest

        req = RerankRequest(model="m", query="q", documents="single doc")
        assert req.documents == "single doc"


class TestPoolingPromptLimit:
    """PoolingCompletionRequest rejects oversized input lists."""

    def test_string_list_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            PoolingCompletionRequest,
        )

        with pytest.raises(VLLMValidationError, match="input list length"):
            PoolingCompletionRequest(model="m", input=["x"] * 10)

    def test_token_id_lists_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            PoolingCompletionRequest,
        )

        with pytest.raises(VLLMValidationError, match="input list length"):
            PoolingCompletionRequest(model="m", input=[[1]] * 10)

    def test_single_token_id_list_unaffected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            PoolingCompletionRequest,
        )

        req = PoolingCompletionRequest(model="m", input=[1, 2, 3, 4, 5, 6])
        assert req.input == [1, 2, 3, 4, 5, 6]

    def test_within_limit_accepted(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            PoolingCompletionRequest,
        )

        req = PoolingCompletionRequest(model="m", input=["x"] * 5)
        assert len(req.input) == 5

    def test_scalar_string_unaffected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            PoolingCompletionRequest,
        )

        req = PoolingCompletionRequest(model="m", input="hello world")
        assert req.input == "hello world"


class TestIOProcessorPromptLimit:
    """IOProcessorRequest rejects oversized data lists."""

    def test_data_list_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            IOProcessorRequest,
        )

        with pytest.raises(VLLMValidationError, match="data list length"):
            IOProcessorRequest(model="m", data=["x"] * 10)

    def test_data_list_within_limit(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.pooling.protocol import (
            IOProcessorRequest,
        )

        req = IOProcessorRequest(model="m", data=["x"] * 5)
        assert len(req.data) == 5


class TestClassifyPromptLimit:
    """ClassificationCompletionRequest rejects oversized input lists."""

    def test_input_list_rejected(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.classify.protocol import (
            ClassificationCompletionRequest,
        )

        with pytest.raises(VLLMValidationError, match="input list length"):
            ClassificationCompletionRequest(model="m", input=["x"] * 10)

    def test_within_limit_accepted(self, monkeypatch):
        monkeypatch.setenv("VLLM_MAX_COMPLETION_PROMPTS", "5")
        _clear_envs_cache()

        from vllm.entrypoints.pooling.classify.protocol import (
            ClassificationCompletionRequest,
        )

        req = ClassificationCompletionRequest(model="m", input=["x"] * 5)
        assert len(req.input) == 5
