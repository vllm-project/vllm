# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColQwen3 late interaction model for multi-modal retrieval.

ColQwen3 is a multi-vector retrieval model based on Qwen3-VL backbone with
ColBERT-style late interaction scoring (MaxSim). It produces per-token
embeddings for both text and image inputs.
"""

import pytest
import torch

from ....conftest import VllmRunner

MODELS = [
    "TomoroAI/tomoro-colqwen3-embed-4b",
    "OpenSearch-AI/Ops-Colqwen3-4B",
]

EMBED_DIMS = {
    "TomoroAI/tomoro-colqwen3-embed-4b": 320,
    "OpenSearch-AI/Ops-Colqwen3-4B": 2560,
}

TEXT_QUERIES = [
    "What is the capital of France?",
    "Describe the contents of the document.",
]

TEXT_DOCUMENTS = [
    "The capital of France is Paris.",
    "This document contains important financial data.",
]

DTYPE = "half"


def _run_token_embed_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify per-token embedding shape and L2 normalization."""
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.token_embed([TEXT_QUERIES[0]])

        assert len(outputs) == 1
        emb = torch.tensor(outputs[0])
        # Token embeddings should be 2D: [num_tokens, embed_dim]
        assert emb.dim() == 2
        assert emb.shape[1] == EMBED_DIMS[model]
        assert emb.shape[0] > 1

        # Verify L2 normalization
        norms = torch.norm(emb, p=2, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            rtol=1e-2,
            atol=1e-2,
        )


def _run_late_interaction_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify MaxSim scoring matches manual computation."""
    from vllm.entrypoints.pooling.score.utils import compute_maxsim_score

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed([TEXT_QUERIES[0]])
        d_outputs = vllm_model.token_embed([TEXT_DOCUMENTS[0]])

        q_emb = torch.tensor(q_outputs[0])
        d_emb = torch.tensor(d_outputs[0])

        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        vllm_scores = vllm_model.score(TEXT_QUERIES[0], TEXT_DOCUMENTS[0])

        assert len(vllm_scores) == 1
        assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


def _run_relevance_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify that relevant documents score higher than irrelevant ones."""
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather forecast shows rain tomorrow.",
        "Deep learning uses neural networks for complex tasks.",
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        scores = vllm_model.score(query, documents)

        assert len(scores) == 3
        assert scores[0] > scores[1], "ML doc should score higher than weather doc"
        assert scores[2] > scores[1], "DL doc should score higher than weather doc"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_token_embed(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_token_embed_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_late_interaction_scoring(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_late_interaction_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_relevance_ordering(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_relevance_test(vllm_runner, model, dtype=dtype)
