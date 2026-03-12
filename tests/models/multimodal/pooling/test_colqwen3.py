# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColQwen3 late interaction model for multi-modal retrieval.

ColQwen3 is a multi-vector retrieval model based on Qwen3-VL backbone with
ColBERT-style late interaction scoring (MaxSim). It produces per-token
embeddings for both text and image inputs.
"""

import base64
from io import BytesIO

import pytest
import torch
from PIL import Image

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from vllm.entrypoints.pooling.score.utils import ScoreMultiModalParam

from ....conftest import VllmRunner

MODELS = [
    "TomoroAI/tomoro-colqwen3-embed-4b",
    "OpenSearch-AI/Ops-Colqwen3-4B",
    "nvidia/nemotron-colembed-vl-4b-v2",
]

EMBED_DIMS = {
    "TomoroAI/tomoro-colqwen3-embed-4b": 320,
    "OpenSearch-AI/Ops-Colqwen3-4B": 2560,
    "nvidia/nemotron-colembed-vl-4b-v2": 2560,
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
GPU_MEMORY_UTILIZATION = 0.7


def _make_base64_image(
    width: int = 64, height: int = 64, color: tuple[int, int, int] = (255, 0, 0)
) -> str:
    """Create a small solid-color PNG image and return its base64 data URI."""
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_image_mm_param(
    image_uri: str,
    text: str | None = None,
) -> ScoreMultiModalParam:
    """Build a ScoreMultiModalParam containing an image (and optional text)."""
    content: list = [
        ChatCompletionContentPartImageParam(
            type="image_url",
            image_url={"url": image_uri},
        ),
    ]
    if text is not None:
        content.append(
            ChatCompletionContentPartTextParam(type="text", text=text),
        )
    return ScoreMultiModalParam(content=content)


def _make_text_mm_param(text: str) -> ScoreMultiModalParam:
    """Build a ScoreMultiModalParam containing only text."""
    return ScoreMultiModalParam(
        content=[ChatCompletionContentPartTextParam(type="text", text=text)],
    )


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
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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


# ── Multimodal scoring tests ────────────────────────────────


def _run_multimodal_text_query_image_docs_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Score a text query against image documents via the multimodal path.

    Verifies that score_data_to_prompts correctly handles image content
    and produces valid MaxSim scores.
    """
    red_image = _make_base64_image(64, 64, color=(255, 0, 0))
    blue_image = _make_base64_image(64, 64, color=(0, 0, 255))

    query = "Describe the red object"
    image_docs = [
        _make_image_mm_param(red_image),
        _make_image_mm_param(blue_image),
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    ) as vllm_model:
        scores = vllm_model.llm.score(query, image_docs)

        assert len(scores) == 2
        for s in scores:
            assert isinstance(s.outputs.score, float)


def _run_multimodal_mixed_docs_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Score a text query against a mix of text and image documents.

    Ensures the late-interaction path handles heterogeneous document
    types (plain strings alongside ScoreMultiModalParam images) in
    a single call.
    """
    red_image = _make_base64_image(64, 64, color=(255, 0, 0))

    query = "What is the capital of France?"
    documents: list = [
        "The capital of France is Paris.",
        _make_image_mm_param(red_image),
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    ) as vllm_model:
        scores = vllm_model.llm.score(query, documents)

        assert len(scores) == 2
        for s in scores:
            assert isinstance(s.outputs.score, float)
        # Text document about France should score higher than a random image
        assert scores[0].outputs.score > scores[1].outputs.score


def _run_multimodal_image_query_text_docs_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Score an image query against text documents.

    Verifies the reverse direction: multimodal query with text-only
    documents through the late-interaction scoring path.
    """
    red_image = _make_base64_image(64, 64, color=(255, 0, 0))
    image_query = _make_image_mm_param(red_image, text="red color")

    documents = [
        "A bright red sports car.",
        "The weather forecast shows rain tomorrow.",
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    ) as vllm_model:
        scores = vllm_model.llm.score(image_query, documents)

        assert len(scores) == 2
        for s in scores:
            assert isinstance(s.outputs.score, float)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_multimodal_text_query_image_docs(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_multimodal_text_query_image_docs_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_multimodal_mixed_docs(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_multimodal_mixed_docs_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_multimodal_image_query_text_docs(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_multimodal_image_query_text_docs_test(vllm_runner, model, dtype=dtype)
