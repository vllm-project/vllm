# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import pytest
from transformers import AutoModel

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from vllm.entrypoints.pooling.score.utils import ScoreMultiModalParam

from ....conftest import HfRunner, VllmRunner

MODELS = ["jinaai/jina-reranker-m0"]

MM_PROCESSOR_KWARGS = {
    "min_pixels": 3136,
    "max_pixels": 602112,
}

LIMIT_MM_PER_PROMPT = {"image": 2}

CHECKPOINT_TO_HF_MAPPER = {
    "visual.": "model.visual.",
    "model.": "model.language_model.",
}

# Shared long text for test data
LONG_TEXT_DOC = """We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient
web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML
into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding
large language models. The models effectiveness results from two key innovations: (1) a three-stage
data synthesis pipeline that generates high quality, diverse training data by iteratively drafting,
refining, and critiquing web content extraction; and (2) a unified training framework combining
continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that
ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated
benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly
lower computational requirements."""  # noqa: E501

# Test data for different scenarios
TEXT_IMAGE_TEST_DATA = {
    "query": [{"text": "slm markdown"}],
    "documents": [
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
        },
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
        },
    ],
}

TEXT_TEXT_TEST_DATA = {
    "query": [{"text": "slm markdown"}],
    "documents": [
        {"text": LONG_TEXT_DOC},
        {"text": "数据提取么？为什么不用正则啊,你用正则不就全解决了么?"},
    ],
}

IMAGE_TEXT_TEST_DATA = {
    "query": [
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
        }
    ],
    "documents": [
        {"text": LONG_TEXT_DOC},
        {"text": "数据提取么?为什么不用正则啊,你用正则不就全解决了么?"},
    ],
}

IMAGE_IMAGE_TEST_DATA = {
    "query": [
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
        }
    ],
    "documents": [
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
        },
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
        },
    ],
}

TEXT_MIXED_DOCS_TEST_DATA = {
    "query": [{"text": "slm markdown"}],
    "documents": [
        {"text": LONG_TEXT_DOC},
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
        },
        {"text": "数据提取么？为什么不用正则啊,你用正则不就全解决了么?"},
        {
            "image": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
        },
    ],
}


def _normalize_image(image_val: str) -> str:
    """Normalize image value to proper format for HF model."""
    return (
        image_val
        if image_val.startswith(("http://", "https://"))
        else f"data:image/png;base64,{image_val}"
    )


def create_score_multimodal_param(
    content_parts: list[dict],
) -> ScoreMultiModalParam:
    """
    Create a ScoreMultiModalParam from a list of content dictionaries.

    Each dict supports the following formats:
    - Text: {'text': 'content'}
    - Image URL: {'image': 'https://...'}
    - Image Base64: {'image': 'base64_str'}
    """
    formatted_content = []

    for part in content_parts:
        if "text" in part:
            formatted_content.append(
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=part["text"],
                )
            )
        elif "image" in part:
            image_val = part["image"]
            if image_val.startswith(("http://", "https://")):
                formatted_content.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": image_val},
                    )
                )
            else:
                formatted_content.append(
                    ChatCompletionContentPartImageEmbedsParam(
                        type="image_embeds", image_embeds=image_val
                    )
                )

    return ScoreMultiModalParam(content=formatted_content)


def _run_vllm(
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query_strs: list[dict[str, str]],
    document_strs: list[dict[str, str]],
) -> list[float]:
    """Run vLLM reranker and return scores."""
    query = create_score_multimodal_param(query_strs)
    documents = create_score_multimodal_param(document_strs)

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_num_seqs=2,
        max_model_len=2048,
        mm_processor_kwargs=MM_PROCESSOR_KWARGS,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
    ) as vllm_model:
        outputs = vllm_model.llm.score(query, documents)

    return [output.outputs.score for output in outputs]


def _run_hf(
    hf_runner: type[HfRunner],
    model: str,
    dtype: str,
    query_strs: list[dict[str, str]],
    document_strs: list[dict[str, str]],
) -> list[float]:
    """Run HuggingFace reranker and return scores."""
    query = query_strs[0]
    if "text" in query:
        query_type = "text"
        query_data = query["text"]
    elif "image" in query:
        query_type = "image"
        query_data = _normalize_image(query["image"])
    else:
        raise ValueError("Unsupported query format")

    # Separate documents by type
    text_docs: list[str] = []
    image_docs: list[str] = []
    text_indices: list[int] = []
    image_indices: list[int] = []

    for idx, doc in enumerate(document_strs):
        if "text" in doc:
            text_docs.append(doc["text"])
            text_indices.append(idx)
        elif "image" in doc:
            image_docs.append(_normalize_image(doc["image"]))
            image_indices.append(idx)
        else:
            raise ValueError(f"Unsupported document format at index {idx}")

    scores: list[None | float] = [None] * len(document_strs)

    with hf_runner(
        model,
        dtype=dtype,
        trust_remote_code=True,
        auto_cls=AutoModel,
        model_kwargs={"key_mapping": CHECKPOINT_TO_HF_MAPPER},
    ) as hf_model:
        # Score text documents
        if text_docs:
            text_scores = hf_model.model.compute_score(
                [[query_data, d] for d in text_docs],
                max_length=2048,
                query_type=query_type,
                doc_type="text",
            )
            for i, s in zip(text_indices, text_scores):
                scores[i] = s

        # Score image documents
        if image_docs:
            image_scores = hf_model.model.compute_score(
                [[query_data, d] for d in image_docs],
                max_length=2048,
                query_type=query_type,
                doc_type="image",
            )
            for i, s in zip(image_indices, image_scores):
                scores[i] = s

    assert all(s is not None for s in scores)
    return cast(list[float], scores)


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query_strs: list[dict[str, str]],
    document_strs: list[dict[str, str]],
) -> None:
    """Run comparison test between vLLM and HuggingFace implementations."""
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    vllm_outputs = _run_vllm(vllm_runner, model, dtype, query_strs, document_strs)
    hf_outputs = _run_hf(hf_runner, model, dtype, query_strs, document_strs)

    # Compare outputs
    assert len(hf_outputs) == len(vllm_outputs), (
        f"Output length mismatch: HF={len(hf_outputs)}, vLLM={len(vllm_outputs)}"
    )

    for i, (hf_score, vllm_score) in enumerate(zip(hf_outputs, vllm_outputs)):
        assert hf_score == pytest.approx(vllm_score, rel=0.02), (
            f"Score mismatch at index {i}: HF={hf_score}, vLLM={vllm_score}"
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_text_image(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Visual Documents Reranking"""
    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        TEXT_IMAGE_TEST_DATA["query"],
        TEXT_IMAGE_TEST_DATA["documents"],
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_text_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Textual Documents Reranking"""
    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        TEXT_TEXT_TEST_DATA["query"],
        TEXT_TEXT_TEST_DATA["documents"],
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_image_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Image Querying for Textual Documents"""
    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        IMAGE_TEXT_TEST_DATA["query"],
        IMAGE_TEXT_TEST_DATA["documents"],
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_image_image(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Image Querying for Image Documents"""
    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        IMAGE_IMAGE_TEST_DATA["query"],
        IMAGE_IMAGE_TEST_DATA["documents"],
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_text_mixed_documents(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Text Query for Mixed Text and Image Documents"""
    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        TEXT_MIXED_DOCS_TEST_DATA["query"],
        TEXT_MIXED_DOCS_TEST_DATA["documents"],
    )
