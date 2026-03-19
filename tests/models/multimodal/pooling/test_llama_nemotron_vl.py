# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the LlamaNemotronVL model family:
  - nvidia/llama-nemotron-embed-vl-1b-v2  (LlamaNemotronVLForCausalLM / embed)
  - nvidia/llama-nemotron-rerank-vl-1b-v2
      (LlamaNemotronVLForSequenceClassification / rerank)

Both variants share a SigLIP vision encoder with a bidirectional LLaMA backbone.
"""

from io import BytesIO
from pathlib import Path

import pybase64 as base64
import pytest
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from vllm.entrypoints.pooling.score.utils import ScoreMultiModalParam

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import ROCM_ENGINE_KWARGS
from ...conftest import patch_hf_vision_attn_for_rocm
from ...utils import check_embeddings_close

# Prefixes used by the model API
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Text prompts for text-only embedding
HF_TEXT_PROMPTS = [
    # T -> X (text embedding queries)
    f"{QUERY_PREFIX}The label of the object is stop sign",
    f"{QUERY_PREFIX}cherry blossom",
]

# Image prompts using the model's expected format
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        # I -> X (image embedding as passage/document)
        "stop_sign": f"{PASSAGE_PREFIX}<image>",
        "cherry_blossom": f"{PASSAGE_PREFIX}<image>",
    }
)

MODELS = ["nvidia/llama-nemotron-embed-vl-1b-v2"]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_texts: list[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    """Run embedding comparison test between HF and vLLM.

    NOTE: Run vLLM first to avoid CUDA initialization issues with multiprocessing.
    """
    # Run vLLM inference first
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        **ROCM_ENGINE_KWARGS,
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    # Run HF inference using the model's encode_queries/encode_documents API
    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        patch_hf_vision_attn_for_rocm(hf_model.model)
        hf_outputs = []
        for text, image in zip(input_texts, input_images):
            with torch.inference_mode():
                if text.startswith(QUERY_PREFIX):
                    # Strip prefix and use encode_queries for query texts
                    query_text = text[len(QUERY_PREFIX) :]
                    embedding = hf_model.model.encode_queries([query_text])
                elif text.startswith(PASSAGE_PREFIX):
                    # Strip prefix and use encode_documents for passages/images
                    passage_text = text[len(PASSAGE_PREFIX) :]
                    if image is not None:
                        # Image document - pass image to encode_documents
                        embedding = hf_model.model.encode_documents(
                            images=[image],
                            texts=[passage_text],
                        )
                    else:
                        # Text-only document
                        embedding = hf_model.model.encode_documents(
                            texts=[passage_text]
                        )
                else:
                    raise ValueError(
                        f"Text must start with '{QUERY_PREFIX}' or '{PASSAGE_PREFIX}'"
                    )

                hf_outputs.append(embedding[0].tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_text(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test text-only embedding."""
    input_texts_images = [(text, None) for text in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,  # type: ignore
        model,
        dtype=dtype,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test image embedding."""
    input_texts_images = [
        (text, asset.pil_image) for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        model,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Reranker tests — nvidia/llama-nemotron-rerank-vl-1b-v2
# ---------------------------------------------------------------------------

RERANKER_MODELS = ["nvidia/llama-nemotron-rerank-vl-1b-v2"]

# The tokenizer's built-in chat template is not suitable for the Score/Rerank
# APIs (it's inherited from the base LLM).  We must use the provided override.
_RERANKER_SCORE_TEMPLATE = (
    Path(__file__).parents[4]
    / "examples/pooling/score/template/nemotron-vl-rerank.jinja"
).read_text()

RERANKER_TEXT_QUERY = "How is AI improving the intelligence and capabilities of robots?"
RERANKER_TEXT_DOCS = [
    "AI enables robots to perceive, plan, and act autonomously.",
    (
        "A biological foundation model designed to analyze DNA, RNA, "
        "and protein sequences."
    ),
]

RERANKER_IMAGE_QUERY = "photo of a red stop sign on a street"


def _pil_to_data_uri(image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _run_hf_reranker(
    hf_runner: type[HfRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> list[float]:
    """Run HF reranker inference; docs is a list of (doc_text, doc_image|None)."""
    with hf_runner(
        model,
        dtype=dtype,
        trust_remote_code=True,
        auto_cls=AutoModelForSequenceClassification,
    ) as hf_model:
        patch_hf_vision_attn_for_rocm(hf_model.model)
        processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=2048,
        )
        examples = [
            {
                "question": query,
                "doc_text": doc_text if doc_text is not None else "",
                "doc_image": doc_image if doc_image is not None else "",
            }
            for doc_text, doc_image in docs
        ]
        batch_dict = processor.process_queries_documents_crossencoder(examples)
        batch_dict = {
            k: v.to(hf_model.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }
        with torch.inference_mode():
            logits = hf_model.model(**batch_dict, return_dict=True).logits
        # vLLM applies sigmoid activation to the raw logits before returning
        # scores; apply the same here so both sides are comparable.
        scores = torch.sigmoid(logits.squeeze(-1).float())
        return scores.detach().cpu().tolist()


def _run_vllm_reranker(
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> list[float]:
    """Run vLLM reranker inference; docs is a list of (doc_text, doc_image|None)."""
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        **ROCM_ENGINE_KWARGS,
    ) as vllm_model:
        has_images = any(img is not None for _, img in docs)

        if not has_images:
            # Text-only path: use the simple string score API.
            queries = [query] * len(docs)
            doc_texts = [doc_text for doc_text, _ in docs]
            outputs = vllm_model.score(
                queries,
                doc_texts,
                chat_template=_RERANKER_SCORE_TEMPLATE,
            )
        else:
            # Multimodal path: build ScoreMultiModalParam for each pair.
            query_params = [
                ScoreMultiModalParam(
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text=query,
                        )
                    ]
                )
            ] * len(docs)

            doc_params = []
            for doc_text, doc_image in docs:
                content: list = []
                if doc_image is not None:
                    content.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={"url": _pil_to_data_uri(doc_image)},
                        )
                    )
                if doc_text:
                    content.append(
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text=doc_text,
                        )
                    )
                doc_params.append(ScoreMultiModalParam(content=content))

            raw_outputs = vllm_model.llm.score(
                query_params,
                doc_params,
                chat_template=_RERANKER_SCORE_TEMPLATE,
            )
            outputs = [o.outputs.score for o in raw_outputs]

    return outputs


def _run_reranker_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> None:
    """Compare HF and vLLM reranker scores.

    NOTE: Run vLLM first to avoid CUDA initialization issues with multiprocessing.
    """
    vllm_scores = _run_vllm_reranker(vllm_runner, model, dtype, query, docs)
    hf_scores = _run_hf_reranker(hf_runner, model, dtype, query, docs)

    assert len(hf_scores) == len(vllm_scores), (
        f"Output length mismatch: HF={len(hf_scores)}, vLLM={len(vllm_scores)}"
    )
    for i, (hf_score, vllm_score) in enumerate(zip(hf_scores, vllm_scores)):
        assert hf_score == pytest.approx(vllm_score, rel=0.02), (
            f"Score mismatch at index {i}: HF={hf_score:.4f}, vLLM={vllm_score:.4f}"
        )


@pytest.mark.parametrize("model", RERANKER_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Test reranking with text-only query and text documents."""
    docs = [(text, None) for text in RERANKER_TEXT_DOCS]
    _run_reranker_test(hf_runner, vllm_runner, model, dtype, RERANKER_TEXT_QUERY, docs)


@pytest.mark.parametrize("model", RERANKER_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker_image_doc(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test reranking with text query against image documents."""
    docs = [(None, asset.pil_image) for asset in image_assets]
    _run_reranker_test(hf_runner, vllm_runner, model, dtype, RERANKER_IMAGE_QUERY, docs)
