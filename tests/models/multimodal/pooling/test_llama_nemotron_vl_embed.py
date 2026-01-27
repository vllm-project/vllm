# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for LlamaNemotronVL embedding model (nvidia/llama-nemotron-embed-vl-1b-v2).

This model uses SigLIP vision encoder with bidirectional LLaMA for embeddings.
"""

import pytest
import torch
from transformers import AutoModel

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
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
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    # Run HF inference using the model's encode_queries/encode_documents API
    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs = []
        for text, image in zip(input_texts, input_images):
            with torch.inference_mode():
                if text.startswith(QUERY_PREFIX):
                    # Strip prefix and use encode_queries for query texts
                    query_text = text[len(QUERY_PREFIX):]
                    embedding = hf_model.model.encode_queries([query_text])
                elif text.startswith(PASSAGE_PREFIX):
                    # Strip prefix and use encode_documents for passages/images
                    passage_text = text[len(PASSAGE_PREFIX):]
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
                        f"Text must start with '{QUERY_PREFIX}' or "
                        f"'{PASSAGE_PREFIX}'"
                    )

                hf_outputs.append(embedding[0].tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
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
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test image embedding."""
    input_texts_images = [
        (text, asset.pil_image)
        for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
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
