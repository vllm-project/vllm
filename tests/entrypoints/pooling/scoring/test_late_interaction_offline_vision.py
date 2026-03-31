# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from .util import make_base64_image, make_image_mm_param

MODEL_NAME = "vidore/colpali-v1.3-hf"


@pytest.fixture(scope="module")
def llm():
    # ROCm: Use FLEX_ATTENTION backend as it's the only attention backend
    # that supports encoder-only models on ROCm.
    attention_config = None
    if current_platform.is_rocm():
        attention_config = {"backend": "FLEX_ATTENTION"}

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        attention_config=attention_config,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_query_text_vs_docs_image(llm):
    """Score a text query against image documents via the multimodal path."""
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    blue_image = make_base64_image(64, 64, color=(0, 0, 255))

    query = "Describe the red object"
    image_docs = [
        make_image_mm_param(red_image),
        make_image_mm_param(blue_image),
    ]

    scores = llm.score(query, image_docs)

    assert len(scores) == 2
    assert scores[0].outputs.score > scores[1].outputs.score


@pytest.mark.skip_global_cleanup
def test_query_text_vs_docs_mix(llm) -> None:
    """Score a text query against a mix of text and image documents."""
    red_image = make_base64_image(64, 64, color=(255, 0, 0))

    query = "What is the capital of France?"
    documents: list = [
        "The capital of France is Paris.",
        make_image_mm_param(red_image),
    ]

    scores = llm.score(query, documents)

    assert len(scores) == 2
    assert scores[0].outputs.score > scores[1].outputs.score


@pytest.mark.skip_global_cleanup
def test_query_image_vs_docs_text(llm) -> None:
    """Score an image query against text documents."""
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    image_query = make_image_mm_param(red_image, text="red color")

    documents = [
        "Describe the red object.",
        "The capital of France is Paris.",
    ]

    scores = llm.score(image_query, documents)

    assert len(scores) == 2
    assert scores[0].outputs.score > scores[1].outputs.score
