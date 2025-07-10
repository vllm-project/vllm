# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoModel

model_name = "jinaai/jina-reranker-m0"

mm_processor_kwargs = {
    "min_pixels": 3136,
    "max_pixels": 602112,
}

limit_mm_per_prompt = {"image": 2}


def vllm_reranker(model_name,
                  query,
                  documents,
                  query_type="text",
                  doc_type="text"):
    from vllm import LLM

    model = LLM(
        model=model_name,
        task="score",
        max_model_len=32768,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    def create_image_param(url: str):
        return {"type": "image_url", "image_url": {"url": f"{url}"}}

    if query_type == "image":
        query = {"content": [create_image_param(url) for url in query]}

    if doc_type == "image":
        documents = {"content": [create_image_param(url) for url in documents]}

    outputs = model.score(query, documents)

    return [output.outputs.score for output in outputs]


def hf_reranker(model_name,
                query,
                documents,
                query_type="text",
                doc_type="text"):

    checkpoint_to_hf_mapper = {
        "visual.": "model.visual.",
        "model.": "model.language_model.",
    }

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        key_mapping=checkpoint_to_hf_mapper).to("cuda").eval()

    data_pairs = [[query[0], d] for d in documents]

    scores = model.compute_score(data_pairs,
                                 max_length=2048,
                                 query_type=query_type,
                                 doc_type=doc_type)
    return scores


# Visual Documents Reranking
@pytest.mark.parametrize("model_name", [model_name])
def test_model_text_image(model_name):

    query = ["slm markdown"]
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "text", "image")
    vllm_outputs = vllm_reranker(model_name, query, documents, "text", "image")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.02)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.02)


# Textual Documents Reranking
@pytest.mark.parametrize("model_name", [model_name])
def test_model_text_text(model_name):

    query = ["slm markdown"]
    documents = [
        """We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient 
        web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML 
        into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding 
        large language models. The models effectiveness results from two key innovations: (1) a three-stage 
        data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, 
        refining, and critiquing web content extraction; and (2) a unified training framework combining 
        continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that 
        ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated 
        benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly 
        lower computational requirements.""",  # noqa: E501
        "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "text", "text")
    vllm_outputs = vllm_reranker(model_name, query, documents, "text", "text")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.02)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.02)


# Image Querying for Textual Documents
@pytest.mark.parametrize("model_name", [model_name])
def test_model_image_text(model_name):

    query = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
    ]
    documents = [
        """We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient
        web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML
        into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding
        large language models. The models effectiveness results from two key innovations: (1) a three-stage
        data synthesis pipeline that generates high quality, diverse training data by iteratively drafting,
        refining, and critiquing web content extraction; and (2) a unified training framework combining
        continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that
        ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated
        benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly
        lower computational requirements.""",  # noqa: E501
        "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "image", "text")
    vllm_outputs = vllm_reranker(model_name, query, documents, "image", "text")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.02)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.02)


# Image Querying for Image Documents
@pytest.mark.parametrize("model_name", [model_name])
def test_model_image_image(model_name):

    query = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
    ]
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "image", "image")
    vllm_outputs = vllm_reranker(model_name, query, documents, "image",
                                 "image")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.02)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.02)
