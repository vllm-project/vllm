# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import pytest
import torch.nn.functional as F

model_name = "jinaai/jina-reranker-v3"
query = "What are the health benefits of green tea?"
documents = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]


REFERENCE_1_VS_1 = [
    0.345703125,
    -0.10498046,
    0.314453125,
    -0.1376953125,
    0.3398437500,
    0.2539062,
]
REFERENCE_1_VS_N = [
    0.294921875,
    -0.16015625,
    0.189453125,
    -0.1708984375,
    0.2255859375,
    0.1640625,
]
TOL = 0.01


def _test_offline_1_v_1(llm):
    # test llm.score
    outputs = llm.score(query, documents[0])
    assert len(outputs) == 1
    assert outputs[0].outputs.score == pytest.approx(REFERENCE_1_VS_1[0], abs=TOL)

    # test llm.encode
    outputs = llm.encode(documents[:1] + [query], pooling_task="token_embed")
    embeds = outputs[0].outputs.data.float()

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)
    assert scores[0] == pytest.approx(REFERENCE_1_VS_1[0], abs=TOL)


def _test_offline_1_v_n(llm):
    # test llm.score
    outputs = llm.score(query, documents)
    assert len(outputs) == len(documents)

    for expected, output in zip(REFERENCE_1_VS_N, outputs):
        actual = output.outputs.score
        assert actual == pytest.approx(expected, abs=TOL)

    # test llm.encode
    outputs = llm.encode(documents + [query], pooling_task="token_embed")
    embeds = outputs[0].outputs.data.float()

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)

    for expected, actual in zip(REFERENCE_1_VS_N, scores):
        assert actual == pytest.approx(expected, abs=TOL)


def _test_offline_n_v_n(llm):
    # test llm.score
    outputs = llm.score([query] * len(documents), documents)
    assert len(outputs) == len(documents)

    for expected, output in zip(REFERENCE_1_VS_1, outputs):
        actual = output.outputs.score
        assert actual == pytest.approx(expected, abs=TOL)

    # test llm.encode
    for doc, expected in zip(documents, REFERENCE_1_VS_1):
        outputs = llm.encode([doc, query], pooling_task="token_embed")
        embeds = outputs[0].outputs.data.float()

        doc_embeds = embeds[:-1]
        query_embeds = embeds[-1]

        scores = F.cosine_similarity(query_embeds, doc_embeds)
        assert scores[0] == pytest.approx(expected, abs=TOL)


def test_offline(vllm_runner):
    with vllm_runner(model_name, runner="pooling") as llm_runner:
        llm = llm_runner.get_llm()
        _test_offline_1_v_1(llm)
        _test_offline_1_v_n(llm)
        _test_offline_n_v_n(llm)
