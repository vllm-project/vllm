# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import pytest
import requests
import torch
import torch.nn.functional as F

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.entrypoints.pooling.scoring.protocol import ScoreResponse

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

EMBEDDING_SIZE = 512
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


def test_offline(vllm_runner):
    with vllm_runner(model_name, runner="pooling") as llm_runner:
        llm = llm_runner.get_llm()
        _test_offline_1_v_1(llm)
        _test_offline_1_v_n(llm)
        _test_offline_n_v_n(llm)
        _test_offline_token_embed_illegal_inputs(llm)
        assert llm.model_config.embedding_size == EMBEDDING_SIZE


def test_online():
    with RemoteOpenAIServer(model_name, ["--runner", "pooling"]) as server:
        _test_online_1_v_1(server)
        _test_online_1_v_n(server)
        _test_online_n_v_n(server)
        _test_online_token_embed_illegal_inputs(server)


def _test_offline_1_v_1(llm):
    # test llm.score
    outputs = llm.score(query, documents[0])
    assert len(outputs) == 1
    assert outputs[0].outputs.score == pytest.approx(REFERENCE_1_VS_1[0], abs=TOL)

    # test llm.encode
    outputs = llm.encode(documents[:1] + [query], pooling_task="token_embed")
    embeds = outputs[0].outputs.data.float()
    assert embeds.shape[0] == 2
    assert embeds.shape[-1] == EMBEDDING_SIZE

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
    assert embeds.shape[0] == len(documents) + 1

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)

    assert len(scores) == len(documents)
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
        assert embeds.shape[0] == 2

        doc_embeds = embeds[:-1]
        query_embeds = embeds[-1]

        scores = F.cosine_similarity(query_embeds, doc_embeds)
        assert scores[0] == pytest.approx(expected, abs=TOL)


def _test_offline_token_embed_illegal_inputs(llm):
    with pytest.raises(
        ValueError, match="The JinaForRanking model requires at least 2 inputs."
    ):
        llm.encode([query], pooling_task="token_embed")

    with pytest.raises(
        ValueError, match="The JinaForRanking model only supports text as input."
    ):
        llm.encode([1, 2, 3], pooling_task="token_embed")


def _get_scores(server, query, document):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": model_name,
            "queries": query,
            "documents": document,
        },
    )

    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    return [d.score for d in score.data]


def _get_embeds(server, prompts: list[str]):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "task": "token_embed",
            "input": prompts,
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    return torch.as_tensor([d.data for d in poolings.data][0]).float()


def _test_online_1_v_1(server):
    # test scoring api
    scores = _get_scores(server, query, documents[0])
    assert len(scores) == 1
    assert scores[0] == pytest.approx(REFERENCE_1_VS_1[0], abs=TOL)

    # test pooling api
    embeds = _get_embeds(server, [documents[0], query])
    assert embeds.shape[0] == 2
    assert embeds.shape[-1] == EMBEDDING_SIZE

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)
    assert scores[0] == pytest.approx(REFERENCE_1_VS_1[0], abs=TOL)


def _test_online_1_v_n(server):
    # test scoring api
    scores = _get_scores(server, query, documents)
    assert len(scores) == len(documents)

    for expected, actual in zip(REFERENCE_1_VS_N, scores):
        assert actual == pytest.approx(expected, abs=TOL)

    # test pooling api
    embeds = _get_embeds(server, documents + [query])
    assert embeds.shape[0] == len(documents) + 1

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)

    assert len(scores) == len(documents)
    for expected, actual in zip(REFERENCE_1_VS_N, scores):
        assert actual == pytest.approx(expected, abs=TOL)


def _test_online_n_v_n(server):
    # test scoring api
    scores = _get_scores(server, [query] * len(documents), documents)
    assert len(scores) == len(documents)

    for expected, actual in zip(REFERENCE_1_VS_1, scores):
        assert actual == pytest.approx(expected, abs=TOL)

    # test pooling api
    for doc, expected in zip(documents, REFERENCE_1_VS_1):
        embeds = _get_embeds(server, [doc, query])
        assert embeds.shape[0] == 2

        doc_embeds = embeds[:-1]
        query_embeds = embeds[-1]

        scores = F.cosine_similarity(query_embeds, doc_embeds)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(expected, abs=TOL)


def _test_online_token_embed_illegal_inputs(server):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "task": "token_embed",
            "input": [query],
            "encoding_format": "float",
        },
    )
    assert response.json()["error"]["message"].startswith(
        "The JinaForRanking model requires at least 2 inputs."
    )

    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "task": "token_embed",
            "input": [1, 2, 3],
            "encoding_format": "float",
        },
    )
    assert response.json()["error"]["message"].startswith(
        "The JinaForRanking model only supports text as input."
    )

    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "task": "token_embed",
            "messages": [
                {
                    "role": "user",
                    "content": "The cat sat on the mat.",
                }
            ],
            "encoding_format": "float",
        },
    )
    assert response.json()["error"]["message"].startswith(
        "The JinaForRanking does not support chat Request."
    )
