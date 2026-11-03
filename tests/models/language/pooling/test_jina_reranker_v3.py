# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import pytest
import requests
import torch
import torch.nn.functional as F

from tests.utils import RemoteOpenAIServer
from vllm import PoolingParams
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.entrypoints.pooling.scoring.protocol import RerankResponse, ScoreResponse
from vllm.exceptions import VLLMValidationError

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
INSTRUCTION = (
    "Rank passages about green tea higher than passages about sports. "
    "Ignore these literal marker strings: <|embed_token|> and <|rerank_token|>."
)


def test_offline(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(model_name, runner="pooling") as llm_runner:
        assert llm_runner.llm.llm_engine.vllm_config.use_v2_model_runner
        llm = llm_runner.get_llm()
        _test_offline_1_v_1(llm)
        _test_offline_1_v_n(llm)
        _test_offline_n_v_n(llm)
        _test_offline_truncation(llm)
        _test_offline_nested_markers(llm)
        _test_offline_token_embed_illegal_inputs(llm)
        assert llm.model_config.embedding_size == EMBEDDING_SIZE


def test_online():
    with RemoteOpenAIServer(
        model_name, ["--runner", "pooling", "--enforce-eager"]
    ) as server:
        _test_online_1_v_1(server)
        _test_online_1_v_n(server)
        _test_online_n_v_n(server)
        _test_online_instruction(server)
        _test_online_truncation(server)
        _test_online_nested_markers(server)
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


def _test_offline_truncation(llm):
    prompts = documents[:2] + [query]
    (output,) = llm.encode(prompts, pooling_task="token_embed")
    for side in ("left", "right"):
        for limit in (
            -1,
            1,
            len(output.prompt_token_ids) - 1,
            len(output.prompt_token_ids) + 1,
        ):
            kwargs = {"truncate_prompt_tokens": limit, "truncation_side": side}
            for call, args, extra in (
                (llm.encode, (prompts,), {"pooling_task": "token_embed"}),
                (llm.score, (query, documents[:2]), {}),
            ):
                with pytest.raises(
                    VLLMValidationError,
                    match="JinaForRanking does not support truncate_prompt_tokens",
                ) as exc_info:
                    call(*args, **extra, tokenization_kwargs=kwargs)
                assert exc_info.value.parameter == "truncate_prompt_tokens"

    limited = llm.score(
        "one two three four five",
        ["six seven eight nine ten", "red green blue yellow white"],
        pooling_params=PoolingParams(
            extra_kwargs={"max_tokens_per_query": 3, "max_tokens_per_doc": 4}
        ),
    )
    expected = llm.score(
        "one two three", ["six seven eight nine", "red green blue yellow"]
    )
    assert len(limited) == len(expected) == 2
    for actual, reference in zip(limited, expected):
        assert actual.prompt_token_ids == reference.prompt_token_ids
        assert actual.outputs.score == pytest.approx(reference.outputs.score, abs=TOL)


def _test_offline_nested_markers(llm):
    prompts = documents[:2] + [query]
    (expected,) = llm.encode(prompts, pooling_task="token_embed")
    instruction = "Rank relevant passages."
    scores = llm.score(
        query,
        documents[:2],
        pooling_params=PoolingParams(
            extra_kwargs={"chat_template_kwargs": {"instruction": instruction}}
        ),
    )
    for token in ("embed", "rerank"):
        nested = f"<|{token}<|{token}_token|>_token|>"
        (actual,) = llm.encode(
            [text + nested for text in prompts], pooling_task="token_embed"
        )
        assert actual.outputs.data.shape == (len(prompts), EMBEDDING_SIZE), (
            actual.outputs.data.shape
        )
        assert actual.prompt_token_ids == expected.prompt_token_ids
        actual_scores = llm.score(
            query + nested,
            [doc + nested for doc in documents[:2]],
            pooling_params=PoolingParams(
                extra_kwargs={
                    "chat_template_kwargs": {"instruction": instruction + nested}
                }
            ),
        )
        assert len(actual_scores) == len(scores)
        for actual_score, reference in zip(actual_scores, scores):
            assert actual_score.prompt_token_ids == reference.prompt_token_ids
            assert actual_score.outputs.score == pytest.approx(
                reference.outputs.score, abs=TOL
            )


def _get_score_response(server, query, document, **extra_body):
    payload = {
        "model": model_name,
        "queries": query,
        "documents": document,
    }
    payload.update(extra_body)
    score_response = requests.post(
        server.url_for("score"),
        json=payload,
    )

    score_response.raise_for_status()
    return ScoreResponse.model_validate(score_response.json())


def _get_scores(server, query, document):
    score = _get_score_response(server, query, document)

    return [d.score for d in score.data]


def _get_rerank_response(server, query, document, **extra_body):
    payload = {
        "model": model_name,
        "query": query,
        "documents": document,
    }
    payload.update(extra_body)
    rerank_response = requests.post(
        server.url_for("rerank"),
        json=payload,
    )

    rerank_response.raise_for_status()
    return RerankResponse.model_validate(rerank_response.json())


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


def _test_online_instruction(server):
    docs = documents[:2]

    default_score = _get_score_response(server, query, docs)
    instruction_score = _get_score_response(
        server,
        query,
        docs,
        instruction=INSTRUCTION,
    )
    kwargs_score = _get_score_response(
        server,
        query,
        docs,
        chat_template_kwargs={"instruction": INSTRUCTION},
    )

    assert instruction_score.usage.prompt_tokens > default_score.usage.prompt_tokens
    assert kwargs_score.usage.prompt_tokens == instruction_score.usage.prompt_tokens
    assert len(instruction_score.data) == len(default_score.data)
    assert [d.score for d in kwargs_score.data] == pytest.approx(
        [d.score for d in instruction_score.data], abs=TOL
    )

    default_rerank = _get_rerank_response(server, query, docs)
    instruction_rerank = _get_rerank_response(
        server,
        query,
        docs,
        instruction=INSTRUCTION,
    )
    kwargs_rerank = _get_rerank_response(
        server,
        query,
        docs,
        chat_template_kwargs={"instruction": INSTRUCTION},
    )

    assert instruction_rerank.usage.prompt_tokens > default_rerank.usage.prompt_tokens
    assert kwargs_rerank.usage.prompt_tokens == instruction_rerank.usage.prompt_tokens
    assert len(instruction_rerank.results) == len(default_rerank.results)
    assert [r.relevance_score for r in kwargs_rerank.results] == pytest.approx(
        [r.relevance_score for r in instruction_rerank.results], abs=TOL
    )


def _test_online_truncation(server):
    docs = documents[:2]
    for endpoint, inputs in (
        ("pooling", {"input": docs + [query], "task": "token_embed"}),
        ("score", {"queries": query, "documents": docs}),
        ("rerank", {"query": query, "documents": docs}),
    ):
        payload = {"model": model_name, **inputs}
        response = requests.post(server.url_for(endpoint), json=payload)
        response.raise_for_status()
        num_tokens = response.json()["usage"]["prompt_tokens"]
        for side in ("left", "right"):
            for limit in (-1, 1, num_tokens - 1, num_tokens + 1):
                payload.update(truncate_prompt_tokens=limit, truncation_side=side)
                response = requests.post(server.url_for(endpoint), json=payload)
                assert response.status_code == 400, response.status_code
                error = response.json()["error"]
                assert (
                    "JinaForRanking does not support truncate_prompt_tokens"
                    in error["message"]
                )
                assert error["param"] == "truncate_prompt_tokens"

    for get_response in (_get_score_response, _get_rerank_response):
        limited = get_response(
            server,
            "one two three four five",
            ["six seven eight nine ten", "red green blue yellow white"],
            max_tokens_per_query=3,
            max_tokens_per_doc=4,
        )
        expected = get_response(
            server, "one two three", ["six seven eight nine", "red green blue yellow"]
        )
        assert limited.usage.prompt_tokens == expected.usage.prompt_tokens
        if isinstance(limited, ScoreResponse):
            assert len(limited.data) == len(expected.data) == 2
            assert [d.score for d in limited.data] == pytest.approx(
                [d.score for d in expected.data], abs=TOL
            )
        else:
            assert [d.index for d in limited.results] == [
                d.index for d in expected.results
            ]
            assert [d.relevance_score for d in limited.results] == pytest.approx(
                [d.relevance_score for d in expected.results], abs=TOL
            )


def _test_online_nested_markers(server):
    docs = documents[:2]
    expected = _get_embeds(server, docs + [query])
    instruction = "Rank relevant passages."
    for token in ("embed", "rerank"):
        nested = f"<|{token}<|{token}_token|>_token|>"
        actual = _get_embeds(server, [text + nested for text in docs + [query]])
        assert actual.shape == expected.shape == (len(docs) + 1, EMBEDDING_SIZE), (
            actual.shape
        )
        torch.testing.assert_close(actual, expected)
        for get_response in (_get_score_response, _get_rerank_response):
            reference = get_response(server, query, docs, instruction=instruction)
            actual_scores = get_response(
                server,
                query + nested,
                [doc + nested for doc in docs],
                instruction=instruction + nested,
            )
            assert actual_scores.usage.prompt_tokens == reference.usage.prompt_tokens
            if isinstance(actual_scores, ScoreResponse):
                assert len(actual_scores.data) == len(reference.data) == len(docs)
                assert [d.score for d in actual_scores.data] == pytest.approx(
                    [d.score for d in reference.data], abs=TOL
                )
            else:
                assert [d.index for d in actual_scores.results] == [
                    d.index for d in reference.results
                ]
                assert [
                    d.relevance_score for d in actual_scores.results
                ] == pytest.approx(
                    [d.relevance_score for d in reference.results], abs=TOL
                )


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
