# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import httpx
import openai
import pytest
import pytest_asyncio
import torch

from ....utils import RemoteOpenAIServer
from .embed_utils import run_client_embeddings

MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_LEN = 512


# Example from https://huggingface.co/BAAI/bge-m3
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, "
    "lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set "
    "of documents based on the query terms appearing in each document",
]

similarity_reference = [[0.6265, 0.3477], [0.3499, 0.678]]
lexical_score_reference = [0.19554901123046875, 0.0]
colbert_score_reference = [0.7797, 0.4620]


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--hf-overrides",
        '{"architectures": ["BgeM3EmbeddingModel"]}',
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_bge_m3_api_server_embedding(client: openai.AsyncOpenAI):
    embeddings_list_1 = await run_client_embeddings(
        client,
        MODEL_NAME,
        sentences_1,
    )
    embeddings_list_2 = await run_client_embeddings(
        client,
        MODEL_NAME,
        sentences_2,
    )

    embeddings_1 = torch.tensor(embeddings_list_1)
    embeddings_2 = torch.tensor(embeddings_list_2)
    similarity = embeddings_1 @ embeddings_2.T

    # reference values from BAAI/bge-m3 documentation
    reference = torch.tensor(similarity_reference)

    assert torch.allclose(similarity, reference, rtol=0.01)


async def tokenize(client: openai.AsyncOpenAI, sentences: list[str]) -> list[list[int]]:
    futures = []
    for sentence in sentences:
        futures.append(
            client.post(
                "../tokenize",
                body={"model": MODEL_NAME, "prompt": sentence},
                cast_to=httpx.Response,
            )
        )
    return [(await future).json()["tokens"] for future in futures]


async def sparse_embeddings(
    client: openai.AsyncOpenAI, sentences: list[str]
) -> list[dict[int, float]]:
    all_tokens = await tokenize(client, sentences)
    result = await client.post(
        "../pooling",
        body={"model": MODEL_NAME, "input": sentences, "task": "token_classify"},
        cast_to=httpx.Response,
    )
    all_embeddings = [data["data"] for data in result.json()["data"]]

    ret = []

    for sent_tokens, sent_emb in zip(all_tokens, all_embeddings):
        token_embs = dict[int, float]()
        if sent_tokens[0] == 0:
            sent_tokens = sent_tokens[1:]
        for token, val in zip(sent_tokens, sent_emb):
            token_embs[token] = max(val, token_embs.get(token, 0.0))
        ret.append(token_embs)
    return ret


# Based on https://github.com/FlagOpen/FlagEmbedding/blob/6fd176266f2382878bcc69cd656cff425d52f49b/FlagEmbedding/inference/embedder/encoder_only/m3.py#L129
def compute_lexical_matching_score(
    lw1: dict[int, float], lw2: dict[int, float]
) -> float:
    scores = 0.0
    for token, weight in lw1.items():
        if token in lw2:
            scores += weight * lw2[token]
    return scores


@pytest.mark.asyncio
async def test_bge_m3_api_server_sparse_embedding(client: openai.AsyncOpenAI):
    embeddings_1 = await sparse_embeddings(client, sentences_1)
    embeddings_2 = await sparse_embeddings(client, sentences_2)

    lexical_scores_1_0_x_2_0 = compute_lexical_matching_score(
        embeddings_1[0], embeddings_2[0]
    )
    assert lexical_scores_1_0_x_2_0 == pytest.approx(
        lexical_score_reference[0], rel=0.01
    )

    lexical_scores_1_0_x_1_1 = compute_lexical_matching_score(
        embeddings_1[0], embeddings_1[1]
    )
    assert lexical_scores_1_0_x_1_1 == pytest.approx(
        lexical_score_reference[1], rel=0.01
    )


@pytest.mark.asyncio
async def test_bge_m3_api_server_sparse_embedding_corner_case(
    client: openai.AsyncOpenAI,
):
    embeddings = await sparse_embeddings(client, ["Hi"])
    assert len(embeddings) == 1
    assert 2673 in embeddings[0]
    assert embeddings[0][2673] == pytest.approx(0.26710861921310425, rel=0.01)


# ── embed_sparse task: server-side aggregation ──
# Returns vocab-sized sparse vectors directly usable by vector databases,
# without requiring a separate /tokenize call + client-side aggregation.


async def aggregated_sparse_embeddings(
    client: openai.AsyncOpenAI, sentences: list[str]
) -> list[dict[int, float]]:
    """Call embed_sparse task which returns vocab-sized sparse vectors."""
    result = await client.post(
        "../pooling",
        body={"model": MODEL_NAME, "input": sentences, "task": "embed_sparse"},
        cast_to=httpx.Response,
    )
    all_embeddings = [data["data"] for data in result.json()["data"]]

    ret = []
    for emb in all_embeddings:
        # Convert dense vocab-sized vector to sparse dict {token_id: weight}
        sparse_dict = {i: v for i, v in enumerate(emb) if v > 0.0}
        ret.append(sparse_dict)
    return ret


@pytest.mark.asyncio
async def test_bge_m3_embed_sparse_matches_token_classify(
    client: openai.AsyncOpenAI,
):
    """Verify embed_sparse produces identical results to token_classify +
    client-side aggregation."""
    # Get sparse embeddings via both methods
    tc_embeddings = await sparse_embeddings(client, sentences_1)
    agg_embeddings = await aggregated_sparse_embeddings(client, sentences_1)

    for tc_dict, agg_dict in zip(tc_embeddings, agg_embeddings):
        # Same token IDs should be present
        assert set(tc_dict.keys()) == set(agg_dict.keys()), (
            f"Token ID mismatch: "
            f"token_classify={sorted(tc_dict.keys())}, "
            f"embed_sparse={sorted(agg_dict.keys())}"
        )
        # Same weights
        for token_id in tc_dict:
            assert tc_dict[token_id] == pytest.approx(agg_dict[token_id], rel=0.01), (
                f"Weight mismatch for token {token_id}"
            )


@pytest.mark.asyncio
async def test_bge_m3_embed_sparse_lexical_scores(
    client: openai.AsyncOpenAI,
):
    """Verify embed_sparse produces correct lexical matching scores."""
    embeddings_1 = await aggregated_sparse_embeddings(client, sentences_1)
    embeddings_2 = await aggregated_sparse_embeddings(client, sentences_2)

    lexical_scores_1_0_x_2_0 = compute_lexical_matching_score(
        embeddings_1[0], embeddings_2[0]
    )
    assert lexical_scores_1_0_x_2_0 == pytest.approx(
        lexical_score_reference[0], rel=0.01
    )

    lexical_scores_1_0_x_1_1 = compute_lexical_matching_score(
        embeddings_1[0], embeddings_1[1]
    )
    assert lexical_scores_1_0_x_1_1 == pytest.approx(
        lexical_score_reference[1], rel=0.01
    )


@pytest.mark.asyncio
async def test_bge_m3_embed_sparse_corner_case(
    client: openai.AsyncOpenAI,
):
    """Verify embed_sparse works for short inputs."""
    embeddings = await aggregated_sparse_embeddings(client, ["Hi"])
    assert len(embeddings) == 1
    assert 2673 in embeddings[0]
    assert embeddings[0][2673] == pytest.approx(0.26710861921310425, rel=0.01)


# https://github.com/FlagOpen/FlagEmbedding/blob/6fd176266f2382878bcc69cd656cff425d52f49b/FlagEmbedding/inference/embedder/encoder_only/m3.py#L163
def colbert_score(q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
    token_scores = torch.einsum("in,jn->ij", q_reps, p_reps)
    scores, _ = token_scores.max(-1)
    scores = torch.sum(scores) / q_reps.size(0)
    return scores


@pytest.mark.asyncio
async def test_bge_m3_api_server_multi_vector(client: openai.AsyncOpenAI):
    result_1 = await client.post(
        "../pooling",
        body={"model": MODEL_NAME, "input": sentences_1, "task": "token_embed"},
        cast_to=httpx.Response,
    )
    embeddings_1 = [torch.tensor(data["data"]) for data in result_1.json()["data"]]

    result_2 = await client.post(
        "../pooling",
        body={"model": MODEL_NAME, "input": sentences_2, "task": "token_embed"},
        cast_to=httpx.Response,
    )
    embeddings_2 = [torch.tensor(data["data"]) for data in result_2.json()["data"]]

    colbert_score_1_0_x_2_0 = colbert_score(embeddings_1[0], embeddings_2[0])
    assert colbert_score_1_0_x_2_0 == pytest.approx(
        colbert_score_reference[0], rel=0.01
    )
    colbert_score_1_0_x_2_1 = colbert_score(embeddings_1[0], embeddings_2[1])
    assert colbert_score_1_0_x_2_1 == pytest.approx(
        colbert_score_reference[1], rel=0.01
    )
