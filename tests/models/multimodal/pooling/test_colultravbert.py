# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score

MODEL_NAME = "athrael-soju/ColUltraVBERT"
DTYPE = "bfloat16"
PAGE_PROMPT = "User:<image>Describe the image.<end_of_utterance>\nAssistant:"
IMAGE_SEQ_LEN = 64


@pytest.fixture(scope="module")
def colultravbert_model(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        enforce_eager=True,
    ) as vllm_model:
        yield vllm_model


@pytest.fixture(scope="module")
def embedding_dim(colultravbert_model):
    return colultravbert_model.llm.model_config.hf_config.embedding_dim


def test_colultravbert_text_token_embed(colultravbert_model, embedding_dim):
    outputs = colultravbert_model.token_embed(["What is machine learning?"])

    assert len(outputs) == 1
    emb = torch.tensor(outputs[0])
    assert emb.dim() == 2
    assert emb.shape[1] == embedding_dim
    assert emb.shape[0] > 1


def test_colultravbert_text_relevance_ordering(colultravbert_model):
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather in Paris is mild in spring.",
    ]

    scores = colultravbert_model.score(query, documents)

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_colultravbert_text_late_interaction(colultravbert_model):
    query = "What is the capital of France?"
    doc = "The capital of France is Paris."

    q_out = colultravbert_model.token_embed([query])
    d_out = colultravbert_model.token_embed([doc])

    q_emb = torch.tensor(q_out[0])
    d_emb = torch.tensor(d_out[0])
    manual_score = compute_maxsim_score(q_emb, d_emb).item()

    vllm_scores = colultravbert_model.score(query, doc)

    assert len(vllm_scores) == 1
    assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


def test_colultravbert_image_token_embed(
    colultravbert_model, embedding_dim, image_assets
):
    image = image_assets[0].pil_image
    inputs = colultravbert_model.get_inputs(
        [PAGE_PROMPT],
        images=[image],
    )
    req_outputs = colultravbert_model.llm.encode(
        inputs,
        pooling_task="token_embed",
    )
    outputs = [req_output.outputs.data for req_output in req_outputs]

    assert len(outputs) == 1
    emb = torch.tensor(outputs[0])
    assert emb.dim() == 2
    assert emb.shape[1] == embedding_dim
    assert emb.shape[0] >= 2 * IMAGE_SEQ_LEN
