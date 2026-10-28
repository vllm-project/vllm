# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Idefics3ImageProcessor

from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score
from vllm.model_executor.models.colultravbert import get_capped_longest_edge

MODEL_NAME = "athrael-soju/ColUltraVBERT"
# Later revisions declare the UltraVBertModel architecture instead.
REVISION = "509adbab61b71da0484b20f3baf718e15219d761"
DTYPE = "bfloat16"
PAGE_PROMPT = "User:<image>Describe the image.<end_of_utterance>\nAssistant:"
IMAGE_SEQ_LEN = 64
EMBEDDING_DIM = 768
LONGEST_EDGE = 3072
TILE_SIZE = 512
MAX_IMAGE_TILES = 27


def page_grid(width: int, height: int) -> tuple[int, int]:
    edge = get_capped_longest_edge(
        height, width, LONGEST_EDGE, TILE_SIZE, MAX_IMAGE_TILES
    )
    processor = Idefics3ImageProcessor(
        size={"longest_edge": LONGEST_EDGE},
        max_image_size={"longest_edge": TILE_SIZE},
    )
    _, rows, cols = processor.get_number_of_image_patches(
        height, width, {"size": {"longest_edge": edge}}
    )
    return rows, cols


# Grids from the checkpoint's own UltraVBertImageProcessor.
@pytest.mark.parametrize(
    ("width", "height", "grid"),
    [
        (1700, 2200, (5, 4)),
        (2200, 1700, (4, 5)),
        (2000, 2000, (5, 5)),
        (4000, 1000, (2, 6)),
        (1000, 4000, (6, 2)),
        (300, 200, (4, 6)),
        (3000, 2400, (4, 5)),
    ],
)
def test_colultravbert_tile_cap(width, height, grid):
    assert page_grid(width, height) == grid


@pytest.fixture(scope="module")
def colultravbert_model(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        revision=REVISION,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=2048,
        enforce_eager=True,
    ) as vllm_model:
        yield vllm_model


def test_colultravbert_text_token_embed(colultravbert_model):
    outputs = colultravbert_model.token_embed(["What is machine learning?"])

    assert len(outputs) == 1
    emb = torch.tensor(outputs[0])
    assert emb.dim() == 2
    assert emb.shape[1] == EMBEDDING_DIM
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


def test_colultravbert_image_token_embed(colultravbert_model, image_assets):
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
    assert emb.shape[1] == EMBEDDING_DIM

    rows, cols = page_grid(*image.size)
    tokenizer = colultravbert_model.llm.get_tokenizer()
    prompt_ids = req_outputs[0].prompt_token_ids
    image_tokens = prompt_ids.count(tokenizer.convert_tokens_to_ids("<image>"))
    assert image_tokens == (rows * cols + 1) * IMAGE_SEQ_LEN
    assert emb.shape[0] == len(prompt_ids)
