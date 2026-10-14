# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import httpx
import pytest
import requests
import torch
from transformers import AutoTokenizer

from tests.models.utils import check_embeddings_close
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse

MODEL_NAME = "intfloat/multilingual-e5-small"
DTYPE = "bfloat16"
input_text = "The best thing about vLLM is that it supports many different models"
input_tokens = [
    0,
    581,
    2965,
    13580,
    1672,
    81,
    23708,
    594,
    83,
    450,
    442,
    8060,
    7,
    5941,
    12921,
    115774,
    2,
]

LATE_CHUNK_CASES = [(1, False), (5, True), (512, False)]
LATE_CHUNK_TEXTS = [
    "passage: Berlin has museums. They display art.",
    "passage: 中文 😀 e\u0301",
    "   ",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--pooler-config.task",
        "token_embed",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_token_embed(server: RemoteOpenAIServer, model_name: str):
    task = "token_embed"
    request_id = "client-request-id"
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
            "request_id": request_id,
        },
    )

    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id == f"pooling-{request_id}"
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == len(input_tokens)
    assert isinstance(poolings.data[0].data[0], list)
    assert len(poolings.data[0].data[0]) == 384


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("task", ["embed", "classify", "token_classify", "plugin"])
async def test_pooling_not_supported(
    server: RemoteOpenAIServer, model_name: str, task: str
):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": "test",
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"

    if task == "plugin":
        err_msg = "No IOProcessor plugin installed."
    elif task == "embed":
        err_msg = "Try switching the model's pooling_task via"
    else:
        err_msg = f"Unsupported task: {task!r}"
    assert response.json()["error"]["message"].startswith(err_msg)


@pytest.mark.parametrize("chunk_size, normalize", LATE_CHUNK_CASES)
def test_late_chunking_vectors_and_source_ranges(server, chunk_size, normalize):
    texts = LATE_CHUNK_TEXTS
    common = {"model": MODEL_NAME, "input": texts, "task": "token_embed"}
    raw_response = requests.post(
        server.url_for("pooling"), json={**common, "use_activation": False}
    )
    raw_response.raise_for_status()
    raw = raw_response.json()
    response = requests.post(
        server.url_for("pooling"),
        json={
            **common,
            "use_activation": normalize,
            "late_chunking_params": {"chunk_size": chunk_size},
        },
    )
    response.raise_for_status()
    result = response.json()
    PoolingResponse.model_validate(result)
    assert len(result["data"]) == len(raw["data"]) == len(texts)
    assert result["usage"] == raw["usage"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    for index, (text, tokens, output) in enumerate(
        zip(texts, raw["data"], result["data"])
    ):
        assert "late_chunking" not in tokens
        expected = torch.stack(
            [part.mean(0) for part in torch.tensor(tokens["data"]).split(chunk_size)]
        )
        if normalize:
            expected = torch.nn.functional.normalize(expected, dim=-1)
        check_embeddings_close(
            embeddings_0_lst=output["data"],
            embeddings_1_lst=expected.tolist(),
            name_0="online_chunks",
            name_1="contextual_token_reference",
        )
        torch.testing.assert_close(
            torch.tensor(output["data"]).norm(dim=-1),
            expected.norm(dim=-1),
            rtol=1e-2,
            atol=1e-3,
        )
        assert output["index"] == index
        metadata = output["late_chunking"]
        assert metadata["strategy"] == "fixed_tokens"
        encoded = tokenizer(text, return_offsets_mapping=True)
        assert metadata["input_tokens"] == len(encoded["input_ids"])
        assert metadata["chunk_size"] == chunk_size
        assert len(metadata["chunks"]) == len(expected)
        for i, chunk in enumerate(metadata["chunks"]):
            start, end = (
                i * chunk_size,
                min((i + 1) * chunk_size, len(encoded["input_ids"])),
            )
            assert chunk["token_range"] == [start, end]
            offsets = [(a, b) for a, b in encoded["offset_mapping"][start:end] if a < b]
            assert chunk["char_range"] == (
                [min(a for a, _ in offsets), max(b for _, b in offsets)]
                if offsets
                else None
            )


@pytest.mark.parametrize(
    "overrides",
    [
        {"input": [0, 581, 2]},
        {"input": ""},
        {"truncate_prompt_tokens": 4},
        {"padding": "max_length"},
        {"input": "word " * 600},
        {"late_chunking_params": {"chunk_size": 2**64}},
    ],
)
def test_late_chunking_http_rejects_unsupported_requests(server, overrides):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": input_text,
            "late_chunking_params": {"chunk_size": 5},
            **overrides,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_late_chunking_concurrent_requests_do_not_share_ranges(server):
    texts = [
        "passage: A short text.",
        "passage: 一个较长的中文文档，内容不同。",
        input_text,
    ]
    sizes = [1, 5, None]
    async with httpx.AsyncClient(base_url=server.url_root, timeout=60) as client:

        async def send(text, size):
            response = await client.post(
                "/pooling",
                json={
                    "model": MODEL_NAME,
                    "input": text,
                    "request_id": "reused-client-id",
                    "late_chunking_params": {"chunk_size": size} if size else None,
                },
            )
            response.raise_for_status()
            return response.json()["data"][0]

        serial = [await send(text, size) for text, size in zip(texts, sizes)]
        concurrent = await asyncio.gather(
            *(send(text, size) for text, size in zip(texts, sizes))
        )
    for expected, actual in zip(serial, concurrent):
        assert actual.get("late_chunking") == expected.get("late_chunking")
        check_embeddings_close(
            embeddings_0_lst=actual["data"],
            embeddings_1_lst=expected["data"],
            name_0="concurrent",
            name_1="serial",
        )
