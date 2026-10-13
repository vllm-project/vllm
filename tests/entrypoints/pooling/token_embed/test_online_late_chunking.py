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
from vllm import PoolingParams
from vllm.config import PoolerConfig
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.pooling_params import LateChunkingParams

MODEL_NAME = "intfloat/multilingual-e5-small"
DTYPE = "bfloat16"
LATE_CHUNK_CASES = [(1, False), (5, True), (512, False)]
LATE_CHUNK_TEXTS = [
    "passage: Berlin has museums. They display art.",
    "passage: 中文 😀 e\u0301",
    "   ",
]
INPUT_TEXT = "The best thing about vLLM is that it supports many different models"


@pytest.fixture(scope="module")
def offline_late_chunks(vllm_runner):
    results = {}
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        pooler_config=PoolerConfig(task="token_embed"),
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    ) as runner:
        for size, normalize in LATE_CHUNK_CASES:
            outputs = runner.llm.encode(
                LATE_CHUNK_TEXTS,
                pooling_task="token_embed",
                use_tqdm=False,
                pooling_params=PoolingParams(
                    use_activation=normalize,
                    late_chunking_params=LateChunkingParams(size),
                ),
            )
            results[size, normalize] = [
                (output.outputs.data.tolist(), output.late_chunking)
                for output in outputs
            ]
    return results


@pytest.fixture(scope="module")
def server(offline_late_chunks):
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
        "--gpu-memory-utilization",
        "0.5",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("chunk_size, normalize", LATE_CHUNK_CASES)
def test_late_chunking_vectors_and_source_ranges(
    server, offline_late_chunks, chunk_size, normalize
):
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
    parsed = PoolingResponse.model_validate(result)
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
        offline_vectors, offline_metadata = offline_late_chunks[chunk_size, normalize][
            index
        ]
        assert parsed.data[index].late_chunking == offline_metadata
        check_embeddings_close(
            embeddings_0_lst=output["data"],
            embeddings_1_lst=offline_vectors,
            name_0="online",
            name_1="offline",
        )
        metadata = output["late_chunking"]
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
    ],
)
def test_late_chunking_http_rejects_unsupported_requests(server, overrides):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": INPUT_TEXT,
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
        INPUT_TEXT,
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
