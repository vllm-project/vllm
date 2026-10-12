# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import openai

from tests.conftest import HfRunner
from tests.models.utils import check_embeddings_close, matryoshka_fy
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse


def float_embeddings(response: EmbeddingResponse) -> list[list[float]]:
    embeddings = []
    for d in response.data:
        assert isinstance(d.embedding, list)
        embeddings.append(d.embedding)
    return embeddings


def run_embedding_correctness_test(
    hf_model: "HfRunner",
    inputs: list[str],
    vllm_outputs: Sequence[list[float]],
    dimensions: int | None = None,
):
    hf_outputs = hf_model.encode(inputs)
    if dimensions:
        hf_outputs = matryoshka_fy(hf_outputs, dimensions)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


async def run_client_embeddings(
    client: openai.AsyncOpenAI,
    model_name: str,
    queries: list[str],
    instruction: str = "",
) -> list[list[float]]:
    outputs = await client.embeddings.create(
        model=model_name,
        input=[instruction + q for q in queries],
    )
    return [data.embedding for data in outputs.data]
