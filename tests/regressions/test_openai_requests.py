# SPDX-License-Identifier: Apache-2.0

# imports for guided decoding tests
from itertools import chain

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
# downloading lora to test lora requests
from openai.types import Completion

from ..utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture()
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_multiseq_logprobs_streaming(client: openai.AsyncOpenAI):
    """Edge case request combining multiple functionalities

    https://github.com/vllm-project/vllm/pull/15259
    https://github.com/vllm-project/vllm/pull/16805
    """

    # completions
    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt="1 2 3 4 5",
        max_tokens=3,
        # include usage chunk to make sure the stream is complete
        stream_options={"include_usage": True},
        stream=True,
        n=2,
        logprobs=0,  # include 1-top logprob per generated token
        temperature=1.0)

    n0_chunks: list[Completion] = []
    n1_chunks: list[Completion] = []
    usage_chunk: Completion = None
    async for chunk in stream:
        print(chunk)
        if choices := chunk.choices:
            assert len(choices) == 1, \
                (f"Streamed chunk had {len(choices)} choices, when only 1 was"
                 " expected")
            choice = choices[0]
            if choice.index == 0:
                n0_chunks.append(chunk)
            elif choice.index == 1:
                n1_chunks.append(chunk)
            else:
                raise AssertionError(f"Unexpected choice index {choice.index}")

        elif chunk.usage is not None:
            usage_chunk = chunk

        else:
            raise AssertionError(f"Unexpected chunk {chunk}")

    # check that we got the requested number of tokens
    assert sum(
        len(chunk.choices[0].logprobs.tokens) for chunk in n0_chunks
        if chunk.choices[0].logprobs
    ) == 3, "Streamed response did not have the expected number of tokens."
    assert sum(
        len(chunk.choices[0].logprobs.tokens) for chunk in n1_chunks
        if chunk.choices[0].logprobs
    ) == 3, "Streamed response did not have the expected number of tokens."

    # check 1 logprob per token/chunk
    for chunk in chain(n0_chunks, n1_chunks):
        # a finish chunk may not have any text/logprobs
        # V0 does not
        # V1 does
        choice = chunk.choices[0]
        if choice.logprobs is None:
            assert choice.finish_reason
            assert choice.text == ''
            continue

        assert choice.logprobs.top_logprobs
        for top_logprobs in choice.logprobs.top_logprobs:
            assert len(top_logprobs) == 1

    # requested usage
    assert usage_chunk is not None
    assert usage_chunk.usage.completion_tokens == 6
    assert usage_chunk.usage.prompt_tokens == 9
    assert usage_chunk.usage.total_tokens == 15
