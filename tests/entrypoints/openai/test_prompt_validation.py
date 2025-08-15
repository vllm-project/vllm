# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for guided decoding tests
import openai
import pytest
import regex as re

from ...utils import RemoteOpenAIServer


@pytest.fixture(scope="function", autouse=True)
def use_v1_only(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V1', '1')


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match="decoder prompt cannot be empty"):
            await client.completions.create(model=model_name,
                                            prompt="",
                                            max_tokens=5,
                                            temperature=0.0)


@pytest.mark.asyncio
async def test_out_of_vocab_token_ids():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match=re.compile('.*out of vocabulary.*').pattern):
            await client.completions.create(model=model_name,
                                            prompt=[999999],
                                            max_tokens=5,
                                            temperature=0.0)
