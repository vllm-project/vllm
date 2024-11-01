# imports for guided decoding tests
import re

import openai
import pytest

from ...utils import RemoteOpenAIServer


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match=re.compile('.+Prompt cannot be empty.+')):
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
                           match=re.compile('.*out of vocabulary.*')):
            await client.completions.create(model=model_name,
                                            prompt=[999999],
                                            max_tokens=5,
                                            temperature=0.0)


@pytest.mark.asyncio
async def test_reject_multistep_with_guided_decoding():
    model_name = "gpt2"
    server_args = ["--enforce-eager", "--num-scheduler-steps", "8"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match=re.compile(
                               '.*Guided decoding .* multi-step decoding.*')):
            await client.completions.create(
                model=model_name,
                prompt="Hello",
                max_tokens=5,
                temperature=0.0,
                extra_body={"response_format": {
                    "type": "json_object"
                }})
