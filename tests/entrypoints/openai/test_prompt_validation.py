# imports for guided decoding tests
import re

import openai
import pytest

from ...utils import RemoteOpenAIServer


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--disable-frontend-multiprocessing", "--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(openai.BadRequestError,
                           match=re.compile('.+Empty prompt.+')):
            await client.completions.create(model=model_name,
                                            prompt="",
                                            max_tokens=5,
                                            temperature=0.0)