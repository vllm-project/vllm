# SPDX-License-Identifier: Apache-2.0

import openai
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.asyncio
async def test_shutdown_on_engine_failure():
    # dtype, max-len etc set so that this can run in CI
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        async with remote_server.get_async_client() as client:

            with pytest.raises(
                (openai.APIConnectionError, openai.InternalServerError)):
                # Asking for lots of prompt logprobs will currently crash the
                # engine. This may change in the future when that bug is fixed
                prompt = "Hello " * 4000
                await client.completions.create(
                    model=MODEL_NAME,
                    prompt=prompt,
                    extra_body={"prompt_logprobs": 10})

            # Now the server should shut down
            return_code = remote_server.proc.wait(timeout=8)
            assert return_code is not None
