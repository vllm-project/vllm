# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.v1.sample.logits_processors.utils import (DUMMY_LOGITPROC_ARG,
                                                     DUMMY_LOGITPROC_FQCN,
                                                     MAX_TOKENS, MODEL_NAME,
                                                     TEMP_GREEDY, prompts)


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager"
    ]


@pytest.fixture(scope="module",
                params=[["--logits-processors", DUMMY_LOGITPROC_FQCN]])
def server(default_server_args, request):
    """Server cli arg list is parameterized by logitproc source
    
    TODO (andy): entrypoints unit test; currently CLI logitsprocs
    unit test only covers the case where logitproc is specified by
    FQCN
    """
    if request.param:
        default_server_args = default_server_args + request.param
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


api_kwargs = {
    "temperature": TEMP_GREEDY,
    "max_tokens": MAX_TOKENS,
    "logprobs": 0,
}

extra_body_kwargs = {"vllm_xargs": {DUMMY_LOGITPROC_ARG: 128}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_custom_logitsprocs_cli(client: openai.AsyncOpenAI,
                                      model_name: str):
    """Test CLI interface for passing custom logitsprocs
    
    Launch vLLM OpenAI-compatible server with CLI argument to loads a custom
    logitproc that has a well-defined behavior (mask out all tokens except one
    `target_token`). Logitproc is specified by fully-qualified class name (FQCN)

    Pass in requests, 50% of which pass a `target_token` value
    in through `extra_body["vllm_xargs"]`, 50% of which do not.

    Validate that requests which activate the custom logitproc, only output
    `target_token`
    """
    use_dummy_logitproc = True
    for prompt in prompts:
        # Send vLLM API request; for some requests, activate dummy logitproc
        kwargs = {
            **api_kwargs,
        }
        if use_dummy_logitproc:
            target_token = random.choice([128, 67])
            # For requests which activate the dummy logitproc, choose one of
            # two `target_token` values which are known not to be EOS tokens
            kwargs["extra_body"] = {
                "vllm_xargs": {
                    DUMMY_LOGITPROC_ARG: target_token
                }
            }
        batch = await client.completions.create(
            model=model_name,
            prompt=prompt,
            **kwargs,
        )

        if use_dummy_logitproc:
            # Only for requests which activate dummy logitproc - validate that
            # only `target_token` is generated
            choices: openai.types.CompletionChoice = batch.choices
            toks = choices[0].logprobs.tokens
            if not all([x == toks[0] for x in toks]):
                raise AssertionError(
                    f"Generated {toks} should all be {toks[0]}")

        # Alternate whether to activate dummy logitproc for each request
        use_dummy_logitproc = not use_dummy_logitproc
