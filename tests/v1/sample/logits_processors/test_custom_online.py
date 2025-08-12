# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from typing import Any

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.v1.sample.logits_processors.utils import (DUMMY_LOGITPROC_ARG,
                                                     MAX_TOKENS, MODEL_NAME,
                                                     TEMP_GREEDY, prompts)
from vllm.test_utils import DUMMY_LOGITPROC_FQCN

# Inject this code into python interpreter -c flag to launch vllm server with
# patched entrypoint
CMD_STR = ("import importlib.metadata; from vllm.test_utils import "
           "entry_points as fake_entry_points; importlib.metadata."
           "entry_points = fake_entry_points; from vllm.entrypoints.cli "
           "import main; main.main()")


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


@pytest.fixture(scope="function",
                params=[[], ["--logits-processors", DUMMY_LOGITPROC_FQCN]])
def server(default_server_args, request, monkeypatch):
    """Consider two server configurations:
    (1) --logits-processors cli arg specifies dummy logits processor via fully-
    qualified class name (FQCN)
    (2) No --logits-processors cli arg; monkeypatch dummy logits processor
    entrypoint
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    if request.param:
        # Append FQCN argument
        default_server_args = default_server_args + request.param
        with RemoteOpenAIServer(MODEL_NAME,
                                default_server_args) as remote_server:
            yield remote_server
    else:
        # Patch in dummy logit processor entrypoint
        with RemoteOpenAIServer(
                MODEL_NAME,
                default_server_args,
                multiproc_method="spawn",
                cmd_str=CMD_STR,
        ) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


# General request argument values for these tests
api_keyword_args = {
    # Greedy sampling ensures that requests which receive the `target_token`
    # arg will decode it in every step
    "temperature": TEMP_GREEDY,
    # Since EOS will never be decoded (unless `target_token` is EOS)
    "max_tokens": MAX_TOKENS,
    # Return decoded token logprobs (as a way of getting token id)
    "logprobs": 0,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_custom_logitsprocs(client: openai.AsyncOpenAI, model_name: str):
    """Test custom logitsprocs when starting OpenAI server from CLI
    
    Launch vLLM OpenAI-compatible server, configured to load a custom logitproc
    that has a well-defined behavior (mask out all tokens except one
    `target_token`).

    Pass in requests, 50% of which pass a `target_token` value
    in through `extra_body["vllm_xargs"]`, 50% of which do not.

    Validate that requests which activate the custom logitproc, only output
    `target_token`
    """

    use_dummy_logitproc = True
    for prompt in prompts:
        # Build request arguments
        request_keyword_args: dict[str, Any] = {
            **api_keyword_args,
        }
        if use_dummy_logitproc:
            # 50% of requests pass target_token custom arg
            target_token = random.choice([128, 67])
            # For requests which activate the dummy logitproc, choose one of
            # two `target_token` values which are known not to be EOS tokens
            request_keyword_args["extra_body"] = {
                "vllm_xargs": {
                    DUMMY_LOGITPROC_ARG: target_token
                }
            }
        batch = await client.completions.create(
            model=model_name,
            prompt=prompt,
            **request_keyword_args,
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
