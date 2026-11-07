# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for the `hf` parsers reading a checkpoint `response_template`."""

import json
from pathlib import Path

import openai
import pytest
import pytest_asyncio

from tests.parser.test_response_template import QWEN3_6_RESPONSE_TEMPLATE
from tests.utils import RemoteOpenAIServer
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    """Qwen3.5-0.8B with the Qwen3.6 `response_template`; both share one
    output format."""
    src = Path(download_weights_from_hf(MODEL_NAME, None, allow_patterns=["*"]))
    path = tmp_path_factory.mktemp("response_template_checkpoint")
    for file in src.iterdir():
        if file.name != "tokenizer_config.json":
            (path / file.name).symlink_to(file)
    config = json.loads((src / "tokenizer_config.json").read_text())
    config["response_template"] = QWEN3_6_RESPONSE_TEMPLATE
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    return path


@pytest.fixture(scope="module")
def server(model_dir):
    args = [
        "--served-model-name",
        MODEL_NAME,
        "--reasoning-parser",
        "hf",
        "--tool-call-parser",
        "hf",
        "--enable-auto-tool-choice",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
    ]
    with RemoteOpenAIServer(str(model_dir), args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enable_thinking", [True, False])
async def test_reasoning_content_template_routes_reasoning_and_content(
    client: openai.AsyncOpenAI, stream: bool, enable_thinking: bool
):
    """A template naming the reasoning field `reasoning_content`, as chat
    templates read it, and joining repeated content is served. The prompt
    opens the reasoning region, or closes it when thinking is disabled."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=64,
        temperature=0,
        stream=stream,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    if stream:
        reasoning = content = ""
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning += getattr(delta, "reasoning", None) or ""
                content += delta.content or ""
    else:
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning", None)
        content = message.content

    if enable_thinking:
        assert reasoning
        assert "think>" not in (content or "")
    else:
        assert content
        assert not reasoning
