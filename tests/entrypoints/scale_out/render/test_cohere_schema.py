# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cohere request-schema validation through the GPU-less render server."""

from typing import Any

import httpx
import pytest
from transformers import Cohere2Config

from tests.utils import RemoteLaunchRenderServer
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "vllm/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module", params=["cohere_command3", "cohere_command4"])
def cohere_schema_server(request, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("cohere-schema-model")
    Cohere2Config(
        architectures=["Cohere2ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    ).save_pretrained(model_dir)
    tokenizer = get_tokenizer(MODEL_NAME)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    tokenizer.save_pretrained(model_dir)
    with RemoteLaunchRenderServer(
        str(model_dir),
        [
            "--served-model-name",
            "cohere-schema-test",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            request.param,
            "--max-model-len",
            "2048",
        ],
    ) as server:
        yield server


@pytest.mark.asyncio
async def test_cohere_malformed_tool_schema_returns_400(cohere_schema_server):
    """Request preparation rejects malformed schemas before any inference."""
    payload: dict[str, Any] = {
        "model": "cohere-schema-test",
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 1,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    }
    async with httpx.AsyncClient(
        base_url=cohere_schema_server.url_for(""), timeout=60.0
    ) as client:
        good = await client.post("/v1/chat/completions/render", json=payload)
        assert good.status_code == 200, good.text
        payload["tools"][0]["function"]["parameters"]["properties"]["city"] = 1
        bad = await client.post("/v1/chat/completions/render", json=payload)
        assert bad.status_code == 400, bad.text
        assert bad.json()["error"]["param"] == "tools"
        assert "schema" in bad.json()["error"]["message"].lower()
        payload["tools"][0]["function"]["parameters"]["properties"]["city"] = {
            "type": "string"
        }
        good_after = await client.post("/v1/chat/completions/render", json=payload)
        assert good_after.status_code == 200, good_after.text
