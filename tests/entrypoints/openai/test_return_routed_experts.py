# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import numpy as np
import pybase64 as base64
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "TitanML/tiny-mixtral"

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers.
# The published config has sliding_window=4096, which produces
# SlidingWindowSpec kv-cache groups; RoutedExpertsManager requires a
# FullAttentionSpec group, so we override sliding_window=null below.
NUM_LOCAL_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
NUM_HIDDEN_LAYERS = 2


def assert_valid_routed_experts(encoded: str | None) -> None:
    assert encoded is not None
    routed_experts = np.load(io.BytesIO(base64.b64decode(encoded)))
    assert routed_experts.ndim == 3
    num_tokens, num_layers, topk = routed_experts.shape
    assert num_tokens > 0
    assert num_layers == NUM_HIDDEN_LAYERS
    assert topk == NUM_EXPERTS_PER_TOK
    assert (routed_experts >= 0).all()
    assert (routed_experts < NUM_LOCAL_EXPERTS).all()


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "32",
        "--enforce-eager",
        "--enable-return-routed-experts",
        "--hf-overrides",
        '{"sliding_window": null}',
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_routed_experts(server):
    """Test that /v1/completions returns routed_experts when enabled."""
    async with server.get_async_client() as client:
        result = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello, world",
            max_tokens=10,
            temperature=0,
            extra_body={"return_token_ids": True},
        )

        choice = result.model_dump()["choices"][0]

        assert choice["token_ids"] is not None
        assert_valid_routed_experts(choice["routed_experts"])


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat.completions"])
async def test_streaming_routed_experts(server, endpoint):
    """Streaming endpoints return R3 exactly once on the terminal choice."""
    async with server.get_async_client() as client:
        if endpoint == "completions":
            stream = await client.completions.create(
                model=MODEL_NAME,
                prompt="Hello, world",
                max_tokens=10,
                temperature=0,
                stream=True,
            )
        else:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello, world"}],
                max_tokens=10,
                temperature=0,
                stream=True,
            )

        terminal_choices = []
        routed_experts = []
        async for chunk in stream:
            for choice in chunk.model_dump()["choices"]:
                if choice.get("finish_reason") is not None:
                    terminal_choices.append(choice)
                if choice.get("routed_experts") is not None:
                    routed_experts.append(choice["routed_experts"])

        assert len(terminal_choices) == 1
        assert len(routed_experts) == 1
        assert_valid_routed_experts(routed_experts[0])
