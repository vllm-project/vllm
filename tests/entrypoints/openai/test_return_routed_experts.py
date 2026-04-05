# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "TitanML/tiny-mixtral"

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers
NUM_LOCAL_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
NUM_HIDDEN_LAYERS = 2


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "32",
        "--enforce-eager",
        "--enable-return-routed-experts",
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

        assert choice["routed_experts"] is not None
        assert choice["token_ids"] is not None

        routed_experts = choice["routed_experts"]
        assert len(routed_experts) > 0
        for token_experts in routed_experts:
            assert len(token_experts) == NUM_HIDDEN_LAYERS
            for layer_experts in token_experts:
                assert len(layer_experts) == NUM_EXPERTS_PER_TOK
                for expert_id in layer_experts:
                    assert 0 <= expert_id < NUM_LOCAL_EXPERTS
