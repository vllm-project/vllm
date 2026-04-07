# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "TitanML/tiny-mixtral"
GEN_ENDPOINT = "/inference/v1/generate"

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


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers=headers,
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_generate_routed_experts(client):
    """Test that /inference/v1/generate returns routed_experts when enabled."""
    payload = {
        "model": MODEL_NAME,
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 10, "temperature": 0.0},
        "stream": False,
    }
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]

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
