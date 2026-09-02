# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import httpx
import numpy as np
import pybase64 as base64
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "TitanML/tiny-mixtral"
GEN_ENDPOINT = "/inference/v1/generate"

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers.
# The published config has sliding_window=4096, which produces
# SlidingWindowSpec kv-cache groups; RoutedExpertsManager requires a
# FullAttentionSpec group, so we override sliding_window=null below.
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
        "--hf-overrides",
        '{"sliding_window": null}',
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

    # routed_experts is base64-encoded .npy bytes; decode to ndarray.
    routed_experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
    assert routed_experts.ndim == 3
    num_tokens, num_layers, topk = routed_experts.shape
    assert num_tokens > 0
    assert num_layers == NUM_HIDDEN_LAYERS
    assert topk == NUM_EXPERTS_PER_TOK
    assert (routed_experts >= 0).all()
    assert (routed_experts < NUM_LOCAL_EXPERTS).all()
