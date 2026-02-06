# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tempfile import TemporaryDirectory

import httpx
import pytest

from vllm.version import __version__ as VLLM_VERSION

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    with TemporaryDirectory() as tmpdir:
        args = [
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--enforce-eager",
            "--max-num-seqs",
            "128",
            "--uds",
            f"{tmpdir}/vllm.sock",
        ]

        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.asyncio
async def test_show_version(server: RemoteOpenAIServer):
    transport = httpx.HTTPTransport(uds=server.uds)
    client = httpx.Client(transport=transport)
    response = client.get(server.url_for("version"))
    response.raise_for_status()

    assert response.json() == {"version": VLLM_VERSION}
