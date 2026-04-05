# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

import pytest
import pytest_asyncio
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.utils.torch_utils import cuda_device_count_stateless

MODEL = "microsoft/Phi-mini-MoE-instruct"

pytestmark = [
    pytest.mark.skipif(
        cuda_device_count_stateless() < 2,
        reason="Need at least 2 GPUs to run expert parallel tests.",
    ),
    pytest.mark.distributed(num_gpus=2),
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--max-num-seqs",
        "16",
        "--load-format",
        "dummy",
        "--data-parallel-size",
        "2",
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
    ]
    env = {
        "VLLM_COLLECT_EXPERT_USAGE_HISTOGRAM": "1",
        "VLLM_EXPERT_USAGE_HISTOGRAM_SAVE_INTERVAL": "1",
    }
    with RemoteOpenAIServer(MODEL, args, env_dict=env) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as cl:
        yield cl


@pytest.mark.asyncio
async def test_moe_expert_selection_counter(server, client):
    num_requests = 3
    for _ in range(num_requests):
        await client.completions.create(
            model=MODEL,
            prompt="The capital of France is",
            max_tokens=10,
        )

    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    samples = []
    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:moe_expert_selection_counter":
            samples = list(family.samples)
            break

    assert len(samples) > 0, "vllm:moe_expert_selection_counter not found in /metrics"

    for sample in samples:
        assert "layer" in sample.labels, f"Missing 'layer' label: {sample}"
        assert "expert" in sample.labels, f"Missing 'expert' label: {sample}"

    total = sum(s.value for s in samples)
    assert total > 0, f"Expected non-zero total expert selections, but got {total}"

    layers = {s.labels["layer"] for s in samples}
    assert len(layers) == 32, f"Expected 32 MoE layers, but only found layers: {layers}"


@pytest.mark.asyncio
async def test_moe_per_rank_expert_selection_counter(server, client):
    num_requests = 3
    for _ in range(num_requests):
        await client.completions.create(
            model=MODEL,
            prompt="The capital of France is",
            max_tokens=10,
        )

    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    samples = []
    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:moe_per_rank_expert_selection_counter":
            samples = list(family.samples)
            break

    assert len(samples) > 0, (
        "vllm:moe_per_rank_expert_selection_counter not found in /metrics"
    )

    for sample in samples:
        assert "layer" in sample.labels, f"Missing 'layer' label: {sample}"
        assert "rank" in sample.labels, f"Missing 'rank' label: {sample}"

    total = sum(s.value for s in samples)
    assert total > 0, (
        f"Expected non-zero total per-rank expert selections, but got {total}"
    )

    layers = {s.labels["layer"] for s in samples}
    assert len(layers) == 32, f"Expected 32 MoE layers, but only found layers: {layers}"

    ranks = {s.labels["rank"] for s in samples}
    assert len(ranks) == 2, (
        f"Expected 2 ranks (data_parallel_size=2), but found: {ranks}"
    )
