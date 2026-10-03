# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from prometheus_client import CollectorRegistry, Counter, generate_latest

from vllm.benchmarks.serve import (
    fetch_diffusion_metrics,
    fetch_spec_decode_metrics,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["plain-model", "model with spaces"])
@pytest.mark.parametrize("kind", ["spec_decode", "diffusion"])
@pytest.mark.parametrize("timestamps", [False, True])
async def test_decode_metrics_preserve_counters_with_model_labels(
    model_name, kind, timestamps
):
    registry = CollectorRegistry()
    if kind == "spec_decode":
        counter_values = {
            "num_drafts": 3,
            "num_draft_tokens": 9,
            "num_accepted_tokens": 6,
        }
        fetch = fetch_spec_decode_metrics
    else:
        counter_values = {
            "num_denoising_steps": 2,
            "num_canvas_positions": 8,
            "num_committed_tokens": 6,
        }
        fetch = fetch_diffusion_metrics
    for name, value in counter_values.items():
        Counter(f"vllm:{kind}_{name}", name, ["model_name"], registry=registry).labels(
            model_name
        ).inc(value)
    expected: dict[str, object] = dict(counter_values)
    if kind == "spec_decode":
        counter = Counter(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "accepted",
            ["model_name", "position"],
            registry=registry,
        )
        counter.labels(model_name, "0").inc(3)
        expected["accepted_per_pos"] = {0: 3}

    session = MagicMock(spec=aiohttp.ClientSession)
    response = session.get.return_value.__aenter__.return_value
    response.status = 200
    text = generate_latest(registry).decode()
    if timestamps:
        text = "\n".join(
            line + " 1700000000000" if line.startswith("vllm:") else line
            for line in text.splitlines()
        )
    response.text = AsyncMock(return_value=text)

    metrics = await fetch("http://localhost", session)

    assert metrics is not None
    assert asdict(metrics) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fetch", [fetch_spec_decode_metrics, fetch_diffusion_metrics])
async def test_decode_metrics_ignore_non_prometheus_response(fetch):
    session = MagicMock(spec=aiohttp.ClientSession)
    response = session.get.return_value.__aenter__.return_value
    response.status = 200
    response.text = AsyncMock(return_value="<html>not a metrics endpoint</html>")
    assert await fetch("http://localhost", session) is None
