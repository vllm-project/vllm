# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_proxy_module():
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/disaggregated_serving/"
        "disagg_proxy_pushconnector_demo.py"
    )
    spec = importlib.util.spec_from_file_location("nixl_push_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _async_value(value):
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "expected_content_type"),
    [
        (True, "text/event-stream; charset=utf-8"),
        (False, "application/json"),
    ],
)
async def test_push_response_content_type_matches_stream_mode(
    stream, expected_content_type
):
    module = _load_proxy_module()
    proxy = module.PushProxy(
        prefill_instances=["prefill:8001"],
        decode_instances=["decode:8002"],
        model="test-model",
        scheduling_policy=module.RoundRobinSchedulingPolicy(),
        prefill_engine_id="prefill-engine",
        prefill_kv_host="10.0.0.1",
        prefill_side_channel_port=5600,
        prefill_tp_size=1,
        prefill_pp_size=1,
    )

    async def forward_request(url, data, headers, use_chunked=True):
        yield b"{}"

    proxy.forward_request = forward_request
    raw_request = SimpleNamespace(
        json=lambda: _async_value(
            {
                "model": "MiniMax-M3",
                "max_tokens": 4,
                "stream": stream,
            }
        )
    )

    response = await proxy._push_completion(raw_request, "/v1/chat/completions")
    assert response.headers["content-type"] == expected_content_type
    assert [chunk async for chunk in response.body_iterator] == [b"{}"]
