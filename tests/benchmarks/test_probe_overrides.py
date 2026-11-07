# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for probe request generation-length isolation in serve benchmark."""

import ast
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

import vllm.benchmarks.lib.endpoint_request_func as client
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput


async def _run_probe_test(endpoint: str, function, field: str) -> None:
    root = Path(__file__).resolve().parents[2]
    source_path = root / "vllm" / "benchmarks" / "serve.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    benchmark = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "benchmark"
    )
    loop = next(
        n
        for n in benchmark.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "probe_loop"
    )
    code = compile(ast.Module(body=[loop], type_ignores=[]), str(source_path), "exec")

    captured = []
    stop = asyncio.Event()

    async def handler(request: web.Request) -> web.Response:
        payload = await request.json()
        captured.append(payload)
        stop.set()
        if request.path.endswith("responses"):
            events = [
                {"type": "response.output_text.delta", "delta": "x"},
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
                },
            ]
        else:
            is_chat = "/chat/" in request.path
            content = {"delta": {"content": "x"}} if is_chat else {"text": "x"}
            events = [{"choices": [{"index": 0, "finish_reason": "stop", **content}]}]
        body = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        body += "data: [DONE]\n\n"
        return web.Response(text=body, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]

    try:
        # Test with generation length override in extra_body (e.g. 64 tokens)
        extra = {field: 64, "custom_field": "keep_me"}
        outputs = []
        test_input = RequestFuncInput(
            prompt="main workload prompt",
            api_url=f"http://127.0.0.1:{port}/v1/{endpoint}",
            prompt_len=3,
            output_len=64,
            model="fixture",
            extra_body=extra,
        )
        scope = {
            "replace": replace,
            "asyncio": asyncio,
            "test_input": test_input,
            "probe_request_rate": 1000,
            "probe_stop": stop,
            "probe_outputs": outputs,
            "request_func": function,
            "make_session": lambda _limit: aiohttp.ClientSession(),
        }
        exec(code, scope)
        await asyncio.wait_for(scope["probe_loop"](), timeout=5)
        assert len(captured) == 1
        assert outputs[0].success
        # Probe limit must be 1, not 64
        assert captured[0][field] == 1
        # Custom non-generation fields should be preserved
        assert captured[0].get("custom_field") == "keep_me"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    ("endpoint", "function", "field"),
    [
        ("completions", client.async_request_openai_completions, "max_tokens"),
        (
            "chat/completions",
            client.async_request_openai_chat_completions,
            "max_completion_tokens",
        ),
        (
            "responses",
            client.async_request_openai_responses,
            "max_output_tokens",
        ),
    ],
)
def test_probe_loop_strips_generation_length_overrides(
    endpoint: str, function, field: str
) -> None:
    """Probes must stay single-token even with extra_body token limits."""
    asyncio.run(_run_probe_test(endpoint, function, field))
