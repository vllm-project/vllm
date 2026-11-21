# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise benchmark headers through HTTP and a real inference server."""

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import (
    _align_prompts_to_server_tokenizer,
    add_cli_args,
    main_async,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "with_headers,live_model,auth_mode",
    [
        (False, False, None),
        (True, False, None),
        (True, False, "env"),
        (True, False, "override"),
        (True, False, "lowercase"),
        (True, True, "env"),
    ],
    ids=[
        "public_http",
        "protected_http",
        "env_auth",
        "override_auth",
        "lowercase_auth",
        "protected_model",
    ],
)
def test_benchmark_preserves_decode_metrics_with_headers(
    tmp_path: Path,
    with_headers: bool,
    live_model: bool,
    auth_mode: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected metrics must be collected using the same headers as inference."""
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    alignment_calls: list[str] = []
    expected_headers = {"X-Benchmark-Tenant": "test-tenant"} if with_headers else {}

    cli_headers = expected_headers.copy()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if auth_mode:
        monkeypatch.setenv("OPENAI_API_KEY", "mock-benchmark-key")
        authorization = "Bearer mock-benchmark-key"
        if auth_mode in ("override", "lowercase"):
            authorization = "Bearer mock-custom-key"
            key = "authorization" if auth_mode == "lowercase" else "Authorization"
            cli_headers[key] = authorization
        expected_headers["Authorization"] = authorization

    async def run(upstream_url: str | None):
        completed = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal completed
            if any(request.headers.get(k) != v for k, v in expected_headers.items()):
                raise web.HTTPUnauthorized()
            if auth_mode:
                assert len(request.headers.getall("Authorization")) == 1
            if request.path in ("/tokenize", "/detokenize"):
                alignment_calls.append(request.path)
            if request.path == "/metrics":
                # The gateway exposes deterministic counters for its own traffic.
                return web.Response(
                    text="\n".join(
                        f"vllm:{name}_total {completed * count}"
                        for name, count in {
                            "spec_decode_num_drafts": 1,
                            "spec_decode_num_draft_tokens": 2,
                            "spec_decode_num_accepted_tokens": 1,
                            "spec_decode_num_verified_tokens": 2,
                            "diffusion_num_denoising_steps": 3,
                            "diffusion_num_canvas_positions": 6,
                            "diffusion_num_committed_tokens": 2,
                        }.items()
                    )
                )
            if request.path == "/v1/completions":
                completed += 1
            if upstream_url is not None:
                async with client.post(
                    upstream_url.removesuffix("v1/completions")
                    + request.path.lstrip("/"),
                    json=await request.json(),
                ) as response:
                    return web.Response(
                        status=response.status,
                        body=await response.read(),
                        headers={"Content-Type": response.headers["Content-Type"]},
                    )
            return web.Response(
                content_type="text/event-stream",
                text='data: {"choices":[{"text":"hi"}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n",
            )

        app = web.Application()
        app.router.add_get("/metrics", handler)
        app.router.add_post("/v1/completions", handler)
        app.router.add_post("/tokenize", handler)
        app.router.add_post("/detokenize", handler)
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client,
            TestServer(app) as server,
        ):
            parser = FlexibleArgumentParser()
            add_cli_args(parser)
            argv = [
                "--base-url",
                str(server.make_url("")).rstrip("/"),
                "--model",
                "test-model",
                *(
                    [
                        "--dataset-name",
                        "random",
                        "--tokenizer",
                        "HuggingFaceTB/SmolLM2-135M-Instruct",
                        "--random-input-len",
                        "8",
                        "--random-output-len",
                        "1",
                    ]
                    if live_model
                    else [
                        "--dataset-name",
                        "custom",
                        "--dataset-path",
                        str(dataset),
                        "--skip-tokenizer-init",
                    ]
                ),
                "--num-prompts",
                "1",
                "--disable-tqdm",
            ]
            if cli_headers:
                argv.extend(["--header", *[f"{k}={v}" for k, v in cli_headers.items()]])
            return await main_async(parser.parse_args(argv))

    with ExitStack() as stack:
        upstream_url = None
        if live_model:
            from tests.utils import RemoteOpenAIServer

            upstream = stack.enter_context(
                RemoteOpenAIServer(
                    "HuggingFaceTB/SmolLM2-135M-Instruct",
                    [
                        "--served-model-name",
                        "test-model",
                        "--dtype",
                        "float32",
                        "--max-model-len",
                        "128",
                        "--max-num-batched-tokens",
                        "128",
                        "--max-num-seqs",
                        "2",
                        "--enforce-eager",
                    ],
                    env_dict={"VLLM_CPU_KVCACHE_SPACE": "1"},
                    max_wait_seconds=300,
                )
            )
            upstream_url = upstream.url_for("v1", "completions")
        result = asyncio.run(run(upstream_url))
    if live_model:
        assert "/tokenize" in alignment_calls
    assert result["completed"] == 1
    assert result["spec_decode_accepted_tokens"] == 1
    assert result["spec_decode_acceptance_rate"] == 50.0
    assert result["diffusion_committed_tokens"] == 2


def test_tokenizer_alignment_sends_headers() -> None:
    """random/prefix_repetition prompt alignment must pass --header too."""
    headers = {"X-Benchmark-Tenant": "test-tenant"}
    calls: list[str] = []

    async def tokenize(request: web.Request) -> web.Response:
        if request.headers.get("X-Benchmark-Tenant") != "test-tenant":
            raise web.HTTPUnauthorized()
        calls.append(request.path)
        return web.json_response({"tokens": [1, 2, 3]})

    async def detokenize(request: web.Request) -> web.Response:
        if request.headers.get("X-Benchmark-Tenant") != "test-tenant":
            raise web.HTTPUnauthorized()
        calls.append(request.path)
        assert (await request.json())["tokens"] == [1, 2]
        return web.json_response({"prompt": "aligned"})

    async def run() -> list[SampleRequest]:
        app = web.Application()
        app.router.add_post("/tokenize", tokenize)
        app.router.add_post("/detokenize", detokenize)
        async with TestServer(app) as server:
            base_url = str(server.make_url("")).rstrip("/")
            reqs = [SampleRequest(prompt="hello", prompt_len=2, expected_output_len=1)]
            return await _align_prompts_to_server_tokenizer(
                base_url, "test-model", reqs, headers=headers
            )

    result = asyncio.run(run())
    assert result[0].prompt == "aligned"
    assert result[0].prompt_len == 2
    assert calls == ["/tokenize", "/tokenize", "/detokenize"]
