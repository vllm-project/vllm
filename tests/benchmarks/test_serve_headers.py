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

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "with_headers,live_model",
    [(False, False), (True, False), (True, True)],
    ids=["public_http", "protected_http", "protected_model"],
)
def test_benchmark_preserves_decode_metrics_with_headers(
    tmp_path: Path, with_headers: bool, live_model: bool
) -> None:
    """Protected metrics must be collected using the same headers as inference."""
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    expected_headers = {"X-Benchmark-Tenant": "test-tenant"} if with_headers else {}

    async def run(upstream_url: str | None):
        completed = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal completed
            if any(request.headers.get(k) != v for k, v in expected_headers.items()):
                raise web.HTTPUnauthorized()
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
            completed += 1
            if upstream_url is not None:
                async with client.post(
                    upstream_url, json=await request.json()
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
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(dataset),
                "--skip-tokenizer-init",
                "--num-prompts",
                "1",
                "--disable-tqdm",
            ]
            for key, value in expected_headers.items():
                argv.extend(["--header", f"{key}={value}"])
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
    assert result["completed"] == 1
    assert result["spec_decode_accepted_tokens"] == 1
    assert result["spec_decode_acceptance_rate"] == 50.0
    assert result["diffusion_committed_tokens"] == 2
