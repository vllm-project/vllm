# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for automatic benchmark model discovery and authentication."""

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = [pytest.mark.benchmark, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    "environment_key,explicit_header",
    [
        pytest.param("test-key", None, id="environment-key"),
        pytest.param(
            "test-key", "X-Trace=auth-test", id="environment-key-with-extra-header"
        ),
        pytest.param(
            "wrong-key", "Authorization=Bearer test-key", id="explicit-header-wins"
        ),
        pytest.param(
            "wrong-key", "authorization=Bearer test-key", id="lowercase-header-wins"
        ),
        pytest.param(None, None, id="missing-key"),
    ],
)
def test_bench_serve_authenticated_model_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str | None,
    explicit_header: str | None,
) -> None:
    """Model discovery must use the same credentials as generation requests."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost,::1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if environment_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", environment_key)
    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": [1, 2, 3], "output_tokens": 1}) + "\n"
    )
    seen: list[tuple[str, str | None]] = []
    trace_headers: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        authorization = request.headers.get("Authorization")
        seen.append((request.path, authorization))
        trace_headers.append(request.headers.get("X-Trace"))
        if authorization != "Bearer test-key":
            return web.json_response({"error": "Unauthorized"}, status=401)
        if request.path == "/v1/models":
            return web.json_response(
                {"data": [{"id": "test-model", "root": "test-model"}]}
            )
        assert (await request.json())["model"] == "test-model"
        chunks = [
            {"choices": [{"text": "ok", "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        ]
        stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return web.Response(
            text=stream + "data: [DONE]\n\n", content_type="text/event-stream"
        )

    async def run() -> None:
        app = web.Application()
        app.router.add_get("/v1/models", handler)
        app.router.add_post("/v1/completions", handler)
        async with TestServer(app) as server:
            parser = FlexibleArgumentParser()
            add_cli_args(parser)
            command = [
                "--base-url",
                str(server.make_url("/")).rstrip("/"),
                "--backend",
                "openai",
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(dataset_path),
                "--skip-tokenizer-init",
                "--skip-chat-template",
                "--num-prompts",
                "1",
                "--num-warmups",
                "0",
                "--disable-tqdm",
                "--ready-check-timeout-sec",
                "2",
            ]
            if explicit_header is not None:
                command += ["--header", explicit_header]
            args = parser.parse_args(command)
            if environment_key is None and explicit_header is None:
                with pytest.raises(RuntimeError, match="401"):
                    await asyncio.wait_for(main_async(args), timeout=15)
                assert seen == [("/v1/models", None)]
            else:
                result = await asyncio.wait_for(main_async(args), timeout=15)
                assert result["completed"] == 1
                assert result["failed"] == 0
                assert seen[0] == ("/v1/models", "Bearer test-key")
                assert any(path == "/v1/completions" for path, _ in seen)
                assert all(header == "Bearer test-key" for _, header in seen)
                if explicit_header == "X-Trace=auth-test":
                    assert all(header == "auth-test" for header in trace_headers)

    asyncio.run(run())
