# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


async def _max_tokens_sent(argv: list[str]) -> set[int]:
    """Returns the `max_tokens` a stub server gets from `vllm bench serve`."""
    max_tokens: set[int] = set()

    async def completions(request: web.Request) -> web.StreamResponse:
        max_tokens.add((await request.json())["max_tokens"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in (
            {"choices": [{"text": "x"}]},
            {"usage": {"completion_tokens": 1}},
        ):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    async with TestServer(app) as server:
        base_url = str(server.make_url("")).rstrip("/")
        await main_async(parser.parse_args([*argv, "--base-url", base_url]))
    return max_tokens


@pytest.mark.benchmark
def test_bench_serve_output_len_maps_to_speed_bench(tmp_path: Path) -> None:
    """`vllm bench serve --output-len` must set `max_tokens` for speed_bench."""
    with (tmp_path / "qualitative.jsonl").open("w") as f:
        for i in range(4):
            message = {"role": "user", "content": f"row {i}"}
            f.write(json.dumps({"category": "writing", "messages": [message]}) + "\n")
    argv = ["--model", "stub", "--tokenizer", "openai-community/gpt2"]
    argv += ["--dataset-name", "speed_bench", "--dataset-path", str(tmp_path)]
    argv += ["--num-prompts", "4", "--skip-chat-template", "--output-len", "7"]

    assert asyncio.run(_max_tokens_sent(argv)) == {7}
