# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise a tokenizer-free custom pooling dataset through the benchmark CLI."""

import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Exact prompt payload from the failing benchmark run against a
# Prithvi-EO-2.0 pooling endpoint (URL-in / base64-out format).
_PRITHVI_PROMPT = {
    "data": {
        "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
        "data_format": "url",
        "out_data_format": "b64_json",
        "indices": [1, 2, 3, 8, 11, 12],
    },
    "priority": 0,
    "softmax": False,
}


@pytest.mark.benchmark
@pytest.mark.skip_global_cleanup
@pytest.mark.asyncio
async def test_custom_pooling_sends_object_prompt_without_tokenizer(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "pooling.jsonl"
    dataset_path.write_text(json.dumps({"prompt": _PRITHVI_PROMPT}) + "\n")
    received = []

    async def pooling(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.json_response({"usage": {"prompt_tokens": 1}})

    app = web.Application()
    app.router.add_post("/pooling", pooling)
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    async with TestServer(app) as server:
        args = parser.parse_args(
            [
                "--base-url",
                str(server.make_url("")).rstrip("/"),
                "--model",
                "prithvi",
                "--backend",
                "vllm-pooling",
                "--endpoint",
                "/pooling",
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(dataset_path),
                "--skip-tokenizer-init",
                "--num-prompts",
                "1",
                "--ready-check-timeout-sec",
                "0",
            ]
        )
        await main_async(args)

    [payload] = received
    assert "input" not in payload
    assert {key: payload[key] for key in _PRITHVI_PROMPT} == _PRITHVI_PROMPT
