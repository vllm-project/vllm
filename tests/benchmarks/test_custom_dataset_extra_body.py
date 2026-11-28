# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.datasets import CustomDataset
from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


@pytest.mark.benchmark
def test_extra_body_is_attached_per_entry(tmp_path: Path) -> None:
    rows = [
        {"prompt": "a", "extra_body": {"tools": [{"type": "function"}]}},
        {"prompt": "b"},
        {"prompt": "c", "extra_body": None},
        {"prompt": "d", "extra_body": {"state": {"page": "search"}}},
    ]
    dataset = CustomDataset(
        dataset_path=_write_jsonl(tmp_path / "data.jsonl", rows),
        disable_shuffle=True,
    )
    requests = dataset.sample(None, num_requests=4, no_oversample=True)

    assert [r.request_overrides for r in requests] == [
        {"tools": [{"type": "function"}]},
        None,
        None,
        {"state": {"page": "search"}},
    ]


@pytest.mark.benchmark
@pytest.mark.parametrize("bad", ["tools", ["tools"]])
def test_extra_body_must_be_an_object(tmp_path: Path, bad) -> None:
    path = _write_jsonl(
        tmp_path / "data.jsonl",
        [{"prompt": "a"}, {"prompt": "b", "extra_body": bad}],
    )
    with pytest.raises(ValueError, match="Line 2 of .*'extra_body' must be"):
        CustomDataset(dataset_path=path, disable_shuffle=True)


async def _bodies_sent(endpoint: str, response, argv: list[str]) -> list[dict]:
    """Runs `vllm bench serve` against a stub server; returns the bodies it got."""
    bodies: list[dict] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        return await response(request)

    app = web.Application()
    app.router.add_post(endpoint, handler)
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    async with TestServer(app) as server:
        base_url = str(server.make_url("")).rstrip("/")
        await main_async(parser.parse_args([*argv, "--base-url", base_url]))
    return bodies


async def _completion_stream(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for chunk in ({"choices": [{"text": "x"}]}, {"usage": {"completion_tokens": 1}}):
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


async def _decision(request: web.Request) -> web.Response:
    return web.json_response({"usage": {"input_tokens": 7, "output_tokens": 0}})


def _bench_argv(dataset_path: str, *extra: str) -> list[str]:
    return [
        *("--model", "stub", "--tokenizer", "openai-community/gpt2"),
        *("--dataset-name", "custom", "--dataset-path", dataset_path),
        *("--num-prompts", "3", "--disable-shuffle", "--custom-output-len", "4"),
        *extra,
    ]


@pytest.mark.benchmark
def test_bench_serve_sends_each_entry_its_extra_body(tmp_path: Path) -> None:
    """Entry fields reach the server merged over --extra-body; entry keys win."""
    rows = [
        {"prompt": "first", "extra_body": {"user": "row-0", "stop": ["\n"]}},
        {"prompt": "second", "extra_body": {"user": "row-1"}},
        {"prompt": "third"},
    ]
    argv = _bench_argv(
        _write_jsonl(tmp_path / "data.jsonl", rows),
        *("--skip-chat-template", "--extra-body", '{"user": "all", "top_k": 5}'),
    )
    bodies = asyncio.run(_bodies_sent("/v1/completions", _completion_stream, argv))

    sent = {b["prompt"]: b for b in bodies}
    assert sent["first"]["user"] == "row-0"
    assert sent["first"]["stop"] == ["\n"]
    assert sent["second"]["user"] == "row-1"
    assert "stop" not in sent["second"]
    assert sent["third"]["user"] == "all"
    assert all(b["top_k"] == 5 for b in bodies)


@pytest.mark.benchmark
def test_bench_serve_sends_each_decision_its_state_and_questions(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "prompt": "unused",
            "extra_body": {
                "state": {"request": f"request {i}", "page": "search"},
                "questions": {
                    f"q{i}": {"type": "choice", "criteria": {"a": "A", "b": "B"}}
                },
            },
        }
        for i in range(3)
    ]
    argv = _bench_argv(
        _write_jsonl(tmp_path / "data.jsonl", rows),
        *("--backend", "vllm-systemone", "--endpoint", "/v1/systemone"),
    )
    bodies = asyncio.run(_bodies_sent("/v1/systemone", _decision, argv))

    by_request = {b["state"]["request"]: b for b in bodies}
    assert set(by_request) == {"request 0", "request 1", "request 2"}
    for i in range(3):
        body = by_request[f"request {i}"]
        assert body["state"]["page"] == "search"
        assert list(body["questions"]) == [f"q{i}"]
