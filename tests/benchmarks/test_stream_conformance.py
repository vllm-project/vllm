# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the stream cases shared with the Rust `vllm-bench` client.

`stream_cases.json` holds recorded SSE streams and the result both benchmark
clients must report for each. The Rust side runs the same file in
`rust/src/bench/src/backends/mod.rs`.

Each case is fed to the client one fixture chunk per read, so a case that splits
JSON, a separator or a UTF-8 character across chunks tests exactly that split.
The same case also goes through aiohttp and a local socket, unless the client's
result changes when the whole stream arrives in one read or one byte per read:
TCP keeps neither the fixture's split nor any other, so that run could not be
relied on.

A case lists a client under `known_gap` while that client does not meet it yet.
`got` records what the client reports instead, for the fields that differ; every
other field must still match `expect`.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

pytestmark = pytest.mark.skip_global_cleanup

CASES_PATH = Path(__file__).parent / "stream_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
ENDPOINTS = {
    "openai": "/v1/completions",
    "openai-chat": "/v1/chat/completions",
    "openai-responses": "/v1/responses",
}
# The client-side estimate; a case expects another value only when the stream
# reports usage.
CLIENT_PROMPT_LEN = 100


def _chunks(case: dict[str, Any]) -> list[bytes]:
    if "chunks_hex" in case:
        return [bytes.fromhex(chunk) for chunk in case["chunks_hex"]]
    return [chunk.encode() for chunk in case["chunks"]]


def _input(case: dict[str, Any], base_url: str) -> RequestFuncInput:
    return RequestFuncInput(
        prompt="hello",
        api_url=base_url + ENDPOINTS[case["backend"]],
        prompt_len=CLIENT_PROMPT_LEN,
        output_len=3,
        model="test-model",
    )


class _Content:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _Response:
    def __init__(self, chunks: list[bytes]) -> None:
        self.status = 200
        self.reason = "OK"
        self.content = _Content(chunks)

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _Session:
    """Hands the client each chunk as one read, without a socket."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def post(self, **kwargs: Any) -> _Response:
        del kwargs
        return _Response(self._chunks)


def _direct(case: dict[str, Any], chunks: list[bytes]) -> RequestFuncOutput:
    request = ASYNC_REQUEST_FUNCS[case["backend"]]
    return asyncio.run(request(_input(case, "http://test"), _Session(chunks)))


async def _loopback(case: dict[str, Any]) -> RequestFuncOutput:
    """Serve the case over a raw socket, one write per chunk, then close."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        head = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        await reader.readexactly(length)
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"connection: close\r\n\r\n"
        )
        for chunk in _chunks(case):
            writer.write(chunk)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            request = ASYNC_REQUEST_FUNCS[case["backend"]]
            return await request(_input(case, f"http://127.0.0.1:{port}"), session)
    finally:
        server.close()
        await server.wait_closed()


def _observed(output: RequestFuncOutput) -> dict[str, Any]:
    return {
        "success": bool(output.success),
        "itl": len(output.itl),
        "text": output.generated_text,
        "output_tokens": output.output_tokens,
        "prompt_len": output.prompt_len,
        "error": output.error or "",
    }


def _mismatches(expect: dict[str, Any], got: dict[str, Any]) -> list[str]:
    bad = []
    for key, want in expect.items():
        if key == "error_contains":
            if want not in got["error"]:
                bad.append(f"error {got['error']!r} lacks {want!r}")
        elif key != "success" and expect["success"] is False:
            continue
        elif got[key] != want:
            bad.append(f"{key}: got {got[key]!r}, want {want!r}")
    return bad


def _gap_got(expect: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    """The `known_gap` entry `got` that describes this output."""
    fields = ("itl", "text", "output_tokens", "prompt_len")
    if got["success"] and not expect["success"]:
        return {"success": True, **{key: got[key] for key in fields}}
    if not got["success"]:
        # The last line names the exception when the error is a traceback.
        lines = [line for line in got["error"].splitlines() if line.strip()]
        return {"success": False, "error_contains": lines[-1] if lines else ""}
    return {
        key: got[key] for key in fields if key in expect and got[key] != expect[key]
    }


def _gap_expect(expect: dict[str, Any], gap_got: dict[str, Any]) -> dict[str, Any]:
    merged = {**expect, **gap_got}
    if gap_got.get("success") is True:
        merged.pop("error_contains", None)
    return merged


def _check(case: dict[str, Any], got: dict[str, Any]) -> list[str]:
    bad = _mismatches(case["expect"], got)
    gap = case.get("known_gap", {}).get("python")
    if gap is None:
        return bad
    if not bad:
        return [f"python now meets this case; drop its known_gap ({gap['pr']})"]
    if _mismatches(_gap_expect(case["expect"], gap["got"]), got):
        new_got = json.dumps(_gap_got(case["expect"], got))
        return [f"known_gap.python.got no longer matches ({gap['pr']}); got {new_got}"]
    return []


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_stream_case(case: dict[str, Any]):
    failures = _check(case, _observed(_direct(case, _chunks(case))))
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_stream_case_over_socket(case: dict[str, Any]):
    joined = b"".join(_chunks(case))
    split = _observed(_direct(case, _chunks(case)))
    whole = _observed(_direct(case, [joined]))
    bytewise = _observed(_direct(case, [joined[i : i + 1] for i in range(len(joined))]))
    if not split == whole == bytewise:
        pytest.skip("the result depends on the split, which TCP does not keep")
    failures = _check(case, _observed(asyncio.run(_loopback(case))))
    assert not failures, "; ".join(failures)
