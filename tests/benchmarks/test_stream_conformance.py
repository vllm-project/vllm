# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the stream cases shared with the Rust `vllm-bench` client.

`stream_cases.json` holds recorded SSE streams and the result both benchmark
clients must report for each. The Rust side runs the same file in
`rust/src/bench/src/backends/mod.rs`. A case lists a client under `known_gap`
while that client does not meet it yet.
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


async def _request(case: dict[str, Any]) -> RequestFuncOutput:
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
            # Keep each chunk a separate read on the client.
            await asyncio.sleep(0.005)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    backend = case["backend"]
    try:
        async with aiohttp.ClientSession() as session:
            return await ASYNC_REQUEST_FUNCS[backend](
                RequestFuncInput(
                    prompt="hello",
                    api_url=f"http://127.0.0.1:{port}{ENDPOINTS[backend]}",
                    prompt_len=CLIENT_PROMPT_LEN,
                    output_len=3,
                    model="test-model",
                ),
                session,
            )
    finally:
        server.close()
        await server.wait_closed()


def _mismatches(expect: dict[str, Any], output: RequestFuncOutput) -> list[str]:
    got = {
        "success": bool(output.success),
        "itl": len(output.itl),
        "text": output.generated_text,
        "output_tokens": output.output_tokens,
        "prompt_len": output.prompt_len,
    }
    bad = []
    for key, want in expect.items():
        if key == "error_contains":
            if want not in (output.error or ""):
                bad.append(f"error {output.error!r} lacks {want!r}")
        elif key != "success" and expect["success"] is False:
            continue
        elif got[key] != want:
            bad.append(f"{key}: got {got[key]!r}, want {want!r}")
    return bad


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_stream_case(case: dict[str, Any]):
    bad = _mismatches(case["expect"], asyncio.run(_request(case)))
    gap = case.get("known_gap", {}).get("python")
    if gap:
        # Strict: closing the gap must also remove it from the case file.
        assert bad, f"python now meets this case; drop its known_gap ({gap})"
    else:
        assert not bad, "; ".join(bad)
