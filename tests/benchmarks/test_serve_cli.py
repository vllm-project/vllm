# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypedDict

import aiohttp
import pytest
import requests
import urllib3
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    StreamedResponseHandler,
    async_request_openai_chat_completions,
    async_request_openai_completions,
)

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
INLINE_USAGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class _InlineUsageState(TypedDict):
    text: str
    usage: dict[str, int] | None
    inline: bool
    trailer: int
    done: bool


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function")
def ssl_server():
    """Start a vLLM server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
        "--probe-request-rate",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    # Probes use their own session, which must also honor --insecure.
    assert "Probe requests completed" in result.stdout


@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.fixture(scope="module")
def inline_usage_server():
    args = [
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.45",
        "--dtype",
        "half",
        "--attention-backend",
        "TRITON_ATTN",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(INLINE_USAGE_MODEL, args) as remote_server:
        yield remote_server


@pytest.mark.benchmark
@pytest.mark.parametrize("chat", [False, True], ids=["completions", "chat"])
def test_benchmark_reads_inline_usage_from_real_server(inline_usage_server, chat):
    """Keep real streamed usage when a proxy omits the usage-only trailer."""

    async def run():
        path = "/v1/chat/completions" if chat else "/v1/completions"
        seen: _InlineUsageState = {
            "text": "",
            "usage": None,
            "inline": False,
            "trailer": 0,
            "done": False,
        }
        async with aiohttp.ClientSession() as session:

            async def proxy(request):
                body = await request.json()
                async with session.post(
                    inline_usage_server.url_for(path), json=body
                ) as upstream:
                    upstream.raise_for_status()
                    response = web.StreamResponse(
                        headers={"Content-Type": "text/event-stream"}
                    )
                    await response.prepare(request)
                    parser = StreamedResponseHandler()
                    async for raw in upstream.content.iter_any():
                        for message in parser.add_chunk(raw):
                            data = message.removeprefix("data: ")
                            if data == "[DONE]":
                                seen["done"] = True
                            else:
                                frame = json.loads(data)
                                choices = frame.get("choices")
                                usage = frame.get("usage")
                                if usage:
                                    seen["usage"] = usage
                                if not choices and usage:
                                    seen["trailer"] += 1
                                    continue
                                if choices:
                                    choice = choices[0]
                                    text = (
                                        choice.get("delta", {}).get("content")
                                        if chat
                                        else choice.get("text")
                                    ) or ""
                                    seen["text"] += text
                                    seen["inline"] |= bool(text and usage)
                            await response.write(f"{message}\n\n".encode())
                    await response.write_eof()
                    return response

            app = web.Application()
            app.router.add_post(path, proxy)
            async with TestServer(app, host="127.0.0.1") as proxy_server:
                request = RequestFuncInput(
                    prompt="Write a short greeting.",
                    api_url=str(proxy_server.make_url(path)),
                    prompt_len=999,
                    output_len=8,
                    model=INLINE_USAGE_MODEL,
                    ignore_eos=True,
                    extra_body={
                        "temperature": 0,
                        "stream_options": {
                            "include_usage": True,
                            "continuous_usage_stats": True,
                        },
                    },
                )
                func = (
                    async_request_openai_chat_completions
                    if chat
                    else async_request_openai_completions
                )
                output = await func(request, session)
        assert seen["inline"] and seen["trailer"] == 1 and seen["done"], seen
        assert output.success, output.error
        assert output.generated_text == seen["text"]
        assert seen["usage"] is not None
        assert output.prompt_len == seen["usage"]["prompt_tokens"]
        assert output.output_tokens == seen["usage"]["completion_tokens"] == 8

    asyncio.run(run())
