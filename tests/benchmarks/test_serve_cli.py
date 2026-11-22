# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import urllib3
from aiohttp import web
from aiohttp.test_utils import TestServer
from prometheus_client.parser import text_string_to_metric_families

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


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
def test_bench_serve_warmups_do_not_prime_prefix_cache(server):
    """The ready check and warmups resend the first prompt; they must not leave it
    in the prefix cache."""
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "256",
        "--random-output-len",
        "4",
        "--num-prompts",
        "3",
        "--max-concurrency",
        "1",
        "--num-warmups",
        "2",
        "--ready-check-timeout-sec",
        "60",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    counters: dict[str, float] = {}
    metrics = requests.get(server.url_for("metrics")).text
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            counters[sample.name] = counters.get(sample.name, 0.0) + sample.value
    assert counters["vllm:prefix_cache_queries_total"] > 0
    assert counters["vllm:prefix_cache_hits_total"] == 0


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
    assert re.search(r"Probe requests completed:\s+[1-9]\d*", result.stdout)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_bench_serve_reports_failed_probes(tmp_path, capsys):
    probe_seen = asyncio.Event()
    failed_probes = 0

    async def completions(request):
        nonlocal failed_probes
        if (await request.json())["prompt"] == "Hi":
            failed_probes += 1
            probe_seen.set()
            raise web.HTTPServiceUnavailable()
        await asyncio.wait_for(probe_seen.wait(), timeout=10)
        return web.Response(
            content_type="text/event-stream",
            text='data: {"choices":[{"text":"x"}]}\n\n'
            'data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n',
        )

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt":"main workload"}\n')
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    async with TestServer(app) as server:
        args = ["--base-url", str(server.make_url("")).rstrip("/"), "--model", "stub"]
        args += ["--dataset-name", "custom", "--dataset-path", str(dataset)]
        args += ["--skip-tokenizer-init", "--skip-chat-template", "--num-prompts", "1"]
        args += ["--ready-check-timeout-sec", "0", "--probe-request-rate", "1"]
        args += ["--save-result", "--result-dir", str(tmp_path)]
        args += ["--result-filename", "result.json"]
        await main_async(parser.parse_args(args))

    stats = json.loads((tmp_path / "result.json").read_text())
    assert stats["completed"] == 1
    assert stats["probe_completed"] == 0
    assert stats["probe_failed"] == failed_probes > 0
    assert not any(key.startswith("probe_") and key.endswith("_ms") for key in stats)
    assert re.search(
        rf"Probe requests failed:\s+{failed_probes}\b", capsys.readouterr().out
    )


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
