# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import os
import signal
import socket
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve import run_comb
from vllm.benchmarks.sweep.server import ServerProcess

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires POSIX process groups"),
]


@pytest.mark.parametrize("stop_kwargs", [{}, {"grace_period": 1.0}])
@pytest.mark.parametrize("parent_exits", [False, True])
def test_stop_cleans_up_server_process_group(parent_exits: bool, stop_kwargs):
    """Sweep cleanup must terminate workers even if the server already exited."""
    script = """
import os
import signal
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1]))) as connection:
    if os.fork() == 0:
        # A graceful stop should signal the parent and allow it to drain workers.
        signal.signal(signal.SIGTERM, lambda *_: connection.sendall(b"T"))
        connection.sendall(b"R")
        connection.recv(1)
    elif sys.argv[2] == "True":
        os._exit(1)
    else:
        signal.pause()
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(10)
        server = ServerProcess(
            [
                sys.executable,
                "-c",
                script,
                str(listener.getsockname()[1]),
                str(parent_exits),
            ],
            [],
            show_stdout=True,
        )
        assert server.stop() is False
        try:
            with server:
                process = server._server_process
                connection, _ = listener.accept()
                with contextlib.closing(connection):
                    connection.settimeout(10)
                    assert connection.recv(1) == b"R"
                    if parent_exits:
                        assert process.wait(timeout=10) == 1
                    forced = server.stop(**stop_kwargs)
                    assert forced is ("grace_period" not in stop_kwargs)
                    assert server.stop() is False
                    assert connection.recv(1) == b"", "Server worker is still alive"
                    assert process.returncode is not None, "Server was not reaped"
        finally:
            # Keep failing regressions from leaking the deliberately orphaned worker.
            if server._server_process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(server._server_process.pid, signal.SIGKILL)
                server._server_process.wait(timeout=10)


@pytest.mark.parametrize(
    "env_key,benchmark_fails,cross_origin",
    [
        (None, False, False),
        ("env-key", False, False),
        (None, True, False),
        ("env-key", False, True),
    ],
)
def test_sweep_reset_uses_benchmark_authorization(
    tmp_path, monkeypatch, env_key, benchmark_fails, cross_origin
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_USE_RUST_BENCH", "0")
    if env_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", env_key)

    dataset = tmp_path / "prompts.jsonl"
    if not benchmark_fails:
        dataset.write_text('{"prompt":"hello"}\n')

    reset_calls = []
    explicit = "Bearer cli=key"
    reset_authorization = f"Bearer {env_key}" if cross_origin else explicit

    async def reset(request):
        reset_calls.append(request.path)
        if request.headers.getall("Authorization", []) != [reset_authorization]:
            return web.Response(status=401)
        return web.Response()

    async def completions(request):
        if request.headers.getall("Authorization", []) != [explicit]:
            return web.Response(status=401)
        return web.Response(
            content_type="text/event-stream",
            text=(
                'data: {"choices":[{"text":"hello"}]}\n\n'
                'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    async def sweep():
        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        for path in ServerProcess.VLLM_RESET_CACHE_ENDPOINTS:
            app.router.add_post(path, reset)

        async with contextlib.AsyncExitStack() as stack:
            managed = await stack.enter_async_context(TestServer(app))
            if cross_origin:
                bench_app = web.Application()
                bench_app.router.add_post("/v1/completions", completions)
                target = await stack.enter_async_context(TestServer(bench_app))
                destination = [f"--base_url={str(target.make_url('')).rstrip('/')}"]
            else:
                destination = ["--port", str(managed.port)]

            server = ServerProcess(
                ["vllm", "--port", str(managed.port)], [], show_stdout=True
            )
            bench_cmd = [
                "vllm",
                "bench",
                "serve",
                *destination,
                "--model",
                "stub",
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(dataset),
                "--skip-tokenizer-init",
                "--skip-chat-template",
                "--num-prompts",
                "1",
                "--ready-check-timeout-sec",
                "0",
                "--header",
                "Authorization=Bearer replaced",
            ]
            return await asyncio.to_thread(
                run_comb,
                server,
                bench_cmd,
                serve_comb=ParameterSweepItem(),
                bench_comb=ParameterSweepItem({"header": f"authorization={explicit}"}),
                link_vars=[],
                base_path=tmp_path / "results",
                num_runs=1,
                dry_run=False,
                continue_on_error=benchmark_fails,
            )

    results = asyncio.run(sweep())
    assert reset_calls == ServerProcess.VLLM_RESET_CACHE_ENDPOINTS
    if benchmark_fails:
        assert results == []
        assert (tmp_path / "results" / "run=0.failure.json").exists()
    else:
        assert results and results[0]["completed"] == 1
