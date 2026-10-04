# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import tempfile
from pathlib import Path
from ssl import SSLContext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.launchers import launcher
from vllm.entrypoints.launchers.utils.ssl import SSLCertRefresher


class MockSSLContext(SSLContext):
    def __init__(self):
        self.load_cert_chain_count = 0
        self.load_ca_count = 0

    def load_cert_chain(
        self,
        certfile,
        keyfile=None,
        password=None,
    ):
        self.load_cert_chain_count += 1

    def load_verify_locations(
        self,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        self.load_ca_count += 1


def create_file() -> str:
    with tempfile.NamedTemporaryFile(dir="/tmp", delete=False) as f:
        return f.name


def touch_file(path: str) -> None:
    Path(path).touch()


async def wait_for_counts(
    ssl_context: MockSSLContext,
    *,
    cert_chain_count: int,
    ca_count: int,
    timeout: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if (
            ssl_context.load_cert_chain_count >= cert_chain_count
            and ssl_context.load_ca_count >= ca_count
        ):
            return

        if asyncio.get_running_loop().time() >= deadline:
            assert ssl_context.load_cert_chain_count >= cert_chain_count
            assert ssl_context.load_ca_count >= ca_count

        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_ssl_refresher():
    ssl_context = MockSSLContext()
    key_path = create_file()
    cert_path = create_file()
    ca_path = create_file()
    ssl_refresher = SSLCertRefresher(ssl_context, key_path, cert_path, ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 0
    assert ssl_context.load_ca_count == 0

    touch_file(key_path)
    await wait_for_counts(
        ssl_context,
        cert_chain_count=1,
        ca_count=0,
    )
    assert ssl_context.load_ca_count == 0

    touch_file(cert_path)
    touch_file(ca_path)
    await wait_for_counts(
        ssl_context,
        cert_chain_count=2,
        ca_count=1,
    )

    ssl_refresher.stop()
    await asyncio.sleep(0)
    cert_chain_count = ssl_context.load_cert_chain_count
    ca_count = ssl_context.load_ca_count

    touch_file(cert_path)
    touch_file(ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == cert_chain_count
    assert ssl_context.load_ca_count == ca_count


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", [None, "serve", "signal", "shutdown_task"])
async def test_serve_http_stops_ssl_refresher_when_server_exits(
    monkeypatch, failure_stage
):
    config = SimpleNamespace(
        ssl=object(),
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        ssl_ca_certs="ca.pem",
        load=MagicMock(),
    )
    server_error = RuntimeError("server failed") if failure_stage == "serve" else None
    server = SimpleNamespace(serve=AsyncMock(side_effect=server_error))
    ssl_cert_refresher = MagicMock()
    loop = asyncio.get_running_loop()

    def add_signal_handler(*_args):
        if failure_stage == "signal":
            raise RuntimeError("signal failed")

    create_task = loop.create_task

    def create_task_or_fail(coro, **kwargs):
        if (
            failure_stage == "shutdown_task"
            and coro.cr_code.co_name == "handle_shutdown"
        ):
            coro.close()
            raise RuntimeError("shutdown_task failed")
        return create_task(coro, **kwargs)

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    monkeypatch.setattr(loop, "create_task", create_task_or_fail)
    monkeypatch.setattr(launcher.uvicorn, "Config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(launcher, "NoSignalServer", lambda _config: server)
    monkeypatch.setattr(
        launcher,
        "SSLCertRefresher",
        lambda **_kwargs: ssl_cert_refresher,
    )
    monkeypatch.setattr(launcher, "watchdog_loop", AsyncMock())
    app = SimpleNamespace(routes=[], state=SimpleNamespace(engine_client=object()))

    shutdown = None
    try:
        if failure_stage is None:
            shutdown = await launcher.serve_http(
                app, sock=None, enable_ssl_refresh=True, port=8000
            )
        else:
            error = "server" if failure_stage == "serve" else failure_stage
            with pytest.raises(RuntimeError, match=f"{error} failed"):
                await launcher.serve_http(
                    app, sock=None, enable_ssl_refresh=True, port=8000
                )
        ssl_cert_refresher.stop.assert_called_once_with()
    finally:
        if shutdown is not None:
            await shutdown
        await asyncio.sleep(0)
