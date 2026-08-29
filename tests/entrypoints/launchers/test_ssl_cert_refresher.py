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
async def test_serve_http_stops_ssl_refresher_when_server_exits(monkeypatch):
    config = SimpleNamespace(
        ssl=object(),
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        ssl_ca_certs="ca.pem",
        load=MagicMock(),
    )
    server = SimpleNamespace(serve=AsyncMock())
    ssl_cert_refresher = MagicMock()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_args: None)
    monkeypatch.setattr(launcher.uvicorn, "Config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(launcher, "NoSignalServer", lambda _config: server)
    monkeypatch.setattr(
        launcher,
        "SSLCertRefresher",
        lambda **_kwargs: ssl_cert_refresher,
    )
    monkeypatch.setattr(launcher, "watchdog_loop", AsyncMock())
    app = SimpleNamespace(routes=[], state=SimpleNamespace(engine_client=object()))

    shutdown = await launcher.serve_http(
        app, sock=None, enable_ssl_refresh=True, port=8000
    )
    try:
        ssl_cert_refresher.stop.assert_called_once_with()
    finally:
        await shutdown
        await asyncio.sleep(0)
