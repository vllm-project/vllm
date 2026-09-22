# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import tempfile
from pathlib import Path
from ssl import SSLContext

import pytest

from vllm.entrypoints.serve.utils.ssl import SSLCertRefresher


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
