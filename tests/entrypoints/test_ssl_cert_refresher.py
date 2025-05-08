# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
from pathlib import Path
from ssl import SSLContext

import pytest

from vllm.entrypoints.ssl import SSLCertRefresher


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
    with tempfile.NamedTemporaryFile(dir='/tmp', delete=False) as f:
        return f.name


def touch_file(path: str) -> None:
    Path(path).touch()


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
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 1
    assert ssl_context.load_ca_count == 0

    touch_file(cert_path)
    touch_file(ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 2
    assert ssl_context.load_ca_count == 1

    ssl_refresher.stop()

    touch_file(cert_path)
    touch_file(ca_path)
    await asyncio.sleep(1)
    assert ssl_context.load_cert_chain_count == 2
    assert ssl_context.load_ca_count == 1
