# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for https://github.com/vllm-project/vllm/issues/57157.

Server-side request forgery via multimodal media URLs: the domain allowlist
(`allowed_media_domains`) is checked once against the *initial* URL, while
redirects are followed unchecked, and resolved IPs are never validated, so
loopback / RFC1918 / link-local destinations are reachable by default.

All servers below bind to loopback only; no real cloud metadata or external
targets are involved.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from vllm.exceptions import VLLMUnprocessableEntityError
from vllm.multimodal.media.connector import MediaConnector

pytestmark = pytest.mark.cpu_test


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _PNGHandler(BaseHTTPRequestHandler):
    payload = b""
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        body = type(self).payload
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _redirect_handler(target_url: str):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *args):
            pass

    return _Handler


@pytest.fixture
def png_payload():
    return _png_bytes()


@pytest.fixture
def image_server(png_payload):
    _PNGHandler.payload = png_payload
    _PNGHandler.hits = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PNGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture
def redirect_server(image_server):
    target = f"http://blocked.test:{image_server.server_port}/img.png"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _redirect_handler(target))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _fake_getaddrinfo(host, port, *args, **kwargs):
    mapping = {
        "allowed.test": "127.0.0.1",
        "blocked.test": "127.0.0.1",
    }
    if host in mapping:
        ip = mapping[host]
        family = socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]
    return _REAL_GETADDRINFO(host, port, *args, **kwargs)


_REAL_GETADDRINFO = socket.getaddrinfo


@pytest.fixture
def controlled_dns():
    with patch.object(socket, "getaddrinfo", side_effect=_fake_getaddrinfo):
        yield


def test_redirect_target_domain_rechecked(redirect_server, controlled_dns, monkeypatch):
    """A 302 to a non-allowlisted host must not be followed.

    Private-IP validation is opted out here to isolate the domain
    re-check: both test hosts are loopback. Fails before the #57157 fix
    (the redirect is followed and fetched); passes after.
    """
    monkeypatch.setenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", "1")
    first_url = f"http://allowed.test:{redirect_server.server_port}/start"
    connector = MediaConnector(allowed_media_domains=["allowed.test"])
    with pytest.raises(
        VLLMUnprocessableEntityError, match="Failed to fetch"
    ) as exc_info:
        connector.fetch_image(first_url)
    assert "allowed domains" in str(exc_info.value.__cause__)
    assert _PNGHandler.hits == 0


@pytest.mark.asyncio
async def test_redirect_target_domain_rechecked_async(
    redirect_server, controlled_dns, monkeypatch
):
    monkeypatch.setenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", "1")
    first_url = f"http://allowed.test:{redirect_server.server_port}/start"
    connector = MediaConnector(allowed_media_domains=["allowed.test"])
    with pytest.raises(
        VLLMUnprocessableEntityError, match="Failed to fetch"
    ) as exc_info:
        await connector.fetch_image_async(first_url)
    assert "allowed domains" in str(exc_info.value.__cause__)
    assert _PNGHandler.hits == 0


def test_loopback_blocked_by_default(image_server, monkeypatch):
    """Direct fetches of loopback IPs are rejected unless explicitly opted in.

    Fails before the #57157 fix: the fetch succeeds.
    """
    monkeypatch.delenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", raising=False)
    url = f"http://127.0.0.1:{image_server.server_port}/img.png"
    with pytest.raises(
        VLLMUnprocessableEntityError, match="Failed to fetch"
    ) as exc_info:
        MediaConnector().fetch_image(url)
    assert "non-public IP" in str(exc_info.value.__cause__)
    assert _PNGHandler.hits == 0


def test_private_ip_opt_out_allows_loopback(image_server, monkeypatch):
    """Escape hatch for tests / trusted networks: fetching works when opted in."""
    monkeypatch.setenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", "1")
    url = f"http://127.0.0.1:{image_server.server_port}/img.png"
    image = MediaConnector().fetch_image(url)
    assert image.size == (1, 1)
    assert _PNGHandler.hits == 1
