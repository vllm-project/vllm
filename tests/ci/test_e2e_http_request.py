# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/e2e_http_request.py"
)


class RequestHandler(BaseHTTPRequestHandler):
    request_body = b""
    content_type = ""

    def do_GET(self) -> None:
        if self.path == "/failure":
            self.send_response(503)
        else:
            self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ready":true}')

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).request_body = self.rfile.read(length)
        type(self).content_type = self.headers["Content-Type"]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"accepted":true}')

    def log_message(self, format: str, *args: object) -> None:
        pass


class ProxyRequestHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_GET(self) -> None:
        type(self).request_count += 1
        self.send_response(502)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def run_server(
    handler: type[BaseHTTPRequestHandler] = RequestHandler,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_request_avoids_broken_curl_under_contaminated_environment(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "echo 'curl: symbol lookup error: /lib64/libldap.so.2: undefined symbol: "
        "EVP_md2, version OPENSSL_3.0.0' >&2\n"
        "exit 127\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    contaminated_env = os.environ.copy()
    contaminated_env["PATH"] = f"{fake_bin}:{contaminated_env['PATH']}"
    contaminated_env["LD_LIBRARY_PATH"] = str(tmp_path / "conda-lib")

    broken_curl = subprocess.run(
        ["curl", "--version"],
        env=contaminated_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert broken_curl.returncode == 127
    assert "EVP_md2" in broken_curl.stderr

    server, thread = run_server()
    proxy, proxy_thread = run_server(ProxyRequestHandler)
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    contaminated_env.update(
        {
            "ALL_PROXY": proxy_url,
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "NO_PROXY": "",
            "all_proxy": proxy_url,
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
            "no_proxy": "",
        }
    )
    ProxyRequestHandler.request_count = 0
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                f"http://127.0.0.1:{server.server_port}/v1/models",
            ],
            env=contaminated_env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        proxy.shutdown()
        proxy_thread.join()
        proxy.server_close()

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"ready":true}'
    assert ProxyRequestHandler.request_count == 0


def test_request_posts_json_body_and_header():
    server, thread = run_server()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--method",
                "POST",
                "--header",
                "Content-Type: application/json",
                "--data",
                '{"prompt":"hello"}',
                f"http://127.0.0.1:{server.server_port}/v1/completions",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"accepted":true}'
    assert RequestHandler.request_body == b'{"prompt":"hello"}'
    assert RequestHandler.content_type == "application/json"


def test_request_fails_on_non_success_status():
    server, thread = run_server()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                f"http://127.0.0.1:{server.server_port}/failure",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.returncode == 1
    assert "HTTP Error 503" in result.stderr
