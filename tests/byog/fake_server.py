# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal strict fake OpenAI-compatible vLLM server for CPU tests."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


def _solve_mini_gsm8k(prompt: str) -> str:
    idx = prompt.rfind("Question:")
    q = (prompt[idx:] if idx != -1 else prompt).lower()
    answers = [
        ("janet's ducks", "18"),
        ("blue fiber", "3"),
        ("12 apples", "7"),
        ("3 boxes", "24"),
        ("60 miles per hour", "180"),
        ("45 pages", "60"),
    ]
    for needle, ans in answers:
        if needle in q:
            return f"The answer is {ans}."
    return "The answer is 42."


@dataclass
class FakeServerState:
    model_name: str = "fake-model"
    strict_model: bool = True
    allow_reset_cache: bool = False
    reset_error_code: int = 404
    fail_on_calls: set[int] = field(default_factory=set)
    wrong_answer_mode: bool = False
    calls: int = 0
    reset_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


class FakeVLLMServer:
    """In-process fake vLLM server validating JSON, model, and stream flags."""

    def __init__(self, state: FakeServerState | None = None):
        self.state = state or FakeServerState()
        self._srv: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def start(self) -> str:
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _json(self, code: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/health":
                    self._json(200, {"status": "ok"})
                elif path == "/v1/models":
                    self._json(200, {"data": [{"id": state.model_name}]})
                else:
                    self._json(404, {"error": path})

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path == "/reset_prefix_cache":
                    if not state.allow_reset_cache:
                        self._json(state.reset_error_code, {"error": "disabled"})
                        return
                    with state._lock:
                        state.reset_calls += 1
                    self._json(200, {"status": "reset"})
                    return
                if path != "/v1/completions":
                    self._json(404, {"error": path})
                    return

                nbytes = int(self.headers.get("Content-Length", "0"))
                try:
                    payload = json.loads(self.rfile.read(nbytes).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json(400, {"error": "invalid json"})
                    return
                if not isinstance(payload, dict) or payload.get("stream") is True:
                    self._json(400, {"error": "unsupported request"})
                    return
                if state.strict_model and payload.get("model") != state.model_name:
                    self._json(404, {"error": "unknown model"})
                    return
                if not isinstance(payload.get("prompt"), str):
                    self._json(400, {"error": "missing prompt"})
                    return

                with state._lock:
                    state.calls += 1
                    n = state.calls
                if n in state.fail_on_calls:
                    self._json(500, {"error": f"injected fault #{n}"})
                    return

                text = (
                    "The answer is -999."
                    if state.wrong_answer_mode
                    else _solve_mini_gsm8k(payload["prompt"])
                )
                self._json(
                    200,
                    {
                        "id": f"cmpl-{n}",
                        "model": state.model_name,
                        "choices": [{"index": 0, "text": text}],
                        "usage": {"completion_tokens": len(text.split())},
                    },
                )

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._srv.server_port}"
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeVLLMServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
