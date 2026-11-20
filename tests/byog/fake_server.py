# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic fake OpenAI-compatible vLLM HTTP server for CPU tests."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


def _extract_last_question(prompt: str) -> str:
    """Return the last `Question:` segment from a few-shot GSM8K prompt."""
    idx = prompt.rfind("Question:")
    if idx == -1:
        return prompt
    segment = prompt[idx + len("Question:") :]
    ans_idx = segment.find("Answer:")
    if ans_idx != -1:
        segment = segment[:ans_idx]
    return segment.strip()


def _solve_gsm8k_mini(prompt: str) -> str:
    """Deterministic canned responses matching `gsm8k_mini_test.jsonl`."""
    q = _extract_last_question(prompt).lower()
    if "janet's ducks" in q:
        return "Janet sells 16 - 3 - 4 = 9 eggs at $2 each. The answer is 18."
    if "blue fiber" in q:
        return "2 + 1 = 3 bolts. The answer is 3."
    if "12 apples" in q:
        return "12 - 5 = 7 apples. The answer is 7."
    if "3 boxes" in q and "8 pencils" in q:
        return "3 * 8 = 24 pencils. The answer is 24."
    if "60 miles per hour" in q:
        return "60 * 3 = 180 miles. The answer is 180."
    if "45 pages" in q and "15 pages" in q:
        return "45 + 15 = 60 pages. The answer is 60."
    return "The answer is 42."


@dataclass
class FakeServerState:
    """Observable counters and fault-injection knobs for `FakeVLLMServer`."""

    model_name: str = "fake-model"
    strict_model: bool = True
    allow_reset_cache: bool = False
    reset_error_code: int = 404
    latency_s: float = 0.0
    fail_after_n: int | None = None
    fail_on_calls: set[int] = field(default_factory=set)
    wrong_answer_mode: bool = False
    health_calls: int = 0
    reset_calls: int = 0
    completion_calls: int = 0
    chat_calls: int = 0
    bad_requests: int = 0
    requests: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total_inference_calls(self) -> int:
        with self._lock:
            return self.completion_calls + self.chat_calls


class FakeVLLMServer:
    """Lightweight in-process HTTP server emulating vLLM's OpenAI endpoints."""

    def __init__(self, state: FakeServerState | None = None):
        self.state = state or FakeServerState()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str = ""

    def start(self) -> str:
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any] | None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b""
                if not raw:
                    return {}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
                return parsed if isinstance(parsed, dict) else None

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/health":
                    with state._lock:
                        state.health_calls += 1
                    self._send_json(200, {"status": "ok"})
                    return
                if path == "/v1/models":
                    self._send_json(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": state.model_name, "object": "model"}],
                        },
                    )
                    return
                self._send_json(404, {"error": f"Unknown path: {path}"})

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path == "/reset_prefix_cache":
                    if not state.allow_reset_cache:
                        self._send_json(
                            state.reset_error_code,
                            {"error": "reset_prefix_cache disabled"},
                        )
                        return
                    with state._lock:
                        state.reset_calls += 1
                    self._send_json(200, {"status": "reset"})
                    return

                if path not in ("/v1/completions", "/v1/chat/completions"):
                    self._send_json(404, {"error": f"Unknown path: {path}"})
                    return

                payload = self._read_json()
                if payload is None:
                    with state._lock:
                        state.bad_requests += 1
                    self._send_json(400, {"error": "Invalid JSON body"})
                    return

                if payload.get("stream") is True:
                    with state._lock:
                        state.bad_requests += 1
                    self._send_json(
                        400,
                        {"error": "Streaming (stream=true) is not supported"},
                    )
                    return

                if state.strict_model:
                    req_model = payload.get("model")
                    if req_model != state.model_name:
                        with state._lock:
                            state.bad_requests += 1
                        self._send_json(
                            404,
                            {"error": f"Model '{req_model}' not found"},
                        )
                        return

                if path == "/v1/completions":
                    if "prompt" not in payload or not isinstance(
                        payload["prompt"], (str, list)
                    ):
                        with state._lock:
                            state.bad_requests += 1
                        self._send_json(400, {"error": "Missing valid 'prompt'"})
                        return
                    prompt_val = payload["prompt"]
                    prompt_str = (
                        prompt_val if isinstance(prompt_val, str) else str(prompt_val)
                    )
                else:
                    msgs = payload.get("messages")
                    if not isinstance(msgs, list) or not msgs:
                        with state._lock:
                            state.bad_requests += 1
                        self._send_json(400, {"error": "Missing valid 'messages'"})
                        return
                    prompt_str = "\n".join(
                        str(m.get("content", "")) for m in msgs if isinstance(m, dict)
                    )

                if state.latency_s > 0:
                    time.sleep(state.latency_s)

                with state._lock:
                    if path == "/v1/completions":
                        state.completion_calls += 1
                    else:
                        state.chat_calls += 1
                    call_num = state.completion_calls + state.chat_calls
                    state.requests.append({"path": path, "payload": payload})
                    should_fail = call_num in state.fail_on_calls or (
                        state.fail_after_n is not None and call_num > state.fail_after_n
                    )

                if should_fail:
                    self._send_json(
                        500, {"error": f"Injected fault on call #{call_num}"}
                    )
                    return

                text = (
                    "The answer is -999."
                    if state.wrong_answer_mode
                    else _solve_gsm8k_mini(prompt_str)
                )
                out_tokens = max(1, len(text.split()))
                if path == "/v1/completions":
                    self._send_json(
                        200,
                        {
                            "id": f"cmpl-fake-{call_num}",
                            "object": "text_completion",
                            "model": state.model_name,
                            "choices": [
                                {"index": 0, "text": text, "finish_reason": "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": max(1, len(prompt_str.split())),
                                "completion_tokens": out_tokens,
                                "total_tokens": max(1, len(prompt_str.split()))
                                + out_tokens,
                            },
                        },
                    )
                else:
                    self._send_json(
                        200,
                        {
                            "id": f"chatcmpl-fake-{call_num}",
                            "object": "chat.completion",
                            "model": state.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": max(1, len(prompt_str.split())),
                                "completion_tokens": out_tokens,
                                "total_tokens": max(1, len(prompt_str.split()))
                                + out_tokens,
                            },
                        },
                    )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_port
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeVLLMServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
