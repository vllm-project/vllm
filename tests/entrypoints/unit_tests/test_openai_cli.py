# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest

from vllm.entrypoints.cli.openai import ChatCommand


@pytest.fixture
def chat_api():
    """Only the external model service is replaced; use the real CLI and SDK."""
    requests = []
    failures: dict[str, int | str] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "0.001")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200, b'{"data":[{"id":"test-model"}]}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            failure = failures.get(body["messages"][-1]["content"])
            if isinstance(failure, int):
                self.respond(
                    failure,
                    b'{"error":{"message":"diagnostic details","type":"test_error"}}',
                )
                return
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "reply"}}],
            }
            payload = "data: " + json.dumps(chunk) + "\n\n"
            if failure == "stream":
                payload += 'data: {"error":{"message":"stream interrupted"}}\n\n'
            else:
                payload += "data: [DONE]\n\n"
            self.respond(200, payload.encode(), "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, failures
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_chat(monkeypatch, url, text, *flags):
    parser = ChatCommand.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(["--url", url, *flags])
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    ChatCommand.cmd(args)


@pytest.mark.parametrize("flag", ["-i", "--interactive"])
def test_commands_clear_history_but_preserve_system_prompt(
    chat_api, monkeypatch, capsys, flag
):
    url, requests, _ = chat_api
    run_chat(
        monkeypatch,
        url,
        "/help\n/unknown\n\nfirst\n/clear\n//literal\n/exit\n",
        flag,
        "--system-prompt",
        "Be concise.",
    )
    assert len(requests) == 2
    assert requests[-1]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "/literal"},
    ]
    output = capsys.readouterr().out
    assert "Unknown command" in output
    assert "Conversation cleared" in output


@pytest.mark.parametrize("failure,attempts", [(400, 1), (500, 3), ("stream", 1)])
def test_api_errors_keep_completed_history_and_full_diagnostics(
    chat_api, monkeypatch, capsys, failure, attempts
):
    url, requests, failures = chat_api
    failures["reject"] = failure
    run_chat(monkeypatch, url, "first\nreject\ncontinue\n/exit\n", "-i")
    assert len(requests) == attempts + 2
    assert requests[-1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "continue"},
    ]
    error = capsys.readouterr().err
    assert "Traceback" in error
    assert "Previous conversation is unchanged" in error
    assert (
        "stream interrupted" in error
        if failure == "stream"
        else (f"Error code: {failure}" in error and "diagnostic details" in error)
    )


def test_default_mode_preserves_literal_commands_and_propagates_errors(
    chat_api, monkeypatch, capsys
):
    url, requests, failures = chat_api
    failures["reject"] = 400
    with pytest.raises(openai.BadRequestError, match="diagnostic details"):
        run_chat(monkeypatch, url, "/clear\nreject\n")
    assert requests[0]["messages"] == [{"role": "user", "content": "/clear"}]
    assert "Conversation cleared" not in capsys.readouterr().out


def test_interactive_rejects_quick_mode(capsys):
    parser = ChatCommand.add_cli_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "--quick", "hello"])
    assert "not allowed" in capsys.readouterr().err
