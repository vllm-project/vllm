# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the client wire contract and failure evidence without a GPU."""

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import benchmark


class BenchmarkTest(unittest.TestCase):
    def run_client(self, mode="ok"):
        pair = json.loads(Path(benchmark.__file__).with_name("pair.json").read_text())
        captured = []

        def respond(request, **kwargs):
            body = json.loads(request.data)
            captured.append(body)
            if mode == "http":
                raise urllib.error.HTTPError(
                    request.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad body")
                )
            events = [
                {"choices": [{"text": "hello", "finish_reason": None}]},
                {"choices": [{"text": " world", "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(body["prompt"]),
                        "completion_tokens": 3,
                    },
                },
            ]
            if mode == "error":
                events = [{"error": {"message": "stream error"}}]
            wire = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
            if mode != "cut":
                wire += "data: [DONE]\n\n"
            return io.BytesIO(wire.encode())

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.jsonl"
            argv = [
                "benchmark.py",
                "--base-url",
                "http://localhost:8000",
                "--label",
                "test",
                "--output",
                str(output),
            ]
            with (
                patch("sys.argv", argv),
                patch("urllib.request.urlopen", side_effect=respond),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if mode == "ok":
                    benchmark.main()
                    with self.assertRaises(FileExistsError):
                        benchmark.main()
                else:
                    with self.assertRaises(RuntimeError):
                        benchmark.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        return pair, captured, rows

    def test_frozen_requests_usage_only_event_and_measurement(self):
        pair, bodies, rows = self.run_client()
        self.assertEqual(len(bodies), 2)
        for task, body, row in zip(pair["tasks"], bodies, rows):
            self.assertEqual(
                body,
                {
                    "model": "flashnext",
                    "prompt": task["prompt_token_ids"],
                    "max_tokens": 2048,
                    "temperature": 0,
                    "top_p": 1,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "seed": 0,
                },
            )
            self.assertEqual(row["content"], "hello world")
            self.assertTrue(row["sse_done"])
            self.assertEqual(row["role"], task["role"])
            self.assertEqual(
                row["decode_tok_s"], 2 / (row["last_token_s"] - row["first_token_s"])
            )

    def test_failure_is_saved_and_second_request_is_not_sent(self):
        for mode in ("http", "error", "cut"):
            with self.subTest(mode=mode):
                _, bodies, rows = self.run_client(mode)
                self.assertEqual(len(bodies), 1)
                self.assertEqual(len(rows), 1)
                self.assertIn("error", rows[0])
                if mode == "http":
                    self.assertEqual(rows[0]["error_body"], "bad body")


if __name__ == "__main__":
    unittest.main()
