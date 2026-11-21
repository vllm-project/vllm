# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol-only local fixture; it is not GPU performance evidence."""

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from benchmarks.adaptive_prefill.compare import compare
from benchmarks.adaptive_prefill.serve import run


class Handler(BaseHTTPRequestHandler):
    mode = "single"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "truncated":
            return
        count = payload["max_tokens"]
        chunks = [count] if self.mode == "multi" else [1] * count
        for n in chunks:
            data = {
                "choices": [
                    {
                        "text": "x" * n,
                        "logprobs": {"tokens": ["x"] * n},
                        "finish_reason": None,
                    }
                ]
            }
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
        data = {
            "choices": [{"text": "", "finish_reason": "length"}],
            "usage": {"completion_tokens": count},
        }
        self.wfile.write(
            ("data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n").encode()
        )


class TestStreamingBenchmark(unittest.TestCase):
    def setUp(self):
        Handler.mode = "single"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.args = SimpleNamespace(
            url=f"http://127.0.0.1:{self.server.server_port}",
            model="fixture",
            label="protocol-test-not-gpu",
            seed=1,
            vocab_size=100,
            long_tokens=8,
            short_tokens=4,
            long_output=2,
            short_output=3,
            long_requests=1,
            short_requests=2,
            long_interval=0.0,
            short_interval=0.0,
            short_delay=0.0,
            warmup_requests=0,
            timeout=5,
            output="unused",
        )

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_single_token_chunks_and_compare(self):
        result = asyncio.run(run(self.args))
        self.assertEqual(result["failures"], 0)
        self.assertIsNotNone(result["decode_token_gap_p99_ms"])
        self.assertTrue(result["all_long_requests_completed"])
        self.assertEqual(
            compare(result, result)["output_tokens_per_s"]["change_percent"], 0
        )

    def test_multitoken_chunks_are_not_token_latency(self):
        Handler.mode = "multi"
        result = asyncio.run(run(self.args))
        self.assertIsNone(result["decode_token_gap_p99_ms"])

    def test_truncated_stream_fails_closed(self):
        Handler.mode = "truncated"
        result = asyncio.run(run(self.args))
        self.assertEqual(result["failures"], 3)
        with self.assertRaises(ValueError):
            compare(result, result)


if __name__ == "__main__":
    unittest.main()
