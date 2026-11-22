# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for multiprocessing queue handling in benchmark_serving_multi_turn."""

from __future__ import annotations

import asyncio
import importlib
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

_MULTI_TURN_DIR = str(Path(__file__).resolve().parents[2] / "benchmarks" / "multi_turn")
if _MULTI_TURN_DIR not in sys.path:
    sys.path.insert(0, _MULTI_TURN_DIR)

bmt = importlib.import_module("benchmark_serving_multi_turn")


class _SimpleTokenizer:
    """Picklable minimal tokenizer compatible with `send_turn`."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        tokens = text.split()
        return SimpleNamespace(input_ids=list(range(max(1, len(tokens)))))


def _make_request_stats(client_id: int, conv_id: str, idx: int) -> bmt.RequestStats:
    return bmt.RequestStats(
        ttft_ms=1.0,
        tpot_ms=0.5,
        latency_ms=2.0,
        start_time_ms=float(idx),
        input_num_turns=1,
        input_num_tokens=4,
        output_num_tokens=4,
        output_num_chunks=2,
        output_num_first_chunk_tokens=1,
        approx_cached_percent=0.0,
        conversation_id=conv_id,
        client_id=client_id,
    )


def _make_args(
    *,
    num_clients: int,
    url: str = "http://127.0.0.1:1",
    max_num_requests: int | None = None,
    early_stop: bool = False,
    timeout_sec: int = 10,
) -> tuple[bmt.ClientArgs, bmt.RequestArgs, bmt.BenchmarkArgs]:
    client_args = bmt.ClientArgs(
        seed=0,
        max_num_requests=max_num_requests,
        skip_first_turn=False,
        max_turns=None,
        max_active_conversations=1,
        verbose=False,
        print_content=False,
        verify_output=False,
        conversation_sampling=bmt.ConversationSampling.ROUND_ROBIN,
        request_rate=0.0,
        max_retries=0,
    )
    req_args = bmt.RequestArgs(
        chat_url=f"{url}/v1/chat/completions",
        model="test-model",
        stream=True,
        limit_min_tokens=-1,
        limit_max_tokens=-1,
        timeout_sec=timeout_sec,
        send_conversation_id=False,
        headers={},
    )
    bench_args = bmt.BenchmarkArgs(
        url=url,
        num_clients=num_clients,
        early_stop=early_stop,
    )
    return client_args, req_args, bench_args


# Top-level worker functions so multiprocessing works with both `fork` and `spawn`.


def _worker_delayed_result_flush(
    client_id: int,
    tokenizer,
    client_args,
    req_args,
    stop_event,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    """Reproduces Bug (a): conv_queue TERM_SIGNAL arrives before result_queue."""
    results_per_client = client_args.max_num_requests or 4
    conv_id, messages = task_queue.get()
    if conv_id is not bmt.TERM_SIGNAL:
        conv_queue.put((conv_id, messages))
    # Signal completion on conv_queue first, then flush results slightly later
    # to deterministically model OS pipe / feeder-thread skew between queues.
    conv_queue.put((bmt.TERM_SIGNAL, bmt.TERM_SIGNAL))
    time.sleep(0.05)
    for i in range(results_per_client):
        result_queue.put(_make_request_stats(client_id, str(conv_id), i))
    if client_args.verify_output:
        result_queue.put(bmt.TERM_SIGNAL)


def _worker_large_result_burst(
    client_id: int,
    tokenizer,
    client_args,
    req_args,
    stop_event,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    """Emits >64 KiB of results per worker to ensure draining prevents pipe deadlock."""
    num_results = client_args.max_num_requests or 400
    conv_id, messages = task_queue.get()
    for i in range(num_results):
        result_queue.put(_make_request_stats(client_id, f"{conv_id}_{'x' * 256}", i))
    result_queue.put(bmt.TERM_SIGNAL)
    if conv_id is not bmt.TERM_SIGNAL:
        conv_queue.put((conv_id, messages))
    conv_queue.put((bmt.TERM_SIGNAL, bmt.TERM_SIGNAL))


def _worker_exit_early_leave_task_queue_full(
    client_id: int,
    tokenizer,
    client_args,
    req_args,
    stop_event,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    """Reproduces Bug (b): workers exit early while parent's task_queue is full."""
    result_queue.put(_make_request_stats(client_id, "early", 0))
    result_queue.put(bmt.TERM_SIGNAL)
    conv_queue.put((bmt.TERM_SIGNAL, bmt.TERM_SIGNAL))


def _worker_crash_without_sentinel(
    client_id: int,
    tokenizer,
    client_args,
    req_args,
    stop_event,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    """Client 0 crashes abnormally; other clients finish normally."""
    if client_id == 0:
        os._exit(2)
    conv_id, messages = task_queue.get()
    result_queue.put(_make_request_stats(client_id, str(conv_id), 0))
    result_queue.put(bmt.TERM_SIGNAL)
    if conv_id is not bmt.TERM_SIGNAL:
        conv_queue.put((conv_id, messages))
    conv_queue.put((bmt.TERM_SIGNAL, bmt.TERM_SIGNAL))


@pytest.mark.parametrize("send_result_sentinel", [True, False])
def test_main_mp_drains_all_results_when_conv_queue_beats_result_queue(
    monkeypatch: pytest.MonkeyPatch, send_result_sentinel: bool
) -> None:
    """Regression test for Bug (a): draining with `not result_queue.empty()`
    before `client.join()` drops results still in-flight in worker feeder threads."""
    num_clients = 8
    results_per_client = 8
    input_conv = {
        f"conv_{i}": [{"role": "user", "content": "hi"}] for i in range(num_clients)
    }
    client_args, req_args, bench_args = _make_args(
        num_clients=num_clients, max_num_requests=results_per_client
    )
    client_args = client_args._replace(verify_output=send_result_sentinel)
    monkeypatch.setattr(bmt, "worker_function", _worker_delayed_result_flush)

    output_conv, client_metrics = asyncio.run(
        asyncio.wait_for(
            bmt.main_mp(
                client_args, req_args, bench_args, _SimpleTokenizer(), input_conv
            ),
            timeout=15.0,
        )
    )

    assert len(output_conv) == num_clients
    assert len(client_metrics) == num_clients * results_per_client


def test_main_mp_drains_large_result_burst_without_pipe_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draining while workers flush prevents OS pipe-buffer deadlock on large bursts."""
    num_clients = 4
    results_per_client = 400
    input_conv = {
        f"conv_{i}": [{"role": "user", "content": "hi"}] for i in range(num_clients)
    }
    client_args, req_args, bench_args = _make_args(
        num_clients=num_clients, max_num_requests=results_per_client
    )
    monkeypatch.setattr(bmt, "worker_function", _worker_large_result_burst)

    output_conv, client_metrics = asyncio.run(
        asyncio.wait_for(
            bmt.main_mp(
                client_args, req_args, bench_args, _SimpleTokenizer(), input_conv
            ),
            timeout=20.0,
        )
    )

    assert len(output_conv) == num_clients
    assert len(client_metrics) == num_clients * results_per_client


def test_main_mp_teardown_does_not_deadlock_on_full_task_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for Bug (b): unconsumed items backed up in `task_queue`'s
    feeder thread must not deadlock parent teardown in `join_thread()`."""
    num_clients = 2
    # >2 MiB of unread conversations guarantees task_queue's OS pipe is full
    # and the parent's QueueFeederThread still has buffered items on teardown.
    large_payload = "x" * 16384
    input_conv = {
        f"conv_{i}": [{"role": "user", "content": large_payload}] for i in range(160)
    }
    client_args, req_args, bench_args = _make_args(
        num_clients=num_clients, max_num_requests=1, early_stop=True
    )
    monkeypatch.setattr(
        bmt, "worker_function", _worker_exit_early_leave_task_queue_full
    )

    _, client_metrics = asyncio.run(
        asyncio.wait_for(
            bmt.main_mp(
                client_args, req_args, bench_args, _SimpleTokenizer(), input_conv
            ),
            timeout=15.0,
        )
    )

    assert len(client_metrics) == num_clients


def test_main_mp_recovers_if_worker_crashes_without_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker crashing abnormally must not hang `main_mp` forever."""
    num_clients = 3
    input_conv = {
        f"conv_{i}": [{"role": "user", "content": "hi"}] for i in range(num_clients)
    }
    client_args, req_args, bench_args = _make_args(
        num_clients=num_clients, timeout_sec=5
    )
    monkeypatch.setattr(bmt, "worker_function", _worker_crash_without_sentinel)

    output_conv, client_metrics = asyncio.run(
        asyncio.wait_for(
            bmt.main_mp(
                client_args, req_args, bench_args, _SimpleTokenizer(), input_conv
            ),
            timeout=15.0,
        )
    )

    assert len(output_conv) == num_clients - 1
    assert len(client_metrics) == num_clients - 1


def test_main_mp_end_to_end_with_local_streaming_server() -> None:
    """Exercises real `worker_function` + `client_main` against a local SSE server."""

    class _SSEHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len:
                self.rfile.read(content_len)
            body_parts = [
                f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                for chunk in ("hello ", "world")
            ]
            body_parts.append("data: [DONE]\n\n")
            body = "".join(body_parts).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        num_clients = 4
        num_convs = 8
        # 2 user turns + 2 assistant slots per conversation = 2 requests/conv.
        input_conv = {
            f"conv_{i}": [
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": "placeholder one"},
                {"role": "user", "content": "turn two"},
                {"role": "assistant", "content": "placeholder two"},
            ]
            for i in range(num_convs)
        }
        client_args, req_args, bench_args = _make_args(
            num_clients=num_clients,
            url=f"http://127.0.0.1:{port}",
            early_stop=False,
        )
        output_conv, client_metrics = asyncio.run(
            asyncio.wait_for(
                bmt.main_mp(
                    client_args,
                    req_args,
                    bench_args,
                    _SimpleTokenizer(),
                    input_conv,
                ),
                timeout=20.0,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(output_conv) == 8
    assert len(client_metrics) == 16
