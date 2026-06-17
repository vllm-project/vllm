# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Writer subprocess for the OpenAI request/response jsonl logger.

Receives one record per ZMQ message on a PULL socket and appends it as a
single line to the configured output file. Runs as a separate process so
that fsync/disk pressure cannot block the API server's request path.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path

import zmq

from vllm.entrypoints.openai.request_log.proto import (
    FRAME_RECORD,
    FRAME_SHUTDOWN,
    FSYNC_EVERY_N_RECORDS,
    POLL_INTERVAL_MS,
)


def _install_signal_handlers(stop_flag: list[bool]) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        stop_flag[0] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def writer_main(
    socket_addr: str,
    output_path: str,
    ready_event: MpEvent,
) -> None:
    """Subprocess entry point.

    Args:
        socket_addr: ZMQ endpoint to bind a PULL socket on (e.g. ``ipc://...``).
        output_path: jsonl file to append records to.
        ready_event: set once the socket is bound and the file is opened, so
            the parent can know it is safe to start pushing.
    """

    # Decouple from any logging config the parent set up; print to stderr.
    def _log(msg: str) -> None:
        print(
            f"[vllm-request-log-writer pid={os.getpid()}] {msg}",
            file=sys.stderr,
            flush=True,
        )

    stop_flag = [False]
    _install_signal_handlers(stop_flag)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    try:
        pull.bind(socket_addr)
    except zmq.ZMQError as e:
        _log(f"failed to bind {socket_addr}: {e}")
        ready_event.set()  # unblock parent so it can move on / report
        return

    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    # Open in binary append mode with line buffering disabled; we manage
    # flush/fsync manually to balance durability and throughput.
    try:
        with open(output_path, "ab", buffering=0) as out:
            ready_event.set()
            _log(f"ready, writing to {output_path}")

            pending_records = 0
            try:
                while not stop_flag[0]:
                    try:
                        events = dict(poller.poll(timeout=POLL_INTERVAL_MS))
                    except zmq.error.ZMQError:
                        # interrupted by signal etc.
                        continue
                    if pull not in events:
                        # idle tick: flush whatever we have so far so
                        # ``tail -f`` works.
                        if pending_records:
                            out.flush()
                            pending_records = 0
                        continue
                    try:
                        parts = pull.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        continue
                    if len(parts) != 2:
                        _log(f"dropping malformed message ({len(parts)} frames)")
                        continue
                    tag, payload = parts
                    if tag == FRAME_SHUTDOWN:
                        stop_flag[0] = True
                        break
                    if tag != FRAME_RECORD:
                        _log(f"dropping unknown tag {tag!r}")
                        continue
                    try:
                        out.write(payload)
                        out.write(b"\n")
                        pending_records += 1
                        if pending_records >= FSYNC_EVERY_N_RECORDS:
                            out.flush()
                            pending_records = 0
                    except OSError as e:
                        _log(f"write failed: {e}")
            finally:
                # Drain whatever else is sitting in the queue so we
                # don't lose records that arrived right before shutdown.
                with contextlib.suppress(zmq.Again):
                    while True:
                        parts = pull.recv_multipart(flags=zmq.NOBLOCK)
                        if len(parts) != 2:
                            continue
                        tag, payload = parts
                        if tag == FRAME_RECORD:
                            out.write(payload)
                            out.write(b"\n")
                out.flush()
    finally:
        with contextlib.suppress(Exception):
            pull.close(linger=0)
        with contextlib.suppress(Exception):
            ctx.term()
        _log("exited cleanly")
