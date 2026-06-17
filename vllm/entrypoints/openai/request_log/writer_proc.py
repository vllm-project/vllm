# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Writer subprocess for the OpenAI request/response jsonl logger.

Receives one record per ZMQ message on a PULL socket and appends it as a
single line to the configured output file. Runs as a separate process so
that fsync/disk pressure cannot block the API server's request path.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path

import msgspec
import zmq

from vllm.entrypoints.openai.request_log.proto import (
    FRAME_RECORD,
    FRAME_SHUTDOWN,
    FSYNC_EVERY_N_RECORDS,
    POLL_INTERVAL_MS,
)
from vllm.entrypoints.openai.request_log.rotator import (
    JsonlRotator,
    parse_duration,
)

# Reusable msgpack decoder for the hot-path payload conversion.
_MSGPACK_DECODER = msgspec.msgpack.Decoder()


def _record_payload_to_jsonl(payload: bytes, log) -> bytes | None:
    """Decode an msgpack record and re-encode as a single jsonl line.

    Doing the JSON serialization on the writer side keeps the (often
    expensive) ``json.dumps`` call off the API server's event loop.
    Returns ``None`` if the payload cannot be decoded, in which case the
    record is dropped with a log message.
    """
    try:
        record = _MSGPACK_DECODER.decode(payload)
    except (msgspec.DecodeError, ValueError) as e:
        log(f"failed to decode record: {e}")
        return None
    try:
        return (
            json.dumps(record, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
        )
    except (TypeError, ValueError) as e:
        log(f"failed to json-encode record: {e}")
        return None


def _install_signal_handlers(stop_flag: list[bool]) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        stop_flag[0] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def writer_main(
    socket_addr: str,
    output_path: str,
    ready_event: MpEvent,
    max_bytes: int = 0,
    rotate_interval: str | None = None,
    backup_count: int = 0,
) -> None:
    """Subprocess entry point.

    Args:
        socket_addr: ZMQ endpoint to bind a PULL socket on (e.g. ``ipc://...``).
        output_path: jsonl file to append records to.
        ready_event: set once the socket is bound and the file is opened, so
            the parent can know it is safe to start pushing.
        max_bytes: rotate the output file once it grows past this many bytes;
            0 disables size-based rotation.
        rotate_interval: optional duration string (``30s`` / ``5m`` / ``1h`` /
            ``1d``) or seconds-as-int; rotates on this schedule when set.
        backup_count: keep at most this many rotated files; 0 keeps all.
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

    interval_seconds = parse_duration(rotate_interval)
    rotator = JsonlRotator(
        output_path,
        max_bytes=max_bytes,
        interval_seconds=interval_seconds,
        backup_count=backup_count,
    )
    rotation_desc = []
    if max_bytes:
        rotation_desc.append(f"max_bytes={max_bytes}")
    if interval_seconds:
        rotation_desc.append(f"interval={interval_seconds:.0f}s")
    if backup_count:
        rotation_desc.append(f"backup_count={backup_count}")
    if rotation_desc:
        _log(f"rotation: {', '.join(rotation_desc)}")

    # Append-only with optional rotation; ``rotator`` is a file-like
    # with ``write(bytes)`` / ``flush()`` / ``close()``.
    try:
        ready_event.set()
        _log(f"ready, writing to {output_path}")
        try:
            pending_records = 0
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
                        rotator.flush()
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
                line = _record_payload_to_jsonl(payload, _log)
                if line is None:
                    continue
                try:
                    rotator.write(line)
                    pending_records += 1
                    if pending_records >= FSYNC_EVERY_N_RECORDS:
                        rotator.flush()
                        pending_records = 0
                except OSError as e:
                    _log(f"write failed: {e}")
        finally:
            # Drain whatever else is sitting in the queue so we don't
            # lose records that arrived right before shutdown.
            with contextlib.suppress(zmq.Again):
                while True:
                    parts = pull.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) != 2:
                        continue
                    tag, payload = parts
                    if tag == FRAME_RECORD:
                        line = _record_payload_to_jsonl(payload, _log)
                        if line is not None:
                            rotator.write(line)
            rotator.close()
    finally:
        with contextlib.suppress(Exception):
            pull.close(linger=0)
        with contextlib.suppress(Exception):
            ctx.term()
        _log("exited cleanly")
