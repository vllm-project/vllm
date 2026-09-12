# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Best-effort, container-visible NVIDIA device samples for command timelines."""

from __future__ import annotations

import csv
import io
import math
import os
import secrets
import signal
import subprocess
import sys
import threading
import time

from ci_otel import Span, export_spans, record_spans

INTERVAL_SECONDS = 1.0
BATCH_SECONDS = 30
MAX_DEVICES = 16
QUERY_TIMEOUT_SECONDS = 2
QUERY = "index,uuid,name,utilization.gpu,memory.used,memory.total,mig.mode.current"


def _number(value: str, scale: int = 1, maximum: int | None = None) -> int | None:
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return None
        if maximum is not None and number > maximum:
            return None
        return round(number * scale)
    except ValueError:
        return None


def parse_samples(output: str, timestamp_ns: int) -> list[dict]:
    events = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 7 or len(events) >= MAX_DEVICES:
            continue
        index, uuid, name, util, used, total, mig = (value.strip() for value in row)
        if not index.isdigit() or not uuid.startswith("GPU-"):
            continue
        attributes: dict[str, str | int | bool] = {
            "gpu.uuid": uuid,
            "gpu.index": int(index),
            "gpu.name": name,
        }
        # Parent-device memory on a MIG GPU is not the job's allocated memory.
        if mig.lower() == "enabled":
            attributes["gpu.status"] = "unsupported_mig"
        else:
            for key, value in (
                ("gpu.utilization", _number(util, maximum=100)),
                ("gpu.memory.used", _number(used, 1024**2)),
                ("gpu.memory.total", _number(total, 1024**2)),
            ):
                if value is not None:
                    attributes[key] = value
        events.append(
            {"time_ns": timestamp_ns, "name": "ci.gpu.sample", "attributes": attributes}
        )
    return events


def sample_span(events: list[dict]) -> Span:
    return Span(
        trace_id=os.environ["CI_INFRA_TRACE_ID"],
        span_id=secrets.token_hex(8),
        parent_span_id=os.environ["CI_INFRA_COMMAND_SPAN_ID"],
        name="ci.gpu.samples",
        start_ns=events[0]["time_ns"],
        end_ns=events[-1]["time_ns"],
        attributes={
            "ci.span.kind": "gpu-samples",
            "gpu.sample.interval_ms": int(INTERVAL_SECONDS * 1000),
            "gpu.scope": "container-visible-devices",
        },
        events=events,
    )


def collect(parent_pid: int, stop: threading.Event) -> None:
    events: list[dict] = []
    batch_started = time.monotonic()
    failures = 0
    try:
        while not stop.is_set() and os.getppid() == parent_pid:
            started = time.monotonic()
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={QUERY}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=QUERY_TIMEOUT_SECONDS,
                    check=True,
                )
                if stop.is_set():
                    break
                samples = parse_samples(result.stdout, time.time_ns())
                if not samples:
                    break
                events.extend(samples)
                failures = 0
            except (OSError, subprocess.SubprocessError):
                failures += 1
                if failures >= 3:
                    print(
                        "CI GPU sampling stopped: device query unavailable",
                        file=sys.stderr,
                    )
                    break
            if events and time.monotonic() - batch_started >= BATCH_SECONDS:
                # Bounded memory and upload time even if the receiver is down.
                # Dropped batches appear as gaps, never as zero utilization.
                export_spans([sample_span(events)], timeout_seconds=2)
                events = []
                batch_started = time.monotonic()
            stop.wait(max(0, INTERVAL_SECONDS - (time.monotonic() - started)))
    finally:
        # The shell joins this process before the ordinary job-end spool flush.
        if events:
            record_spans([sample_span(events)])


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        collect(int(sys.argv[1]), stop)
    except Exception:
        print("CI GPU sampling skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
