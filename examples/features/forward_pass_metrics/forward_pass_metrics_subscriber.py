# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subscribe to native vLLM forward-pass metrics.

Start vLLM with FPM enabled before running this script:

    vllm serve MODEL --forward-pass-metrics-port 20380
    python forward_pass_metrics_subscriber.py --port 20380

For data parallel deployments, vLLM publishes rank N on BASE_PORT + N. Run one
subscriber per rank endpoint.

By default, display the latest sample once per second, while receiving and
validating every message. Use --interval 0 to print every iteration. Displayed
sequence numbers can therefore skip without indicating a gap in the stream.

Example output (S=scheduled, Q=queued; variances are in tokens squared):

           seq step_ms   prefill:req/tok/kv/var      decode:req/kv/var
             7      12 S 2/8/6/1                     2/64/0
                      Q 1/12/-/0                    0/0/0

Each sample occupies at most two lines of at most 80 characters. Large values
use scientific notation; '-' means not applicable. The Q row is omitted when
the queue is empty. Worker identity and timing scope are printed separately
when they change.
"""

import argparse
import math
import sys
import time
from textwrap import fill

import msgspec
import zmq

from vllm.v1.metrics.forward_pass_metrics import (
    FPM_TIMING_SCOPE_MODEL_STEP_CUDA,
    FPM_VERSION,
    ForwardPassMetrics,
)

POLL_TIMEOUT_MS = 1000
DISPLAY_WIDTH = 80
HEADER = (
    f"{'seq':>10} {'step_ms':>7}   {'prefill:req/tok/kv/var':<27} decode:req/kv/var"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to native vLLM forward-pass metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    address = parser.add_mutually_exclusive_group()
    address.add_argument(
        "--endpoint",
        help="ZMQ endpoint for one vLLM data-parallel rank.",
    )
    address.add_argument(
        "--port",
        type=int,
        default=20380,
        help="Local FPM port (shorthand for --endpoint tcp://localhost:PORT).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between latest-sample displays; 0 prints every iteration.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval must be finite and nonnegative")
    args.endpoint = args.endpoint or f"tcp://localhost:{args.port}"
    return args


def compact(value: int | float) -> str:
    """Fit one number in six columns, preserving small integer counts exactly."""
    if isinstance(value, int) and len(str(value)) <= 6:
        return str(value)
    for precision in (3, 2, 1):
        text = f"{value:.{precision}g}".replace("e+", "e").replace("e-0", "e-")
        if len(text) <= 6:
            return text
    return text


def print_metrics(sequence: int, metrics: ForwardPassMetrics) -> None:
    if metrics.wall_time == 0.0:
        print(f"{sequence:>10} {'-':>7}   idle (heartbeat)", flush=True)
        return

    prefix = f"{sequence:>10} {compact(metrics.wall_time * 1000):>7}"
    for label, counts, prefill_kv in (
        (
            "S",
            metrics.scheduled_requests,
            compact(metrics.scheduled_requests.sum_prefill_kv_tokens),
        ),
        ("Q", metrics.queued_requests, "-"),
    ):
        if label == "Q" and not (
            counts.num_prefill_requests or counts.num_decode_requests
        ):
            continue
        prefill = "/".join(
            (
                compact(counts.num_prefill_requests),
                compact(counts.sum_prefill_tokens),
                prefill_kv,
                compact(counts.var_prefill_length),
            )
        )
        decode = "/".join(
            compact(value)
            for value in (
                counts.num_decode_requests,
                counts.sum_decode_kv_tokens,
                counts.var_decode_kv_tokens,
            )
        )
        print(f"{prefix} {label} {prefill:<27} {decode}", flush=True)
        prefix = " " * len(prefix)


class MetricsDisplay:
    """Coalesce display updates without skipping validation of received messages."""

    def __init__(self, interval: float):
        self.interval = interval
        self.next_print = 0.0
        self.pending: tuple[int, ForwardPassMetrics] | None = None
        self.identity: tuple[str, int, str] | None = None

    def update(self, sequence: int, metrics: ForwardPassMetrics) -> None:
        self.pending = sequence, metrics
        self.flush()

    def poll_timeout_ms(self) -> int:
        if self.pending is None:
            return POLL_TIMEOUT_MS
        remaining_ms = math.ceil((self.next_print - time.monotonic()) * 1000)
        return max(0, min(POLL_TIMEOUT_MS, remaining_ms))

    def flush(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.pending is None or (not force and now < self.next_print):
            return
        sequence, metrics = self.pending
        identity = metrics.worker_id, metrics.dp_rank, metrics.timing_scope
        if identity != self.identity:
            print(
                fill(
                    f"worker={ascii(metrics.worker_id)} dp={metrics.dp_rank} "
                    f"scope={metrics.timing_scope}",
                    width=DISPLAY_WIDTH,
                )
            )
            print(HEADER)
            self.identity = identity
        print_metrics(sequence, metrics)
        self.pending = None
        self.next_print = now + self.interval


def warn(message: str) -> None:
    print(fill(message, width=DISPLAY_WIDTH), file=sys.stderr)


def main() -> None:
    args = parse_args()
    decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.endpoint)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    last_sequence: int | None = None
    display = MetricsDisplay(args.interval)

    print(fill(f"FPM {args.endpoint} | Ctrl-C to stop", width=DISPLAY_WIDTH))
    print(
        "Every iteration (no display throttling)"
        if args.interval == 0
        else fill(
            f"Latest sample every {args.interval:g}s; "
            "all messages checked (display skips are normal).",
            width=DISPLAY_WIDTH,
        )
    )
    print(
        "S=scheduled; Q=queued (when nonempty); tok/kv=tokens; var=tokens^2; -=n/a",
        flush=True,
    )
    try:
        while True:
            display.flush()
            if not subscriber.poll(display.poll_timeout_ms()):
                continue

            frames = subscriber.recv_multipart()
            if len(frames) != 3:
                warn(f"Ignoring message with {len(frames)} frames; expected 3")
                continue

            topic, sequence_bytes, payload = frames
            if topic or len(sequence_bytes) != 8:
                warn("Ignoring malformed FPM envelope")
                continue

            sequence = int.from_bytes(sequence_bytes, "big")
            try:
                metrics = decoder.decode(payload)
            except msgspec.DecodeError as error:
                warn(f"Ignoring malformed FPM payload: {error}")
                continue

            if metrics.version != FPM_VERSION:
                warn(f"Ignoring FPM version {metrics.version}; expected {FPM_VERSION}")
                continue
            if metrics.timing_scope != FPM_TIMING_SCOPE_MODEL_STEP_CUDA:
                warn(
                    f"Ignoring timing scope {metrics.timing_scope!r}; "
                    f"expected {FPM_TIMING_SCOPE_MODEL_STEP_CUDA!r}"
                )
                continue
            if metrics.counter_id != sequence:
                warn(
                    f"Envelope sequence {sequence} does not match payload counter "
                    f"{metrics.counter_id}"
                )
            if last_sequence is not None and sequence != last_sequence + 1:
                warn(f"Sequence gap: previous={last_sequence}, current={sequence}")

            display.update(sequence, metrics)
            last_sequence = sequence
    except KeyboardInterrupt:
        display.flush(force=True)
        print("Stopped.")
    finally:
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
