# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subscribe to native vLLM forward-pass metrics.

Start vLLM with FPM enabled before running this script:

    vllm serve MODEL --forward-pass-metrics-port 20380
    python forward_pass_metrics_subscriber.py --endpoint tcp://localhost:20380

For data parallel deployments, vLLM publishes rank N on BASE_PORT + N. Run one
subscriber per rank endpoint.
"""

import argparse
import sys

import msgspec
import zmq

FPM_VERSION = 1
FPM_TIMING_SCOPE = "execute_model_cuda"


# Types copied from vllm.v1.metrics.forward_pass_metrics to demonstrate the
# external wire contract without importing vLLM internals.
class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    version: int = FPM_VERSION
    worker_id: str = ""
    dp_rank: int = 0
    counter_id: int = 0
    timing_scope: str = FPM_TIMING_SCOPE
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to native vLLM forward-pass metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://localhost:20380",
        help="ZMQ endpoint for one vLLM data-parallel rank.",
    )
    parser.add_argument(
        "--poll-timeout-ms",
        type=int,
        default=1000,
        help="How long to poll before checking for Ctrl+C.",
    )
    return parser.parse_args()


def print_metrics(sequence: int, metrics: ForwardPassMetrics) -> None:
    prefix = (
        f"sequence={sequence} worker_id={metrics.worker_id!r} dp_rank={metrics.dp_rank}"
    )
    if metrics.wall_time == 0.0:
        print(f"{prefix} heartbeat")
        return

    scheduled = msgspec.to_builtins(metrics.scheduled_requests)
    queued = msgspec.to_builtins(metrics.queued_requests)
    print(
        f"{prefix} wall_time_ms={metrics.wall_time * 1000:.3f} "
        f"scheduled={scheduled} queued={queued}"
    )


def main() -> None:
    args = parse_args()
    if args.poll_timeout_ms <= 0:
        raise ValueError("--poll-timeout-ms must be positive")

    decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.endpoint)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    last_sequence: int | None = None

    print(f"Listening for forward-pass metrics on {args.endpoint}")
    try:
        while True:
            if not subscriber.poll(args.poll_timeout_ms):
                continue

            frames = subscriber.recv_multipart()
            if len(frames) != 3:
                print(
                    f"Ignoring message with {len(frames)} frames; expected 3",
                    file=sys.stderr,
                )
                continue

            topic, sequence_bytes, payload = frames
            if topic or len(sequence_bytes) != 8:
                print("Ignoring malformed FPM envelope", file=sys.stderr)
                continue

            sequence = int.from_bytes(sequence_bytes, "big")
            try:
                metrics = decoder.decode(payload)
            except msgspec.DecodeError as error:
                print(f"Ignoring malformed FPM payload: {error}", file=sys.stderr)
                continue

            if metrics.version != FPM_VERSION:
                print(
                    f"Ignoring FPM version {metrics.version}; expected {FPM_VERSION}",
                    file=sys.stderr,
                )
                continue
            if metrics.timing_scope != FPM_TIMING_SCOPE:
                print(
                    f"Ignoring timing scope {metrics.timing_scope!r}; "
                    f"expected {FPM_TIMING_SCOPE!r}",
                    file=sys.stderr,
                )
                continue
            if metrics.counter_id != sequence:
                print(
                    f"Envelope sequence {sequence} does not match payload counter "
                    f"{metrics.counter_id}",
                    file=sys.stderr,
                )
            if last_sequence is not None and sequence != last_sequence + 1:
                print(
                    f"Sequence gap: previous={last_sequence}, current={sequence}",
                    file=sys.stderr,
                )

            print_metrics(sequence, metrics)
            last_sequence = sequence
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
