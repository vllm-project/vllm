# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subscribe to native vLLM forward-pass metrics.

Start vLLM with FPM enabled before running this script:

    vllm serve MODEL --forward-pass-metrics-port 20380
    python forward_pass_metrics_subscriber.py --port 20380

For data parallel deployments, vLLM publishes rank N on BASE_PORT + N. Run one
subscriber per rank endpoint.

Example output (variances are across requests, in tokens squared):

    seq=7 worker='worker-0' dp=0 step=12.000ms scope=model_step_cuda
      scheduled prefill: reqs=2 new=8 kv=6 attention_var=1.00
      scheduled decode:  reqs=2 kv=64 kv_var=0.00
      queued prefill:    reqs=1 tokens=12 length_var=0.00
      queued decode:     reqs=0 kv=0 kv_var=0.00
"""

import argparse
import sys

import msgspec
import zmq

from vllm.v1.metrics.forward_pass_metrics import (
    FPM_TIMING_SCOPE_MODEL_STEP_CUDA,
    FPM_VERSION,
    ForwardPassMetrics,
)

POLL_TIMEOUT_MS = 1000


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
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    args.endpoint = args.endpoint or f"tcp://localhost:{args.port}"
    return args


def print_metrics(sequence: int, metrics: ForwardPassMetrics) -> None:
    prefix = f"seq={sequence} worker={metrics.worker_id!r} dp={metrics.dp_rank}"
    if metrics.wall_time == 0.0:
        print(f"{prefix} heartbeat")
        return

    scheduled = metrics.scheduled_requests
    queued = metrics.queued_requests
    print(
        f"{prefix} step={metrics.wall_time * 1000:.3f}ms scope={metrics.timing_scope}\n"
        f"  scheduled prefill: reqs={scheduled.num_prefill_requests} "
        f"new={scheduled.sum_prefill_tokens} kv={scheduled.sum_prefill_kv_tokens} "
        f"attention_var={scheduled.var_prefill_length:.2f}\n"
        f"  scheduled decode:  reqs={scheduled.num_decode_requests} "
        f"kv={scheduled.sum_decode_kv_tokens} "
        f"kv_var={scheduled.var_decode_kv_tokens:.2f}\n"
        f"  queued prefill:    reqs={queued.num_prefill_requests} "
        f"tokens={queued.sum_prefill_tokens} "
        f"length_var={queued.var_prefill_length:.2f}\n"
        f"  queued decode:     reqs={queued.num_decode_requests} "
        f"kv={queued.sum_decode_kv_tokens} kv_var={queued.var_decode_kv_tokens:.2f}"
    )


def main() -> None:
    args = parse_args()
    decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.endpoint)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    last_sequence: int | None = None

    print(f"Listening for forward-pass metrics on {args.endpoint}")
    try:
        while True:
            if not subscriber.poll(POLL_TIMEOUT_MS):
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
            if metrics.timing_scope != FPM_TIMING_SCOPE_MODEL_STEP_CUDA:
                print(
                    f"Ignoring timing scope {metrics.timing_scope!r}; "
                    f"expected {FPM_TIMING_SCOPE_MODEL_STEP_CUDA!r}",
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
