# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU serialization and cross-process ZMQ round-trip cost without a model.

Examples:
    python benchmarks/benchmark_engine_ipc.py --mode serialization
    python benchmarks/benchmark_engine_ipc.py --mode roundtrip --transport ipc
    python benchmarks/benchmark_engine_ipc.py --mode roundtrip --transport tcp

Payload sizes refer to one uint8 array, not the complete wire message. The
round trip includes encoding and decoding on both processes. This is a
synthetic, single-request-at-a-time benchmark, not serving throughput.

"""

import argparse
import json
import multiprocessing as mp
import platform
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from tempfile import TemporaryDirectory

import msgspec
import numpy as np
import zmq

from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


class Payload(msgspec.Struct):
    request_id: str
    data: np.ndarray


def echo(path: str, ready: Connection) -> None:
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(Payload)
    with (
        zmq.Context() as context,
        make_zmq_socket(context, path, zmq.DEALER, bind=True, linger=0) as sock,
    ):
        ready.send(sock.getsockopt_string(zmq.LAST_ENDPOINT))
        ready.close()
        while True:
            frames = sock.recv_multipart(copy=False)
            if len(frames) == 1 and len(frames[0]) == 0:
                return
            payload = decoder.decode(frames)
            sock.send_multipart(encoder.encode(payload), copy=False)


def measure(
    operation: Callable[[], Payload],
    expected: Payload,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    latencies = np.empty(iterations, dtype=np.int64)
    started = time.perf_counter_ns()
    for index in range(iterations):
        before = time.perf_counter_ns()
        result = operation()
        latencies[index] = time.perf_counter_ns() - before
    elapsed = (time.perf_counter_ns() - started) / 1e9
    assert result.request_id == expected.request_id
    np.testing.assert_array_equal(result.data, expected.data)
    percentiles = np.percentile(latencies, [50, 90, 99]) / 1000
    return {
        "p50_us": float(percentiles[0]),
        "p90_us": float(percentiles[1]),
        "p99_us": float(percentiles[2]),
        "operations_per_second": iterations / elapsed,
    }


def benchmark(args: argparse.Namespace, sock: zmq.Socket | None = None) -> None:
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(Payload)
    for size in args.sizes:
        payload = Payload("benchmark-request", np.arange(size, dtype=np.uint8))
        frames = encoder.encode(payload)

        def operation(payload: Payload = payload) -> Payload:
            encoded = encoder.encode(payload)
            if sock is None:
                return decoder.decode(encoded)
            sock.send_multipart(encoded, copy=False)
            return decoder.decode(sock.recv_multipart(copy=False))

        result = {
            "mode": args.mode,
            "transport": args.transport if sock is not None else None,
            "array_bytes": size,
            "wire_bytes": sum(memoryview(frame).nbytes for frame in frames),
            "frames": len(frames),
            "inline_arrays": int(len(frames) == 1),
            "aux_arrays": len(frames) - 1,
            "iterations": args.iterations,
            **measure(operation, payload, args.iterations, args.warmup),
        }
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["serialization", "roundtrip"], default="roundtrip"
    )
    parser.add_argument("--transport", choices=["ipc", "tcp"], default="ipc")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[128, 255, 256, 257, 4096, 1048576]
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(size < 0 for size in args.sizes):
        parser.error(
            "iterations must be positive; warmup and sizes must be nonnegative"
        )

    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pyzmq": zmq.__version__,
                "libzmq": zmq.zmq_version(),
                "msgspec": msgspec.__version__,
                "numpy": np.__version__,
                "zero_copy_threshold": MsgpackEncoder().size_threshold,
            }
        ),
        flush=True,
    )
    if args.mode == "serialization":
        benchmark(args)
        return

    ctx = mp.get_context("spawn")
    with TemporaryDirectory(prefix="vllm-ipc-bench-") as directory:
        path = (
            f"ipc://{directory}/echo.sock"
            if args.transport == "ipc"
            else "tcp://127.0.0.1:0"
        )
        parent, child = ctx.Pipe(duplex=False)
        worker = ctx.Process(target=echo, args=(path, child))
        worker.start()
        child.close()
        try:
            if not parent.poll(60):
                raise TimeoutError("Echo worker did not start within 60 seconds")
            endpoint = parent.recv()
            with (
                zmq.Context() as context,
                make_zmq_socket(
                    context, endpoint, zmq.DEALER, bind=False, linger=0
                ) as sock,
            ):
                sock.setsockopt(zmq.RCVTIMEO, 10000)
                sock.setsockopt(zmq.SNDTIMEO, 10000)
                benchmark(args, sock)
                sock.send(b"")
            worker.join(timeout=10)
            if worker.exitcode != 0:
                raise RuntimeError(f"Echo worker exited with code {worker.exitcode}")
        finally:
            parent.close()
            if worker.is_alive():
                worker.terminate()
                worker.join()


if __name__ == "__main__":
    main()
