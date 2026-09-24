# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare JSON-list and raw-base64 hidden-state response serialization."""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import time
import tracemalloc
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Measurement:
    median_ms: float
    median_peak_mib: float
    payload_bytes: int


def serialize_list(values: array, shape: tuple[int, int]) -> bytes:
    return json.dumps(
        {"dtype": "float32", "hidden_states": list(values)},
        separators=(",", ":"),
    ).encode()


def serialize_base64(values: array, shape: tuple[int, int]) -> bytes:
    return json.dumps(
        {
            "dtype": "float32",
            "hidden_states_shape": shape,
            "hidden_states_base64": base64.b64encode(
                memoryview(values).cast("B")
            ).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode()


def measure(
    serializer: Callable[[array, tuple[int, int]], bytes],
    values: array,
    shape: tuple[int, int],
    repeats: int,
) -> Measurement:
    times_ms = []
    peaks_mib = []
    payload = b""
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        payload = serializer(values, shape)
        times_ms.append((time.perf_counter() - started) * 1000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks_mib.append(peak / (1024 * 1024))
    return Measurement(
        median_ms=statistics.median(times_ms),
        median_peak_mib=statistics.median(peaks_mib),
        payload_bytes=len(payload),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--width", type=int, default=10240)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.rows < 1 or args.width < 1 or args.repeats < 1:
        parser.error("rows, width, and repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(0)
    shape = (args.rows, args.width)
    values = array("f", (rng.gauss(0.0, 1.0) for _ in range(args.rows * args.width)))
    old = measure(serialize_list, values, shape, args.repeats)
    new = measure(serialize_base64, values, shape, args.repeats)
    encoded = json.loads(serialize_base64(values, shape))
    assert base64.b64decode(encoded["hidden_states_base64"]) == values.tobytes()
    report = {
        "shape": list(shape),
        "dtype": "float32",
        "raw_bytes": len(values) * values.itemsize,
        "repeats": args.repeats,
        "roundtrip_verified": True,
        "before_list_json": asdict(old),
        "after_raw_base64_json": asdict(new),
        "ratios": {
            "payload": new.payload_bytes / old.payload_bytes,
            "encode_time": new.median_ms / old.median_ms,
            "peak_python_memory": new.median_peak_mib / old.median_peak_mib,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
