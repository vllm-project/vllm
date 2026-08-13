# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from contextlib import contextmanager

arrival_time = time.perf_counter()
previous_timestamp = time.perf_counter()


def set_arrival_time(arrival: float | None = None):
    global arrival_time, previous_timestamp

    arrival_time = arrival or time.perf_counter()
    previous_timestamp = arrival_time


def debug_spend_time(name: str):
    global arrival_time, previous_timestamp
    now = time.perf_counter()
    print(f"{name}: {(now - previous_timestamp) * 1000} ms, "
          f"{(now - arrival_time) * 1000} ms")
    previous_timestamp = now


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f} 秒")