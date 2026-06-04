# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import torch

from vllm.distributed.eplb.eplb_utils import CpuGpuEvent


def test_wait_blocks_until_record():
    event = CpuGpuEvent()
    record_stream = torch.cuda.Stream()
    wait_stream = torch.cuda.Stream()
    wait_returned = threading.Event()

    def waiter():
        event.wait(stream=wait_stream)
        wait_returned.set()

    t = threading.Thread(target=waiter)
    t.start()

    time.sleep(0.05)
    assert not wait_returned.is_set(), "wait() returned before record() was called"

    event.record(stream=record_stream)
    t.join(timeout=5.0)

    assert not event._recorded.is_set()


def test_reuse_across_multiple_cycles():
    wrapper = CpuGpuEvent()
    record_stream = torch.cuda.Stream()
    wait_stream = torch.cuda.Stream()
    NUM_CYCLES = 8
    completed_cycles = []
    barriers = [threading.Barrier(2) for _ in range(NUM_CYCLES)]

    def waiter():
        for i in range(NUM_CYCLES):
            wrapper.wait(stream=wait_stream)
            completed_cycles.append(True)
            barriers[i].wait()

    t = threading.Thread(target=waiter)
    t.start()

    for i in range(NUM_CYCLES):
        wrapper.record(stream=record_stream)
        barriers[i].wait()

    t.join(timeout=10.0)
    assert len(completed_cycles) == NUM_CYCLES


def test_producer_consumer():
    """
    This test uses the CpuGpuEvent to synchronize reads and writes to/from a shared GPU
    tensor on multiple CPU threads.
    """
    worker_stream = torch.cuda.Stream()
    # Create a single element counter that will be shared between two threads
    buf = torch.zeros(1, device="cuda")
    NUM_ROUNDS = 5

    ready_cpu = [threading.Event() for _ in range(NUM_ROUNDS)]
    events = [CpuGpuEvent() for _ in range(NUM_ROUNDS)]
    errors: list[str] = []

    # For each round, the worker thread (writer) sets the counter in buf and waits for
    # the main thread to read it.
    def worker():
        for i in range(NUM_ROUNDS):
            if i > 0:
                events[i - 1].wait(stream=worker_stream)

            with torch.cuda.stream(worker_stream):
                buf.fill_(float(i + 1))

            worker_stream.synchronize()
            ready_cpu[i].set()

    t = threading.Thread(target=worker)
    t.start()

    for i in range(NUM_ROUNDS):
        ready_cpu[i].wait()
        snapshot = buf.clone()
        events[i].record()
        val = snapshot.item()
        if val != float(i + 1):
            errors.append(f"round {i}: expected {i + 1:.1f}, got {val:.1f}")

    t.join(timeout=10.0)
    assert not errors, f"Buffer ordering errors: {errors}"
