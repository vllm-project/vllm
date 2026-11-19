# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU planning interference with a foreground loop, not model TPOT.

Run with `python -m benchmarks.benchmark_eplb_policy --device cpu --output cpu.json`.
The optional CUDA/NPU foreground submits and synchronizes one small tensor op.
Process startup and reference placements are prepared outside the timed region.
"""

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.eplb.policy import DefaultEplbPolicy, EplbStats
from vllm.distributed.eplb.policy_process import EplbPolicyProcess


def measure(args, mode, snapshots, device):
    startup = time.perf_counter()
    process = EplbPolicyProcess(DefaultEplbPolicy()) if mode == "process" else None
    startup_ms = (time.perf_counter() - startup) * 1000
    policy = process or DefaultEplbPolicy()
    if mode != "idle":
        stats, old_map, expected = snapshots[0]
        assert torch.equal(policy.rebalance_experts(stats, old_map), expected)
    begin = threading.Event()
    active = threading.Event()
    stop = threading.Event()
    planning_ms = []
    errors = []
    started = 0.0

    def plan():
        begin.wait()
        index = 0
        while (index + 1) * args.interval < args.seconds and not stop.wait(
            max(0, started + (index + 1) * args.interval - time.perf_counter())
        ):
            stats, old_map, expected = snapshots[index % len(snapshots)]
            active.set()
            planning_started = time.perf_counter_ns()
            try:
                result = policy.rebalance_experts(stats, old_map)
                assert torch.equal(result, expected)
                planning_ms.append((time.perf_counter_ns() - planning_started) / 1e6)
            except Exception as exc:
                errors.append(str(exc))
                stop.set()
            finally:
                active.clear()
            index += 1

    worker = threading.Thread(target=plan) if mode != "idle" else None
    if worker is not None:
        worker.start()
    tensor = torch.zeros(1, device=args.device)
    if device is not None:
        for _ in range(32):
            tensor.add_(1)
        device.synchronize()
    samples = []
    busy_samples = []
    submit_samples = []
    scheduling_samples = []
    started = time.perf_counter()
    deadline = started
    begin.set()
    try:
        while time.perf_counter() - started < args.seconds and not stop.is_set():
            if args.period_ms:
                stop.wait(max(0, deadline - time.perf_counter()))
            busy = active.is_set()
            before = time.perf_counter_ns()
            scheduling_samples.append(max(0, before / 1e9 - deadline) * 1e6)
            sum(i * i for i in range(args.work))
            if device is not None:
                tensor.add_(1)
                submitted = time.perf_counter_ns()
                device.synchronize()
            else:
                submitted = time.perf_counter_ns()
            elapsed_us = (time.perf_counter_ns() - before) / 1000
            samples.append(elapsed_us)
            submit_samples.append((submitted - before) / 1000)
            if busy:
                busy_samples.append(elapsed_us)
            deadline = max(deadline + args.period_ms / 1000, time.perf_counter())
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if worker is not None:
            worker.join(timeout=65)
            assert not worker.is_alive(), "planner did not terminate"
        if process is not None:
            process.close()
    assert not errors, errors
    assert mode == "idle" or planning_ms, "no policy computation completed"

    def summary(values):
        if not values:
            return None
        return dict(
            count=len(values),
            mean_us=float(np.mean(values)),
            p50_us=float(np.percentile(values, 50)),
            p99_us=float(np.percentile(values, 99)),
            p999_us=float(np.percentile(values, 99.9)),
            max_us=float(np.max(values)),
            over_1ms=int(np.count_nonzero(np.asarray(values) > 1000)),
        )

    return dict(
        mode=mode,
        startup_ms=startup_ms,
        seconds=elapsed,
        steps_per_second=len(samples) / elapsed,
        foreground=summary(samples),
        policy_active_foreground=summary(busy_samples),
        cpu_submission=summary(submit_samples),
        scheduling_delay=summary(scheduling_samples),
        planning_ms=planning_ms,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--layers", type=int, default=58)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--replicas", type=int, default=288)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--period-ms", type=float, default=1)
    parser.add_argument("--work", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = None
    if args.device == "npu":
        import torch_npu  # noqa: F401

        device = torch.npu
    elif args.device == "cuda":
        device = torch.cuda
    generator = torch.Generator().manual_seed(17)
    policy = DefaultEplbPolicy()
    snapshots = []
    old_map = None
    for _ in range(8):
        loads = torch.randint(0, 1000, (args.layers, args.experts), generator=generator)
        stats = EplbStats(loads, args.replicas, 1, 1, args.ranks)
        expected = policy.rebalance_experts(stats, old_map)
        snapshots.append((stats, old_map, expected))
        old_map = expected
    results = []
    modes = ["idle", "thread", "process"]
    for repeat in range(args.repeats):
        for mode in modes[repeat % 3 :] + modes[: repeat % 3]:
            result = measure(args, mode, snapshots, device)
            result["repeat"] = repeat
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                platform=platform.platform(),
                torch=torch.__version__,
                args={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
