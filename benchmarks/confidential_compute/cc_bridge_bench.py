#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU<->GPU "bridge" microbenchmark for NVIDIA Confidential Computing (CC).

Under bounce-buffer CC, host<->device copies are encrypted by the driver and a
``cudaMemcpyAsync`` (``tensor.copy_(..., non_blocking=True)``) becomes
host-synchronous: the issuing thread blocks until the copy *and the work
already queued on its stream* has finished. This script measures that
behaviour directly and shows its effect on a vLLM-style decode loop, so that
CC-on and CC-off runs on the same machine can be compared side by side.

Tests (select with ``--tests``):

* ``semantics``   -- does a ``non_blocking`` copy return immediately, or block
                     the host until a busy stream drains? Same stream vs an
                     idle stream vs an idle stream that event-waits on the
                     busy one (the D2H readback pattern).
* ``bandwidth``   -- H2D / D2H latency and bandwidth vs transfer size, pinned
                     vs pageable host memory.
* ``concurrency`` -- aggregate H2D / D2H bandwidth with N streams on one
                     thread, N threads (one stream each), or N processes (one
                     CUDA context each).
* ``decode``      -- an emulated decode loop (CPU prep + small H2D + GPU
                     "forward" + small D2H readback) under sync scheduling,
                     async scheduling (vLLM's default), and async scheduling
                     with the staged-H2D / D2H-worker mitigations from
                     vllm-project/vllm#52226.

Only ``torch`` is required; ``nvidia-ml-py`` is used to detect the CC mode.
vLLM itself is not imported, so the script can be copied anywhere.

Usage::

    python benchmarks/confidential_compute/cc_bridge_bench.py
    python benchmarks/confidential_compute/cc_bridge_bench.py --quick
    python benchmarks/confidential_compute/cc_bridge_bench.py --tests decode

Results are written as JSON + Markdown to ``--out-dir``; compare a CC-off and
a CC-on run with ``compare.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import statistics
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

SCHEMA_VERSION = 1
KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB

ALL_TESTS = ("semantics", "bandwidth", "concurrency", "decode")
DECODE_MODES = (
    "sync_sched",
    "async_sched",
    "async_staged_h2d",
    "async_staged_h2d_d2h_worker",
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def parse_size(text: str) -> int:
    text = text.strip().upper().removesuffix("B").removesuffix("I")
    mult = {"K": KiB, "M": MiB, "G": GiB}.get(text[-1:], 1)
    if mult != 1:
        text = text[:-1]
    return int(float(text) * mult)


def fmt_size(nbytes: int) -> str:
    for unit, scale in (("G", GiB), ("M", MiB), ("K", KiB)):
        if nbytes >= scale and nbytes % scale == 0:
            return f"{nbytes // scale}{unit}"
    return f"{nbytes}B"


def csv(cast):
    return lambda text: [cast(x) for x in text.split(",") if x.strip()]


def stats_us(samples_s: list[float]) -> dict[str, float]:
    us = sorted(x * 1e6 for x in samples_s)
    n = len(us)
    return {
        "median_us": statistics.median(us),
        "p10_us": us[int(0.1 * (n - 1))],
        "p90_us": us[int(0.9 * (n - 1))],
        "min_us": us[0],
        "mean_us": statistics.fmean(us),
        "n": n,
    }


def now() -> float:
    # CLOCK_MONOTONIC is comparable across processes (concurrency test).
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def cpu_busy(us: float) -> None:
    """Hold the CPU (and the GIL) for ``us`` microseconds, like scheduler work."""
    end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < end:
        pass


def host_tensor(nbytes: int, pinned: bool) -> torch.Tensor:
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
    t.fill_(1)  # fault pageable pages in before timing
    return t


def copy_(direction: str, host: torch.Tensor, dev: torch.Tensor) -> None:
    if direction == "h2d":
        dev.copy_(host, non_blocking=True)
    elif direction == "d2h":
        host.copy_(dev, non_blocking=True)
    else:
        raise ValueError(direction)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Environment / CC detection
# --------------------------------------------------------------------------

_CC_ENV = {0: "unavailable", 1: "sim", 2: "prod"}
_CC_MULTI_GPU = {0: "none", 1: "protected_pcie", 2: "nvle"}


def detect_cpu_tee() -> dict[str, Any]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return {
        "tdx_guest": "tdx_guest" in flags
        or os.path.exists("/dev/tdx_guest")
        or os.path.exists("/dev/tdx-guest"),
        "sev_snp_guest": os.path.exists("/dev/sev-guest"),
        "cpu_flags": sorted(flags & {"tdx_guest", "sev", "sev_es", "sev_snp"}),
    }


def detect_gpu_cc() -> dict[str, Any]:
    """Query the system CC state through NVML (same source as vLLM#52226)."""
    try:
        import pynvml
    except ImportError:
        return {"nvml": False, "error": "nvidia-ml-py not installed"}

    info: dict[str, Any] = {"nvml": True}
    try:
        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        return {"nvml": False, "error": repr(e)}
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        info["driver"] = driver.decode() if isinstance(driver, bytes) else driver
        try:
            state = pynvml.nvmlSystemGetConfComputeState()
            info["cc_feature"] = int(state.ccFeature)
            info["cc_environment"] = _CC_ENV.get(state.environment, state.environment)
            info["cc_devtools_mode"] = bool(state.devToolsMode)
            info["cc_enabled"] = int(state.ccFeature) != 0
        except Exception as e:  # noqa: BLE001
            info["cc_state_error"] = repr(e)
        try:
            settings = pynvml.c_nvmlSystemConfComputeSettings_v1_t()
            settings.version = pynvml.nvmlSystemConfComputeSettings_v1
            if pynvml.nvmlSystemGetConfComputeSettings(ctypes.byref(settings)) == 0:
                mode = int(settings.multiGpuMode)
                info["cc_multi_gpu_mode"] = _CC_MULTI_GPU.get(mode, mode)
        except Exception:  # noqa: BLE001
            pass
        try:
            ready = pynvml.nvmlSystemGetConfComputeGpusReadyState()
            info["cc_gpus_ready"] = bool(ready)
        except Exception:  # noqa: BLE001
            pass
        try:
            caps = pynvml.nvmlSystemGetConfComputeCapabilities()
            info["cc_cpu_caps"] = int(caps.cpuCaps)
            info["cc_gpus_capable"] = bool(caps.gpusCaps)
        except Exception:  # noqa: BLE001
            pass
    finally:
        pynvml.nvmlShutdown()
    return info


def collect_env(device: torch.device, cc_label: str) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    gpu_cc = detect_gpu_cc()
    if cc_label == "auto":
        detected = gpu_cc.get("cc_enabled")
        cc = "unknown" if detected is None else ("on" if detected else "off")
    else:
        cc = cc_label
    return {
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "gpu_count": torch.cuda.device_count(),
        "device_index": device.index,
        "compute_capability": f"{props.major}.{props.minor}",
        "cc": cc,
        "cc_label_source": "nvml" if cc_label == "auto" else "--cc-label",
        "gpu_cc": gpu_cc,
        "cpu_tee": detect_cpu_tee(),
        "kernel": platform.release(),
    }


# --------------------------------------------------------------------------
# GPU busy kernel
# --------------------------------------------------------------------------


class GpuSpinner:
    """Enqueue a GPU kernel that runs for a requested number of microseconds."""

    def __init__(self, device: torch.device):
        self.device = device
        cycles = 1 << 22
        torch.cuda._sleep(cycles)
        torch.cuda.synchronize(device)
        per_us = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(cycles)
            end.record()
            end.synchronize()
            per_us.append(cycles / (start.elapsed_time(end) * 1e3))
        self.cycles_per_us = statistics.median(per_us)

    def spin(self, us: float) -> None:
        torch.cuda._sleep(max(1, int(us * self.cycles_per_us)))


# --------------------------------------------------------------------------
# Test 1: async semantics
# --------------------------------------------------------------------------


def bench_semantics(args, device, spinner) -> list[dict[str, Any]]:
    """Time how long a non_blocking copy call takes to *return* while another
    stream (or the same one) is busy for ``busy_us``."""
    size = args.semantics_size
    busy_us = args.busy_us
    compute = torch.cuda.Stream(device)
    other = torch.cuda.Stream(device)
    dev = torch.empty(size, dtype=torch.uint8, device=device)
    dev2 = torch.empty_like(dev)
    hosts = {p: host_tensor(size, p) for p in (True, False)}

    cases = []
    for direction in ("h2d", "d2h"):
        for pinned in (True, False):
            for placement in ("same_stream", "idle_stream", "event_wait_stream"):
                cases.append((direction, pinned, placement))
    cases.append(("d2d", True, "same_stream"))

    rows = []
    for direction, pinned, placement in cases:
        call_s, total_s = [], []
        for rep in range(args.semantics_reps + 1):
            torch.cuda.synchronize(device)
            with torch.cuda.stream(compute):
                spinner.spin(busy_us)
                busy_done = torch.cuda.Event()
                busy_done.record(compute)
            target = compute if placement == "same_stream" else other
            if placement == "event_wait_stream":
                other.wait_event(busy_done)
            t0 = now()
            with torch.cuda.stream(target):
                if direction == "d2d":
                    dev2.copy_(dev, non_blocking=True)
                else:
                    copy_(direction, hosts[pinned], dev)
            t1 = now()
            torch.cuda.synchronize(device)
            t2 = now()
            if rep > 0:
                call_s.append(t1 - t0)
                total_s.append(t2 - t0)
        call = stats_us(call_s)
        frac = call["median_us"] / busy_us
        verdict = (
            "async"
            if frac < 0.1
            else "blocks_on_busy_work"
            if frac > 0.8
            else "partial"
        )
        memory = "device" if direction == "d2d" else "pinned" if pinned else "pageable"
        rows.append(
            {
                "key": f"{direction}/{memory}/{placement}",
                "direction": direction,
                "memory": memory,
                "placement": placement,
                "size": size,
                "busy_us": busy_us,
                "call_return_us": call["median_us"],
                "call_return_p90_us": call["p90_us"],
                "completion_us": stats_us(total_s)["median_us"],
                "blocked_fraction": frac,
                "verdict": verdict,
            }
        )
        log(
            f"  semantics {rows[-1]['key']:<38} call={call['median_us']:>9.1f}us "
            f"({verdict})"
        )
    return rows


# --------------------------------------------------------------------------
# Test 2: latency / bandwidth vs size
# --------------------------------------------------------------------------


def bench_bandwidth(args, device) -> list[dict[str, Any]]:
    stream = torch.cuda.Stream(device)
    rows = []
    for size in args.sizes:
        dev = torch.empty(size, dtype=torch.uint8, device=device)
        for pinned in (True, False):
            host = host_tensor(size, pinned)
            for direction in ("h2d", "d2h"):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        copy_(direction, host, dev)
                    stream.synchronize()

                    lat = []
                    for _ in range(args.latency_iters):
                        t0 = now()
                        copy_(direction, host, dev)
                        stream.synchronize()
                        lat.append(now() - t0)

                    iters = min(
                        args.max_iters,
                        max(args.min_iters, math.ceil(args.target_bytes / size)),
                    )
                    t0 = now()
                    for _ in range(iters):
                        copy_(direction, host, dev)
                    issued = now()
                    stream.synchronize()
                    elapsed = now() - t0

                lat_stats = stats_us(lat)
                row = {
                    "key": f"{direction}/{'pinned' if pinned else 'pageable'}/"
                    f"{fmt_size(size)}",
                    "direction": direction,
                    "memory": "pinned" if pinned else "pageable",
                    "size": size,
                    "latency_median_us": lat_stats["median_us"],
                    "latency_p90_us": lat_stats["p90_us"],
                    "iters": iters,
                    "bandwidth_gbps": size * iters / elapsed / 1e9,
                    # ~1.0 means the host thread was stuck in the copy calls.
                    "issue_fraction": (issued - t0) / elapsed,
                }
                rows.append(row)
                log(
                    f"  bandwidth {row['key']:<22} "
                    f"lat={row['latency_median_us']:>9.1f}us "
                    f"bw={row['bandwidth_gbps']:>7.2f}GB/s"
                )
            del host
        del dev
    return rows


# --------------------------------------------------------------------------
# Test 3: concurrency (streams / threads / processes)
# --------------------------------------------------------------------------


def _alloc_pair(size: int, device: torch.device):
    return host_tensor(size, True), torch.empty(size, dtype=torch.uint8, device=device)


def _run_streams(direction, size, iters, n, device) -> tuple[float, float]:
    streams = [torch.cuda.Stream(device) for _ in range(n)]
    bufs = [_alloc_pair(size, device) for _ in range(n)]
    for s, (h, d) in zip(streams, bufs):
        with torch.cuda.stream(s):
            copy_(direction, h, d)
    torch.cuda.synchronize(device)
    t0 = now()
    for _ in range(iters):
        for s, (h, d) in zip(streams, bufs):
            with torch.cuda.stream(s):
                copy_(direction, h, d)
    torch.cuda.synchronize(device)
    return t0, now()


def _run_threads(direction, size, iters, n, device) -> tuple[float, float]:
    barrier = threading.Barrier(n)
    spans: list[tuple[float, float]] = [(0.0, 0.0)] * n

    def worker(i: int) -> None:
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        host, dev = _alloc_pair(size, device)
        with torch.cuda.stream(stream):
            copy_(direction, host, dev)
            stream.synchronize()
            barrier.wait()
            t0 = now()
            for _ in range(iters):
                copy_(direction, host, dev)
            stream.synchronize()
            spans[i] = (t0, now())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _process_worker(direction, size, iters, device_index, barrier, out_q) -> None:
    try:
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        host, dev = _alloc_pair(size, device)
        with torch.cuda.stream(stream):
            copy_(direction, host, dev)
            stream.synchronize()
            barrier.wait()
            t0 = now()
            for _ in range(iters):
                copy_(direction, host, dev)
            stream.synchronize()
        out_q.put((t0, now()))
    except BaseException as e:  # noqa: BLE001
        barrier.abort()
        out_q.put(repr(e))
        raise


def _run_processes(direction, size, iters, n, device) -> tuple[float, float]:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n)
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_process_worker,
            args=(direction, size, iters, device.index, barrier, out_q),
        )
        for _ in range(n)
    ]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=600) for _ in procs]
    for p in procs:
        p.join()
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        raise RuntimeError(f"concurrency worker failed: {errors[0]}")
    return min(r[0] for r in results), max(r[1] for r in results)


def bench_concurrency(args, device) -> list[dict[str, Any]]:
    runners = {
        "streams": _run_streams,
        "threads": _run_threads,
        "processes": _run_processes,
    }
    size = args.concurrency_size
    iters = max(1, math.ceil(args.concurrency_bytes / size))
    rows = []
    for mode in args.concurrency_modes:
        for direction in ("h2d", "d2h"):
            base = None
            for n in args.workers:
                t0, t1 = runners[mode](direction, size, iters, n, device)
                bw = n * iters * size / (t1 - t0) / 1e9
                base = base or bw
                row = {
                    "key": f"{mode}/{direction}/n{n}",
                    "mode": mode,
                    "direction": direction,
                    "workers": n,
                    "size": size,
                    "iters_per_worker": iters,
                    "aggregate_gbps": bw,
                    "scaling_vs_1": bw / base,
                }
                rows.append(row)
                log(
                    f"  concurrency {row['key']:<20} {bw:>7.2f}GB/s "
                    f"(x{row['scaling_vs_1']:.2f})"
                )
    return rows


# --------------------------------------------------------------------------
# Test 4: emulated decode loop
# --------------------------------------------------------------------------


class DecodeLoop:
    """A minimal model of the vLLM V1 decode step.

    Per step: ``cpu_us`` of host work (schedule + prepare inputs), an H2D of
    ``h2d_bytes`` of input metadata, a GPU "forward" of ``fwd_us``, then a D2H
    readback of ``d2h_bytes`` of sampled tokens. The forward writes
    ``input + 1`` to its output, and every readback is checked, so a racy
    pipeline fails loudly instead of reporting a fast number.

    Modes:
      sync_sched                  -- no overlap; D2H then wait, every step.
      async_sched                 -- vLLM async scheduling: prep step N+1
                                     while step N runs; H2D on the compute
                                     stream; D2H on a copy stream after an
                                     event wait, issued by the main thread.
      async_staged_h2d            -- + H2D into a staging buffer on a prep
                                     stream, then D2D on the compute stream.
      async_staged_h2d_d2h_worker -- + the D2H is issued by a worker thread.
    """

    RING = 4

    def __init__(self, args, device, spinner, fwd_us: float, mode: str):
        assert mode in DECODE_MODES, mode
        self.device = device
        self.spinner = spinner
        self.fwd_us = fwd_us
        self.cpu_us = args.cpu_us
        self.mode = mode
        self.n_in = max(1, args.h2d_bytes // 4)
        self.n_out = max(1, min(self.n_in, args.d2h_bytes // 4))
        i32 = torch.int32

        self.compute = torch.cuda.Stream(device)
        self.copy_stream = torch.cuda.Stream(device)
        self.prep_stream = torch.cuda.Stream(device)
        self.gpu_in = torch.zeros(self.n_in, dtype=i32, device=device)
        self.gpu_out = [
            torch.zeros(self.n_out, dtype=i32, device=device) for _ in range(self.RING)
        ]
        self.stage = [torch.zeros_like(self.gpu_in) for _ in range(2)]
        self.stage_free: list[torch.cuda.Event | None] = [None, None]
        self.host_in = [
            torch.zeros(self.n_in, dtype=i32, pin_memory=True) for _ in range(self.RING)
        ]
        self.host_in_free: list[torch.cuda.Event | None] = [None] * self.RING
        self.host_out = [
            torch.zeros(self.n_out, dtype=i32, pin_memory=True)
            for _ in range(self.RING)
        ]
        self.pool = (
            ThreadPoolExecutor(1, thread_name_prefix="d2h")
            if mode == "async_staged_h2d_d2h_worker"
            else None
        )
        self.worker_stream: torch.cuda.Stream | None = None
        self.host_block_s = 0.0

    def _h2d(self, step: int, slot: int) -> None:
        src = self.host_in[slot]
        if self.mode in ("sync_sched", "async_sched"):
            with torch.cuda.stream(self.compute):
                self.gpu_in.copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.compute)
            self.host_in_free[slot] = ev
            return
        # Staged path (StagedH2DCopier in vLLM#52226). The extra events keep it
        # correct without CC too, where the H2D really is asynchronous.
        idx = step & 1
        stage = self.stage[idx]
        with torch.cuda.stream(self.prep_stream):
            if self.stage_free[idx] is not None:
                self.prep_stream.wait_event(self.stage_free[idx])
            stage.copy_(src, non_blocking=True)
            staged = torch.cuda.Event()
            staged.record(self.prep_stream)
        self.host_in_free[slot] = staged
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(staged)
            self.gpu_in.copy_(stage, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.compute)
        self.stage_free[idx] = ev

    def _d2h_on_worker(self, fwd_done: torch.cuda.Event, slot: int) -> None:
        if self.worker_stream is None:
            torch.cuda.set_device(self.device)
            self.worker_stream = torch.cuda.Stream(self.device)
        stream = self.worker_stream
        stream.wait_event(fwd_done)
        with torch.cuda.stream(stream):
            self.host_out[slot].copy_(self.gpu_out[slot], non_blocking=True)
        stream.synchronize()

    def _check(self, step: int, slot: int) -> None:
        got = self.host_out[slot]
        if int(got[0]) != step + 1 or int(got[-1]) != step + 1:
            raise RuntimeError(
                f"decode[{self.mode}] step {step}: read back {int(got[0])}, "
                f"expected {step + 1} -- pipeline race"
            )

    def run(self, steps: int, warmup: int) -> dict[str, Any]:
        try:
            return self._run(steps, warmup)
        finally:
            if self.pool is not None:
                self.pool.shutdown()

    def _run(self, steps: int, warmup: int) -> dict[str, Any]:
        torch.cuda.synchronize(self.device)
        pending: deque[tuple[int, int, Any]] = deque()
        t_start = 0.0
        total = warmup + steps
        for step in range(total):
            if step == warmup:
                torch.cuda.synchronize(self.device)
                t_start = now()
                self.host_block_s = 0.0
            slot = step % self.RING
            cpu_busy(self.cpu_us)
            if self.host_in_free[slot] is not None:
                self.host_in_free[slot].synchronize()
            self.host_in[slot].fill_(step)

            t0 = now()
            self._h2d(step, slot)
            self.host_block_s += now() - t0

            with torch.cuda.stream(self.compute):
                self.spinner.spin(self.fwd_us)
                torch.add(self.gpu_in[: self.n_out], 1, out=self.gpu_out[slot])
                fwd_done = torch.cuda.Event()
                fwd_done.record(self.compute)

            t0 = now()
            if self.mode == "sync_sched":
                with torch.cuda.stream(self.compute):
                    self.host_out[slot].copy_(self.gpu_out[slot], non_blocking=True)
                self.compute.synchronize()
                self._check(step, slot)
            elif self.pool is not None:
                fut = self.pool.submit(self._d2h_on_worker, fwd_done, slot)
                pending.append((step, slot, fut))
            else:
                self.copy_stream.wait_event(fwd_done)
                with torch.cuda.stream(self.copy_stream):
                    self.host_out[slot].copy_(self.gpu_out[slot], non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self.copy_stream)
                pending.append((step, slot, done))
            self.host_block_s += now() - t0

            # Like vLLM async scheduling: at most one step in flight beyond the
            # one just launched; its output is consumed before scheduling on.
            while len(pending) > 1:
                self._consume(pending.popleft())
        while pending:
            self._consume(pending.popleft())
        torch.cuda.synchronize(self.device)
        wall = now() - t_start

        step_us = wall / steps * 1e6
        ideal_us = (
            self.cpu_us + self.fwd_us
            if self.mode == "sync_sched"
            else max(self.cpu_us, self.fwd_us)
        )
        return {
            "step_us": step_us,
            "steps_per_s": steps / wall,
            "gpu_busy_fraction": min(1.0, self.fwd_us / step_us),
            "host_block_us_per_step": self.host_block_s / steps * 1e6,
            "ideal_step_us": ideal_us,
            "efficiency_vs_ideal": ideal_us / step_us,
        }

    def _consume(self, item: tuple[int, int, Any]) -> None:
        step, slot, handle = item
        if isinstance(handle, Future):
            handle.result()
        else:
            handle.synchronize()
        self._check(step, slot)


def bench_decode(args, device, spinner) -> list[dict[str, Any]]:
    rows = []
    for fwd_us in args.fwd_us:
        sync_step = None
        for mode in DECODE_MODES:
            res = DecodeLoop(args, device, spinner, fwd_us, mode).run(
                args.steps, args.warmup_steps
            )
            if mode == "sync_sched":
                sync_step = res["step_us"]
            row = {
                "key": f"fwd{int(fwd_us)}us/{mode}",
                "fwd_us": fwd_us,
                "cpu_us": args.cpu_us,
                "h2d_bytes": args.h2d_bytes,
                "d2h_bytes": args.d2h_bytes,
                "mode": mode,
                **res,
                "speedup_vs_sync": sync_step / res["step_us"],
            }
            rows.append(row)
            log(
                f"  decode {row['key']:<40} step={res['step_us']:>9.1f}us "
                f"gpu_busy={res['gpu_busy_fraction']:.2f} "
                f"x{row['speedup_vs_sync']:.2f} vs sync"
            )
    return rows


# --------------------------------------------------------------------------
# Findings + reporting
# --------------------------------------------------------------------------


def findings(results: dict[str, Any]) -> list[str]:
    """Plain-language conclusions for the most decision-relevant numbers."""
    out = []
    sem = {r["key"]: r for r in results.get("semantics", [])}
    same = sem.get("h2d/pinned/same_stream")
    idle = sem.get("h2d/pinned/idle_stream")
    if same:
        if same["verdict"] == "blocks_on_busy_work":
            out.append(
                "Pinned H2D with non_blocking=True BLOCKS the host until the busy "
                f"stream drains ({same['call_return_us']:.0f}us of a "
                f"{same['busy_us']:.0f}us kernel). An H2D on the compute stream "
                "stalls the scheduler for ~one forward step."
            )
        else:
            out.append(
                "Pinned H2D with non_blocking=True returns without waiting for "
                f"queued work ({same['call_return_us']:.1f}us): copies are async."
            )
    if idle and same and same["verdict"] == "blocks_on_busy_work":
        if idle["verdict"] == "blocks_on_busy_work":
            out.append(
                "An H2D on an *idle* stream also waits for the busy stream: the "
                "copy path is serialised device-wide, so staged H2D on a prep "
                "stream (vLLM#52226) cannot help on this driver."
            )
        else:
            out.append(
                "An H2D on an idle stream only pays its own transfer "
                f"({idle['call_return_us']:.0f}us): staging H2D on a dedicated "
                "prep stream (vLLM#52226) avoids the stall."
            )
    by_key = {r["key"]: r for r in results.get("decode", [])}
    for fwd in sorted({r["fwd_us"] for r in results.get("decode", [])}):
        a = by_key.get(f"fwd{int(fwd)}us/async_sched")
        s = by_key.get(f"fwd{int(fwd)}us/sync_sched")
        m = by_key.get(f"fwd{int(fwd)}us/async_staged_h2d_d2h_worker")
        if not (a and s and m):
            continue
        line = (
            f"Decode @ {fwd / 1e3:.1f}ms forward: async scheduling is "
            f"x{a['speedup_vs_sync']:.2f} vs sync"
        )
        if a["speedup_vs_sync"] < 1.03:
            line += " (POLICY INVERSION: the default buys nothing or hurts)"
        line += (
            f"; with staged H2D + D2H worker x{m['speedup_vs_sync']:.2f}, "
            f"GPU busy {a['gpu_busy_fraction']:.0%} -> {m['gpu_busy_fraction']:.0%}."
        )
        out.append(line)
    return out


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key, "")
            cells.append(
                format(v, fmt) if fmt and isinstance(v, (int, float)) else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLE_COLUMNS: dict[str, list[tuple[str, str, str]]] = {
    "semantics": [
        ("key", "case", ""),
        ("call_return_us", "call returns (us)", ".1f"),
        ("completion_us", "completes (us)", ".1f"),
        ("blocked_fraction", "blocked / busy", ".2f"),
        ("verdict", "verdict", ""),
    ],
    "bandwidth": [
        ("key", "case", ""),
        ("latency_median_us", "latency (us)", ".1f"),
        ("bandwidth_gbps", "bandwidth (GB/s)", ".2f"),
        ("issue_fraction", "host-issue / total", ".2f"),
    ],
    "concurrency": [
        ("key", "case", ""),
        ("aggregate_gbps", "aggregate (GB/s)", ".2f"),
        ("scaling_vs_1", "scaling vs n=1", ".2f"),
    ],
    "decode": [
        ("key", "case", ""),
        ("step_us", "step (us)", ".1f"),
        ("gpu_busy_fraction", "GPU busy", ".2f"),
        ("host_block_us_per_step", "host blocked in copies (us/step)", ".1f"),
        ("speedup_vs_sync", "speedup vs sync", ".2f"),
    ],
}


def render_markdown(results: dict[str, Any]) -> str:
    env = results["env"]
    gpu_cc = env["gpu_cc"]
    parts = [
        f"# CC bridge microbench: {env['gpu']} (CC {env['cc'].upper()})",
        "",
        f"- host `{env['hostname']}`, {env['timestamp']}",
        (
            f"- torch {env['torch']}, CUDA {env['cuda_runtime']}, "
            f"driver {gpu_cc.get('driver', '?')}"
        ),
        (
            f"- CC: feature={gpu_cc.get('cc_feature', '?')} "
            f"devtools={gpu_cc.get('cc_devtools_mode', '?')} "
            f"multi_gpu={gpu_cc.get('cc_multi_gpu_mode', '?')} "
            f"(source: {env['cc_label_source']}); CPU TEE: "
            f"tdx={env['cpu_tee']['tdx_guest']} "
            f"snp={env['cpu_tee']['sev_snp_guest']}"
        ),
        "",
    ]
    if results.get("findings"):
        parts += ["## Findings", ""] + [f"- {f}" for f in results["findings"]] + [""]
    for test in ALL_TESTS:
        if results.get(test):
            parts += [f"## {test}", "", _table(results[test], TABLE_COLUMNS[test]), ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tests",
        type=csv(str),
        default=list(ALL_TESTS),
        help=f"comma-separated subset of {','.join(ALL_TESTS)}",
    )
    p.add_argument(
        "--quick", action="store_true", help="fewer sizes / reps / steps (~1 min)"
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out-dir", default="cc_bench_results")
    p.add_argument(
        "--cc-label",
        choices=["auto", "on", "off"],
        default="auto",
        help="label the run when NVML cannot report the CC state",
    )

    g = p.add_argument_group("semantics")
    g.add_argument(
        "--busy-us",
        type=float,
        default=20000,
        help="length of the kernel queued ahead of the copy",
    )
    g.add_argument("--semantics-size", type=parse_size, default="64K")
    g.add_argument("--semantics-reps", type=int, default=10)

    g = p.add_argument_group("bandwidth")
    g.add_argument("--sizes", type=csv(parse_size), default="4K,64K,1M,16M,256M")
    g.add_argument("--latency-iters", type=int, default=50)
    g.add_argument(
        "--target-bytes",
        type=parse_size,
        default="2G",
        help="bytes moved per bandwidth point",
    )
    g.add_argument("--min-iters", type=int, default=8)
    g.add_argument("--max-iters", type=int, default=5000)

    g = p.add_argument_group("concurrency")
    g.add_argument(
        "--concurrency-modes", type=csv(str), default="streams,threads,processes"
    )
    g.add_argument("--workers", type=csv(int), default="1,2,4,8")
    g.add_argument("--concurrency-size", type=parse_size, default="64M")
    g.add_argument(
        "--concurrency-bytes",
        type=parse_size,
        default="1G",
        help="bytes moved per worker",
    )

    g = p.add_argument_group("decode")
    g.add_argument(
        "--fwd-us",
        type=csv(float),
        default="3000,10000,25000",
        help="emulated forward times (small = low concurrency)",
    )
    g.add_argument(
        "--cpu-us",
        type=float,
        default=1500,
        help="host work per step (schedule + prepare inputs)",
    )
    g.add_argument("--h2d-bytes", type=parse_size, default="64K")
    g.add_argument("--d2h-bytes", type=parse_size, default="4K")
    g.add_argument("--steps", type=int, default=200)
    g.add_argument("--warmup-steps", type=int, default=20)
    return p


def apply_quick(args) -> None:
    args.semantics_reps = 5
    args.sizes = [64 * KiB, 16 * MiB]
    args.latency_iters = 20
    args.target_bytes = 256 * MiB
    args.workers = [1, 4]
    args.concurrency_modes = [m for m in args.concurrency_modes if m != "processes"]
    args.concurrency_bytes = 256 * MiB
    args.fwd_us = [3000, 10000]
    args.steps = 60
    args.warmup_steps = 10


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.tests) - set(ALL_TESTS)
    if unknown:
        raise SystemExit(f"unknown tests: {sorted(unknown)}")
    if args.quick:
        apply_quick(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: this benchmark needs an NVIDIA GPU.")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    env = collect_env(device, args.cc_label)
    log(
        f"GPU {env['gpu']} | CC {env['cc']} | driver "
        f"{env['gpu_cc'].get('driver', '?')} | torch {env['torch']}"
    )
    if env["cc"] == "unknown":
        log("warning: CC state unknown (install nvidia-ml-py or pass --cc-label)")

    spinner = GpuSpinner(device)
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "env": env,
        "args": {k: v for k, v in vars(args).items()},
        "gpu_cycles_per_us": spinner.cycles_per_us,
    }
    for test in args.tests:
        log(f"[{test}]")
        t0 = time.time()
        if test == "semantics":
            results[test] = bench_semantics(args, device, spinner)
        elif test == "bandwidth":
            results[test] = bench_bandwidth(args, device)
        elif test == "concurrency":
            results[test] = bench_concurrency(args, device)
        elif test == "decode":
            results[test] = bench_decode(args, device, spinner)
        log(f"[{test}] done in {time.time() - t0:.1f}s")
    results["findings"] = findings(results)

    os.makedirs(args.out_dir, exist_ok=True)
    gpu_tag = env["gpu"].replace("NVIDIA ", "").replace(" ", "-")
    stem = os.path.join(
        args.out_dir,
        f"{env['hostname']}_{gpu_tag}_cc-{env['cc']}_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    with open(stem + ".json", "w") as f:
        json.dump(results, f, indent=2)
    md = render_markdown(results)
    with open(stem + ".md", "w") as f:
        f.write(md + "\n")
    print(md)
    log(f"wrote {stem}.json and {stem}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
