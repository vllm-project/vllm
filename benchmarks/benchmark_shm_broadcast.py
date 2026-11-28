# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark `vllm.distributed.device_communicators.shm_broadcast`.

The harness models the `MultiprocExecutor` <-> `WorkerProc` loop: one writer
process (the "executor") broadcasts an RPC-sized object on a `MessageQueue` to
N local reader processes ("workers"); every worker answers on its own
`MessageQueue(1, 1)` and the executor waits for all answers before the next
step, i.e. lock-step like `collective_rpc`.

Only public `MessageQueue` APIs are used (`MessageQueue`, `export_handle`,
`create_from_handle`, `wait_until_ready`, `enqueue`, `dequeue`, `shutdown`), so
the same file runs unchanged against any vLLM tree selected through
`PYTHONPATH`:

```bash
TMPDIR=/dev/shm/vbench PYTHONPATH=/path/to/vllm \
VLLM_TARGET_DEVICE=cpu python benchmarks/benchmark_shm_broadcast.py \
    --variant main --readers 1,4 --json-out results.jsonl
```

Per configuration it reports

* step round-trip latency (executor send -> last reply received),
* one-way broadcast latency (worker receive time - executor send timestamp;
  both from the system-wide `CLOCK_MONOTONIC`),
* worker-side handling time (reply timestamp - receive timestamp),
* CPU-seconds / wall-seconds per process over the measured window, read from
  `/proc/<pid>/stat` (`utime` + `stime`, so every thread counts, including the
  zmq I/O threads),
* an idle scenario: queues connected, no traffic for `--idle-seconds`, CPU
  sampled over that window (once on a freshly built group, once after the
  traffic scenarios, so both the cold and the recently-busy spin behaviour
  are visible).

`--json-out` appends one JSON object per (variant, readers, scenario,
repeat); stdout gets a human-readable table.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from array import array
from collections.abc import Iterable, Sequence
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

import zmq

import vllm.distributed.device_communicators.shm_broadcast as shmb
from vllm import envs as vllm_envs
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue


def log(msg: str) -> None:
    print(f"[shm-bench] {msg}", flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def recv_msg(conn: Connection, timeout: float, what: str) -> tuple:
    """Read one protocol message, guarding against a hung peer."""
    if not conn.poll(timeout):
        fail(f"timeout ({timeout}s) waiting for {what}")
    msg = conn.recv()
    if not isinstance(msg, tuple) or not isinstance(msg[0], str):
        fail(f"malformed message while waiting for {what}: {msg!r}")
    return msg


def expect(conn: Connection, timeout: float, tag: str) -> tuple:
    """Read one protocol message and require `tag` as its first element."""
    msg = recv_msg(conn, timeout, tag)
    if msg[0] != tag:
        fail(f"expected {tag}, got {msg[0]}")
    return msg


def percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sequence."""
    n = len(ordered)
    if n == 0:
        return float("nan")
    idx = min(max(int(round(q * (n - 1))), 0), n - 1)
    return float(ordered[idx])


def summarize_us(values: Sequence[float]) -> dict[str, float]:
    """Latency summary in microseconds; input is integer microseconds."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": n,
        "mean": sum(ordered) / n,
        "p50": percentile(ordered, 0.50),
        "p99": percentile(ordered, 0.99),
        "p999": percentile(ordered, 0.999),
        "max": float(ordered[-1]),
    }


def proc_cpu_seconds(pid: int) -> float:
    """Utime + stime over every thread of `pid`, in seconds."""
    with open(f"/proc/{pid}/stat", "rb") as handle:
        raw = handle.read().decode("ascii", "replace")
    # Skip "pid (comm) " -- comm may contain spaces.
    fields = raw[raw.rfind(")") + 2 :].split()
    jiffies = int(fields[11]) + int(fields[12])  # utime, stime
    return jiffies / os.sysconf("SC_CLK_TCK")


# ---------------------------------------------------------------------------
# CPU affinity
# ---------------------------------------------------------------------------


# Snapshot of the cpuset taken before the harness pins itself to one core,
# which would otherwise shrink sched_getaffinity(0).
ALLOWED_CPUS = frozenset(os.sched_getaffinity(0))


def physical_core_ids(allowed: frozenset[int] = ALLOWED_CPUS) -> list[int]:
    """One (lowest) logical id per physical core, within the allowed cpuset."""
    reps: set[int] = set()
    base = Path("/sys/devices/system/cpu")
    for path in sorted(base.glob("cpu[0-9]*/topology/thread_siblings_list")):
        cpu = int(path.parts[-3].removeprefix("cpu"))
        rep = cpu
        for part in path.read_text().strip().split(","):
            if "-" in part:
                rep = min(rep, int(part.split("-")[0]))
            elif part.isdigit():
                rep = min(rep, int(part))
        if rep == cpu and cpu in allowed:
            reps.add(cpu)
    return sorted(reps) or sorted(allowed)


def cpu_busy_fraction(cpus: Sequence[int], sample_s: float) -> dict[int, float]:
    """Per-CPU busy fraction (1.0 == saturated) measured over `sample_s`."""
    lines = Path("/proc/stat").read_text().splitlines()
    before = {cpu: [int(f) for f in lines[cpu + 1].split()[1:]] for cpu in cpus}
    time.sleep(sample_s)
    lines = Path("/proc/stat").read_text().splitlines()
    busy: dict[int, float] = {}
    for cpu in cpus:
        now = [int(f) for f in lines[cpu + 1].split()[1:]]
        deltas = [n - p for n, p in zip(now, before[cpu])]
        idle = deltas[3] + deltas[4]  # idle + iowait
        busy[cpu] = 1.0 - idle / (sum(deltas) or 1)
    return busy


def pick_cpus(
    count: int, exclude: Iterable[int] = (), sample_s: float = 0.25
) -> list[int]:
    """`count` distinct physical cores, least busy first, none excluded."""
    taken = set(exclude)
    cores = [cpu for cpu in physical_core_ids() if cpu not in taken]
    if len(cores) < count:
        fail(f"need {count} free physical cores, found {len(cores)}")
    busy = cpu_busy_fraction(cores, sample_s)
    chosen = sorted(cores, key=lambda c: (busy[c], c))[:count]
    log("pinned cpus " + ", ".join(f"{c}({100 * busy[c]:.0f}%)" for c in chosen))
    return chosen


def apply_affinity(cpu: int | None) -> None:
    if cpu is None:
        return
    os.sched_setaffinity(0, {cpu})
    got = os.sched_getaffinity(0)
    if got != {cpu}:
        fail(f"could not pin to cpu {cpu}, got {sorted(got)}")


def quiet_native_threads() -> None:
    """Keep OpenMP/BLAS pools from polluting per-process CPU accounting."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover - torch is a vLLM hard dependency
        pass


# ---------------------------------------------------------------------------
# payload (RPC shaped, small enough to stay on the shared-memory path)
# ---------------------------------------------------------------------------


def rpc_object(step: int, send_ts: float, payload: bytes) -> tuple:
    """Stand-in for a `collective_rpc` broadcast."""
    return ("dummy_method", (step, "dummy_arg"), {"kw": step}, send_ts, payload)


def reply_object(
    step: int, rank: int, recv_ts: float, reply_ts: float, payload: bytes
) -> tuple:
    return (step, rank, recv_ts, reply_ts, payload)


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------


def executor_main(conn: Connection, opts: argparse.Namespace) -> None:
    quiet_native_threads()
    apply_affinity(opts.executor_cpu)
    payload = bytes(opts.payload_bytes)

    rpc_q = MessageQueue(
        opts.readers,
        opts.readers,
        max_chunk_bytes=opts.max_chunk_bytes,
        max_chunks=opts.max_chunks,
    )
    conn.send(("rpc_handle", rpc_q.export_handle()))

    _, reply_handles = expect(conn, opts.handshake_timeout, "reply_handles")
    reply_qs = [MessageQueue.create_from_handle(h, 0) for h in reply_handles]

    rpc_q.wait_until_ready()
    for reply_q in reply_qs:
        reply_q.wait_until_ready()
    conn.send(("ready", os.getpid(), sorted(os.sched_getaffinity(0))))

    while True:
        msg = recv_msg(conn, opts.command_timeout, "executor command")
        if msg[0] == "produce":
            _, steps, gap_ms, mode = msg
            produce(conn, rpc_q, reply_qs, steps, gap_ms / 1000.0, mode, payload, opts)
        elif msg[0] == "exit":
            break
        else:
            fail(f"unknown executor command {msg[0]}")

    for queue in [rpc_q, *reply_qs]:
        queue.shutdown()
    conn.send(("bye",))


def produce(
    conn: Connection,
    rpc_q: MessageQueue,
    reply_qs: list[MessageQueue],
    steps: int,
    gap_s: float,
    mode: str,
    payload: bytes,
    opts: argparse.Namespace,
) -> None:
    """Lock-step broadcast loop, mirroring `collective_rpc`.

    Runs exactly `steps` steps so the readers, which are told the same
    count up front, stay in lock-step. `mode` is `"probe"` (report the
    mean step cost only) or `"measured"` (report every sample).
    """
    rtts_us = array("q")
    oneway_us = array("q")
    handle_us = array("q")
    start = time.monotonic()
    conn.send(("measuring",))
    for step in range(steps):
        send_ts = time.monotonic()
        rpc_q.enqueue(rpc_object(step, send_ts, payload), timeout=opts.timeout)
        for reply_q in reply_qs:
            _, _, recv_ts, reply_ts, _ = reply_q.dequeue(timeout=opts.timeout)
            oneway_us.append(int((recv_ts - send_ts) * 1e6))
            handle_us.append(int((reply_ts - recv_ts) * 1e6))
        rtts_us.append(int((time.monotonic() - send_ts) * 1e6))
        if gap_s > 0:
            time.sleep(gap_s)
    wall = time.monotonic() - start
    if mode == "measured":
        conn.send(("measured_end", wall, rtts_us, oneway_us, handle_us, steps))
    else:
        conn.send(("probe_end", wall, steps))


def worker_main(
    conn: Connection,
    rank: int,
    stop_event: EventType,
    opts: argparse.Namespace,
) -> None:
    quiet_native_threads()
    apply_affinity(opts.worker_cpus[rank])
    payload = bytes(opts.payload_bytes)

    _, rpc_handle = expect(conn, opts.handshake_timeout, "rpc_handle")
    rpc_q = MessageQueue.create_from_handle(rpc_handle, rank)
    reply_q = MessageQueue(1, 1)
    conn.send(("reply_handle", reply_q.export_handle()))

    reply_q.wait_until_ready()
    rpc_q.wait_until_ready()
    conn.send(("ready", os.getpid(), sorted(os.sched_getaffinity(0))))

    while True:
        msg = recv_msg(conn, opts.command_timeout, f"worker {rank} command")
        if msg[0] == "consume":
            consume(conn, rank, rpc_q, reply_q, msg[1], payload, opts)
        elif msg[0] == "idle":
            idle(conn, rpc_q, stop_event)
        elif msg[0] == "exit":
            break
        else:
            fail(f"unknown worker command {msg[0]}")

    for queue in [rpc_q, reply_q]:
        queue.shutdown()
    conn.send(("bye",))


def consume(
    conn: Connection,
    rank: int,
    rpc_q: MessageQueue,
    reply_q: MessageQueue,
    steps: int,
    payload: bytes,
    opts: argparse.Namespace,
) -> None:
    conn.send(("consuming",))
    for _ in range(steps):
        msg = rpc_q.dequeue(timeout=opts.timeout)
        recv_ts = time.monotonic()
        reply_q.enqueue(
            reply_object(msg[0][0], rank, recv_ts, time.monotonic(), payload),
            timeout=opts.timeout,
        )


def idle(conn: Connection, rpc_q: MessageQueue, stop_event: EventType) -> None:
    """Block in `dequeue` with no traffic until the parent cancels us."""

    def watchdog() -> None:
        stop_event.wait()
        rpc_q.shutdown()

    stop_event.clear()
    thread = threading.Thread(target=watchdog, name="idle-watchdog", daemon=True)
    thread.start()
    conn.send(("idle_started",))
    try:
        while True:
            rpc_q.dequeue(indefinite=True)
    except RuntimeError as exc:
        if "cancelled" not in str(exc):
            raise
    finally:
        stop_event.set()
        thread.join()
        conn.send(("idle_done",))


# ---------------------------------------------------------------------------
# process group
# ---------------------------------------------------------------------------


class Group:
    """One executor plus `readers` workers, wired together by MessageQueues."""

    def __init__(
        self,
        readers: int,
        cpus: Sequence[int],
        opts: argparse.Namespace,
    ) -> None:
        self.ctx = multiprocessing.get_context("spawn")
        self.readers = readers
        self.opts = opts
        # Values the children read off the namespace (`--readers` is a list).
        opts.readers = readers
        opts.executor_cpu = cpus[0]
        opts.worker_cpus = list(cpus[1:])
        self.exec_conn, exec_child = self.ctx.Pipe()
        self.worker_conns: list[Connection] = []
        self.worker_procs: list[BaseProcess] = []
        self.stop_events: list[EventType] = []
        self.exec_proc = self.ctx.Process(
            target=executor_main, args=(exec_child, opts), name="executor"
        )
        for rank in range(readers):
            parent_conn, child_conn = self.ctx.Pipe()
            stop_event = self.ctx.Event()
            proc = self.ctx.Process(
                target=worker_main,
                args=(child_conn, rank, stop_event, opts),
                name=f"worker{rank}",
            )
            self.worker_conns.append(parent_conn)
            self.worker_procs.append(proc)
            self.stop_events.append(stop_event)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.exec_proc.start()
        for proc in self.worker_procs:
            proc.start()
        _, rpc_handle = expect(
            self.exec_conn, self.opts.handshake_timeout, "rpc_handle"
        )
        assert isinstance(rpc_handle, Handle)
        # The readers need the broadcast handle before they can build (and
        # therefore report) their private reply queues.
        for conn in self.worker_conns:
            conn.send(("rpc_handle", rpc_handle))
        reply_handles = []
        for conn in self.worker_conns:
            _, handle = expect(conn, self.opts.handshake_timeout, "reply_handle")
            reply_handles.append(handle)
        self.exec_conn.send(("reply_handles", reply_handles))
        for conn in [self.exec_conn, *self.worker_conns]:
            expect(conn, self.opts.handshake_timeout, "ready")

    def close(self) -> None:
        for conn in [self.exec_conn, *self.worker_conns]:
            conn.send(("exit",))
        for conn in [self.exec_conn, *self.worker_conns]:
            try:
                expect(conn, 15, "bye")
            except (RuntimeError, EOFError) as exc:
                log(f"shutdown warning: {exc}")
        for proc in [self.exec_proc, *self.worker_procs]:
            proc.join(15)
            if proc.is_alive():
                log(f"force-killing {proc.name}")
                proc.kill()
                proc.join(15)

    def kill(self) -> None:
        """Stop every child still running, e.g. after a failed start or phase."""
        for proc in self.procs():
            if proc.pid is not None and proc.is_alive():
                proc.kill()
                proc.join(15)

    def procs(self) -> list[BaseProcess]:
        return [self.exec_proc, *self.worker_procs]

    # -- scenarios ---------------------------------------------------------

    def _arm_workers(self, steps: int) -> None:
        for conn in self.worker_conns:
            conn.send(("consume", steps))
        for conn in self.worker_conns:
            expect(conn, self.opts.command_timeout, "consuming")

    def _start_writer(self, gap_ms: float, steps: int, mode: str) -> None:
        self.exec_conn.send(("produce", steps, gap_ms, mode))
        expect(self.exec_conn, self.opts.command_timeout, "measuring")

    def _dispatch(self, gap_ms: float, steps: int, mode: str) -> None:
        """Start a phase: every worker first, then the writer."""
        self._arm_workers(steps)
        self._start_writer(gap_ms, steps, mode)

    def _sample_cpu(self) -> dict[int, float]:
        return {p.pid: proc_cpu_seconds(p.pid) for p in self.procs() if p.pid}

    def _named_cpu(self, cpu: dict[int, float]) -> dict[str, float]:
        """Per-process CPU keyed by process name."""
        return {p.name: cpu.get(p.pid or -1, float("nan")) for p in self.procs()}

    def _probe(self, gap_ms: float, steps: int) -> float:
        """Run a discardable phase; return its mean per-step cost in seconds."""
        self._dispatch(gap_ms, steps, "probe")
        _, wall, done = expect(self.exec_conn, self.opts.command_timeout, "probe_end")
        return wall / done if done else 0.0

    def active(self, gap_ms: float) -> dict:
        """Calibrate, settle, then measure a lock-step step sequence."""
        opts = self.opts
        probe = max(1, opts.probe_steps)
        est = self._probe(gap_ms, probe)
        settle = probe
        if est > 0:
            settle = max(20, min(int(opts.probe_wall / est), opts.max_steps))
        est = self._probe(gap_ms, settle)
        steps = int(opts.target_wall / est) if est > 0 else probe
        steps = max(opts.min_steps, min(steps, opts.max_steps))

        self._arm_workers(steps)
        # Sample before the writer starts, so the window covers every step.
        cpu_start = self._sample_cpu()
        wall_start = time.monotonic()
        self._start_writer(gap_ms, steps, "measured")
        _, wall, rtts, oneways, handles, done = expect(
            self.exec_conn, opts.command_timeout, "measured_end"
        )
        wall_end = time.monotonic()
        span = wall_end - wall_start
        cpu_end = self._sample_cpu()
        cpu = {pid: cpu_end[pid] - cpu_start[pid] for pid in cpu_start}
        named = self._named_cpu(cpu)
        return {
            "scenario": f"gap_{gap_ms:g}ms",
            "gap_ms": gap_ms,
            "steps": done,
            "est_step_ms": est * 1000,
            "probe_steps": probe,
            "settle_steps": settle,
            "loop_wall_s": wall,
            "window_s": span,
            "steps_per_s": done / wall if wall > 0 else float("nan"),
            "rtt": summarize_us(rtts),
            "oneway": summarize_us(oneways),
            "handle": summarize_us(handles),
            "cpu_s": cpu,
            "cpu_by_name": named,
            "exec_cpu_s": named.get("executor", float("nan")),
            "workers_cpu_s": sum(v for k, v in named.items() if k != "executor"),
        }

    def idle(self, name: str) -> dict:
        """Sample CPU while every reader sits in `dequeue` with no traffic."""
        opts = self.opts
        for conn in self.worker_conns:
            conn.send(("idle",))
        for conn in self.worker_conns:
            expect(conn, opts.command_timeout, "idle_started")
        cpu_start = self._sample_cpu()
        wall_start = time.monotonic()
        time.sleep(opts.idle_seconds)
        cpu_end = self._sample_cpu()
        wall_end = time.monotonic()
        for event in self.stop_events:
            event.set()
        for conn in self.worker_conns:
            expect(conn, opts.command_timeout, "idle_done")
        span = wall_end - wall_start
        cpu = {pid: cpu_end[pid] - cpu_start[pid] for pid in cpu_start}
        named = self._named_cpu(cpu)
        return {
            "scenario": name,
            "gap_ms": None,
            "steps": 0,
            "window_s": span,
            "cpu_s": cpu,
            "cpu_by_name": named,
            "exec_cpu_s": named.get("executor", float("nan")),
            "workers_cpu_s": sum(v for k, v in named.items() if k != "executor"),
        }


# ---------------------------------------------------------------------------
# environment / metadata
# ---------------------------------------------------------------------------


def git_info(tree: Path) -> dict:
    git = "/usr/bin/git" if Path("/usr/bin/git").exists() else "git"

    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                [git, *args],
                cwd=tree,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return out.stdout.strip()
        except Exception:  # pragma: no cover - best effort metadata
            return "unknown"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "commit_date": run("log", "-1", "--format=%ci"),
        "subject": run("log", "-1", "--format=%s"),
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in status.splitlines() if line],
    }


def reader_grace_probe() -> object:
    """The spin grace a reader starts with, from the tree under test."""
    context = zmq.Context()
    addr = "inproc://shm-bench-grace-probe"
    publisher = context.socket(zmq.PUB)
    publisher.bind(addr)
    try:
        condition = shmb.SpinCondition(
            is_reader=True, context=context, notify_address=addr
        )
        return getattr(condition, "busy_loop_s", None)
    except Exception:  # pragma: no cover - diagnostics only
        return None
    finally:
        publisher.close()
        context.destroy(linger=0)


def tree_meta(variant: str) -> dict:
    module_file = Path(shmb.__file__).resolve()
    tree = module_file.parents[3]
    meta = {
        "variant": variant,
        "tree": str(tree),
        "module": str(module_file),
        "module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest()[:16],
        "python": sys.version.split()[0],
        "pyzmq": zmq.__version__,
        "libzmq": zmq.zmq_version(),
        "spinloop_ext_enabled": bool(getattr(shmb, "SPINLOOP_EXT_ENABLED", False)),
        "full_fence": getattr(shmb, "_full_fence", None) is not None,
        "recheck_interval_ms": getattr(shmb, "SHM_READER_RECHECK_INTERVAL_MS", None),
        "reader_busy_loop_s": reader_grace_probe(),
        "spin_condition_busy_loop_default": str(
            inspect.signature(shmb.SpinCondition.__init__)
            .parameters["busy_loop_s"]
            .default
        ),
        "mq_max_chunk_bytes_mb_env": getattr(
            vllm_envs, "VLLM_MQ_MAX_CHUNK_BYTES_MB", None
        ),
        "argv": " ".join(sys.argv),
    }
    meta.update(git_info(tree))
    return meta


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_matrix(opts: argparse.Namespace) -> list[dict]:
    meta = tree_meta(opts.variant)
    records: list[dict] = []
    readers_list = [int(x) for x in opts.readers.split(",") if x.strip()]
    gaps = [float(x) for x in opts.gaps.split(",") if x.strip()]
    for readers in readers_list:
        # Each group has stopped before the next starts, so cores are reused.
        cpus = pick_cpus(2 + readers)
        parent_cpu, group_cpus = cpus[0], cpus[1:]
        apply_affinity(parent_cpu)
        base: dict = dict(
            meta, readers=readers, repeat=opts.repeat, parent_cpu=parent_cpu
        )
        group = Group(readers, group_cpus, opts)
        try:
            group.start()
            log(f"readers={readers} traffic group up")
            for gap in gaps:
                record = dict(base, **group.active(gap))
                records.append(record)
                log(
                    f"readers={readers} {record['scenario']}: "
                    f"steps={record['steps']} "
                    f"rtt_p50={record['rtt']['p50']:.1f}us "
                    f"rtt_p99={record['rtt']['p99']:.1f}us "
                    f"oneway_p50={record['oneway']['p50']:.1f}us "
                    f"oneway_p99={record['oneway']['p99']:.1f}us "
                    f"cpu_exec={record['exec_cpu_s']:.2f}s "
                    f"cpu_workers={record['workers_cpu_s']:.2f}s"
                )
            if opts.idle_seconds > 0:
                record = dict(base, **group.idle("idle_post"))
                records.append(record)
                log(
                    f"readers={readers} idle_post: {record['window_s']:.1f}s "
                    f"cpu_exec={record['exec_cpu_s']:.2f}s "
                    f"cpu_workers={record['workers_cpu_s']:.2f}s"
                )
            group.close()
        finally:
            group.kill()

        if opts.idle_seconds > 0:
            virgin = Group(readers, group_cpus, opts)
            try:
                virgin.start()
                record = dict(base, **virgin.idle("idle_virgin"))
                records.append(record)
                log(
                    f"readers={readers} idle_virgin: {record['window_s']:.1f}s "
                    f"cpu_exec={record['exec_cpu_s']:.2f}s "
                    f"cpu_workers={record['workers_cpu_s']:.2f}s"
                )
                virgin.close()
            finally:
                virgin.kill()
    return records


def format_table(records: list[dict]) -> str:
    head = (
        f"{'variant':<8} {'rd':>2} {'scenario':<11} {'steps':>7} "
        f"{'rtt p50':>9} {'rtt p99':>9} {'rtt max':>10} "
        f"{'1way p50':>9} {'1way p99':>9} "
        f"{'cpu exec':>9} {'cpu wkers':>10} {'win s':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in records:
        if r["scenario"].startswith("idle"):
            lines.append(
                f"{r['variant']:<8} {r['readers']:>2} {r['scenario']:<11} "
                f"{'-':>7} {'-':>9} {'-':>9} {'-':>10} {'-':>9} {'-':>9} "
                f"{r['exec_cpu_s']:>8.2f}s {r['workers_cpu_s']:>9.2f}s "
                f"{r['window_s']:>6.1f}s"
            )
            continue
        rtt, oneway = r["rtt"], r["oneway"]
        lines.append(
            f"{r['variant']:<8} {r['readers']:>2} {r['scenario']:<11} "
            f"{r['steps']:>7} {rtt['p50']:>8.1f}u {rtt['p99']:>8.1f}u "
            f"{rtt['max']:>9.1f}u {oneway['p50']:>8.1f}u "
            f"{oneway['p99']:>8.1f}u "
            f"{r['exec_cpu_s']:>8.2f}s {r['workers_cpu_s']:>9.2f}s "
            f"{r['window_s']:>6.1f}s"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM's shared-memory MessageQueue.",
    )
    parser.add_argument("--variant", default="unknown", help="label for this tree")
    parser.add_argument("--readers", default="1,4", help="comma list, e.g. 1,4")
    parser.add_argument("--gaps", default="0,1,10", help="inter-step idle, ms")
    parser.add_argument(
        "--target-wall",
        type=float,
        default=6.0,
        help="target measured seconds per active configuration",
    )
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=600000)
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=100,
        help="steps of the first (coarse) calibration phase",
    )
    parser.add_argument(
        "--probe-wall",
        type=float,
        default=0.3,
        help="target seconds of the second (settling) calibration phase",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=10.0,
        help="seconds of the idle scenarios; 0 disables them",
    )
    parser.add_argument(
        "--payload-bytes", type=int, default=1024, help="RPC payload size"
    )
    parser.add_argument(
        "--max-chunk-bytes",
        type=int,
        default=vllm_envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024,
        help="ring chunk size, default = executor's env default",
    )
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-operation MessageQueue timeout, seconds",
    )
    parser.add_argument(
        "--handshake-timeout",
        type=float,
        default=180.0,
        help="budget for queue setup, seconds",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=900.0,
        help="budget for one phase, seconds",
    )
    parser.add_argument("--repeat", type=int, default=0, help="repeat index")
    parser.add_argument("--json-out", default=None, help="append JSONL records")
    opts = parser.parse_args(argv)
    opts.executor_cpu = None
    opts.worker_cpus = []
    return opts


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    try:
        records = run_matrix(opts)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print()
    print(format_table(records))
    if opts.json_out:
        with open(opts.json_out, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        log(f"appended {len(records)} records to {opts.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
