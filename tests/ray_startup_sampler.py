# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only sampler for slow Ray head-node startups in multi_process_parallel.

Run as a separate process while ray.init() is starting a local cluster (the
test process itself is blocked inside Ray's GCS port wait). Every second it
appends one JSON line to OUT with: host load and CPU use, the container's CPU
quota and throttling counters, CPU/IO/memory pressure stalls, the top CPU
consumers in the container, and the state of every Ray process; for the GCS
server, every thread's state, CPU time, kernel wait channel and syscall.

Usage: python ray_startup_sampler.py OUT [DURATION_S]
"""

import json
import os
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _kv(text: str | None) -> dict[str, str]:
    out = {}
    for line in (text or "").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def _stat(path: str) -> list[str] | None:
    text = _read(path)
    if text is None:
        return None
    # comm may contain spaces; it is wrapped in parentheses.
    head, _, rest = text.rpartition(")")
    return [head.split("(", 1)[0].strip(), head.split("(", 1)[1], *rest.split()]


def _cgroup() -> dict:
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):  # cgroup v2
        base = "/sys/fs/cgroup"
        return {
            "v": 2,
            "cpu.max": _read(f"{base}/cpu.max"),
            "cpu.stat": _kv(_read(f"{base}/cpu.stat")),
            "cpu.pressure": _read(f"{base}/cpu.pressure"),
            "io.pressure": _read(f"{base}/io.pressure"),
            "memory.pressure": _read(f"{base}/memory.pressure"),
            "nprocs": len((_read(f"{base}/cgroup.procs") or "").split()),
        }
    base = "/sys/fs/cgroup/cpu,cpuacct"
    if not os.path.isdir(base):
        base = "/sys/fs/cgroup/cpu"
    return {
        "v": 1,
        "cfs_quota_us": _read(f"{base}/cpu.cfs_quota_us"),
        "cfs_period_us": _read(f"{base}/cpu.cfs_period_us"),
        "cpu.stat": _kv(_read(f"{base}/cpu.stat")),
    }


def _procs() -> dict[int, tuple[str, str, int]]:
    """Map each visible pid to (state, cmdline, cpu ticks)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        st = _stat(f"/proc/{pid}/stat")
        if st is None:
            continue
        cmd = (_read(f"/proc/{pid}/cmdline") or "").replace("\0", " ").strip()
        out[int(pid)] = (st[2], cmd or f"[{st[1]}]", int(st[13]) + int(st[14]))
    return out


def _threads(pid: int) -> list[dict]:
    rows = []
    try:
        tids = sorted(os.listdir(f"/proc/{pid}/task"), key=int)
    except OSError:
        return rows
    for tid in tids:
        st = _stat(f"/proc/{pid}/task/{tid}/stat")
        if st is None:
            continue
        rows.append(
            {
                "tid": int(tid),
                "comm": st[1],
                "state": st[2],
                "cpu_s": round((int(st[13]) + int(st[14])) / CLK, 2),
                "wchan": _read(f"/proc/{pid}/task/{tid}/wchan"),
                "syscall": (_read(f"/proc/{pid}/task/{tid}/syscall") or "").split(" ")[
                    0
                ],
            }
        )
    return rows


def main() -> None:
    out_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    start = time.time()
    prev_cpu = None
    prev_ticks: dict[int, int] = {}
    with open(out_path, "a", buffering=1) as out:
        while time.time() - start < duration:
            now = time.time()
            cpu = [
                int(x)
                for x in (_read("/proc/stat") or "cpu 0").split("\n")[0].split()[1:]
            ]
            busy_pct = None
            if prev_cpu is not None:
                d = [a - b for a, b in zip(cpu, prev_cpu)]
                idle = d[3] + (d[4] if len(d) > 4 else 0)
                busy_pct = round(100 * (1 - idle / max(sum(d), 1)), 1)
            prev_cpu = cpu
            procs = _procs()
            top = sorted(
                (
                    (ticks - prev_ticks.get(pid, ticks), pid, state, cmd)
                    for pid, (state, cmd, ticks) in procs.items()
                ),
                reverse=True,
            )[:8]
            prev_ticks = {pid: ticks for pid, (_, _, ticks) in procs.items()}
            ray = []
            for pid, (state, cmd, ticks) in procs.items():
                if (
                    "ray/core/src" in cmd
                    or "ray/_private" in cmd
                    or "ray/dashboard" in cmd
                ):
                    row = {
                        "pid": pid,
                        "state": state,
                        "cpu_s": round(ticks / CLK, 2),
                        "cmd": cmd[:120],
                    }
                    if "gcs_server" in cmd:
                        row["threads"] = _threads(pid)
                    ray.append(row)
            sample = {
                "t": round(now - start, 1),
                "loadavg": _read("/proc/loadavg"),
                "host_busy_pct": busy_pct,
                "host_pressure": {
                    k: _read(f"/proc/pressure/{k}") for k in ("cpu", "io", "memory")
                },
                "cgroup": _cgroup(),
                "nprocs_visible": len(procs),
                "top_cpu_last_s": [
                    {"pid": pid, "ticks": dt, "state": s, "cmd": c[:100]}
                    for dt, pid, s, c in top
                ],
                "ray": ray,
            }
            out.write(json.dumps(sample) + "\n")
            time.sleep(max(0.0, 1.0 - (time.time() - now)))


if __name__ == "__main__":
    main()
