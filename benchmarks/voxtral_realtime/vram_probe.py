# SPDX-License-Identifier: Apache-2.0
"""Sample VRAM, EngineCore host RSS, and vLLM KV-cache usage over time.

For long-run endurance tests: a flat VRAM + flat RSS + flat KV-usage proves the
unbounded-realtime re-anchoring does not leak (the sliding window + re-anchor
bound the KV; the only growing structure is the tiny mm_features list, which RSS
would expose if it mattered). Writes one CSV row per interval to stdout.

Usage: vram_probe.py [interval_s] [duration_s]   (Ctrl-C or duration to stop)
"""

import contextlib
import re
import subprocess
import sys
import time
import urllib.request


def nvidia_vram_mib() -> int:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return int(out[0]) if out else -1


def enginecore_rss_mib() -> float:
    # Sum RSS (KB) of all EngineCore worker processes.
    out = subprocess.run(
        ["ps", "-eo", "rss,args"], capture_output=True, text=True
    ).stdout
    tot = 0
    for line in out.splitlines():
        if "EngineCore" in line or "from multiprocessing.spawn" in line:
            m = re.match(r"\s*(\d+)\s", line)
            if m:
                tot += int(m.group(1))
    return round(tot / 1024, 1)


def host_avail_mib() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def metrics(port: int) -> tuple[float, float]:
    """Return (kv_cache_usage_perc, num_requests_running)."""
    try:
        body = (
            urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5)
            .read()
            .decode()
        )
    except Exception:
        return (-1.0, -1.0)
    kv = run = -1.0
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if "kv_cache_usage_perc" in line or "gpu_cache_usage_perc" in line:
            with contextlib.suppress(Exception):
                kv = float(line.rsplit(" ", 1)[1])
        elif "num_requests_running" in line:
            with contextlib.suppress(Exception):
                run = float(line.rsplit(" ", 1)[1])
    return (kv, run)


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    t0 = time.perf_counter()
    print(
        "elapsed_s,vram_mib,enginecore_rss_mib,kv_usage,running,host_avail_mib",
        flush=True,
    )
    while True:
        el = time.perf_counter() - t0
        if el > duration:
            break
        kv, run = metrics(port)
        print(
            f"{el:.0f},{nvidia_vram_mib()},{enginecore_rss_mib()},{kv:.4f},"
            f"{run:.0f},{host_avail_mib()}",
            flush=True,
        )
        time.sleep(interval)


if __name__ == "__main__":
    main()
