# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in profiling bootstrap, added to server PYTHONPATH by the benchmark."""

import faulthandler
import importlib
import json
import os
import signal
import threading
import time
from pathlib import Path

if directory := os.environ.get("KV_BENCH_CPU_PROFILE_DIR"):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text()
    start_ticks = int(stat[stat.rfind(")") + 2 :].split()[19])
    metadata = root / f"{pid}.json"
    profile = importlib.import_module("yappi")
    profile.set_clock_type("cpu")
    active = False
    profiled_threads = []
    cpu_started = cpu_seconds = None

    def record(state):
        threads = {t.native_id: t.name for t in threading.enumerate()}
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "start_ticks": start_ticks,
                    "state": state,
                    "profiler": "yappi",
                    "timer": "cpu",
                    "epoch": time.time(),
                    "threads": threads,
                    "profiled_threads": profiled_threads,
                    "process_cpu_seconds": cpu_seconds,
                }
            )
        )
        temporary.replace(metadata)

    def toggle(signum, frame):
        global active, profiled_threads, cpu_started, cpu_seconds
        if active:
            profile.stop()
            cpu_seconds = time.process_time() - cpu_started
            active = False
            profile.get_func_stats().save(str(root / f"{pid}.pstats"), type="pstat")
            profiled_threads = []
            directory = root / f"{pid}-threads"
            directory.mkdir(exist_ok=True)
            threads = {thread.ident: thread for thread in threading.enumerate()}
            for thread in profile.get_thread_stats():
                profile.get_func_stats(ctx_id=thread.id).save(
                    str(directory / f"{thread.id}.pstats"), type="pstat"
                )
                profiled_threads.append(
                    {
                        "context_id": thread.id,
                        "python_ident": thread.tid,
                        "native_id": threads[thread.tid].native_id
                        if thread.tid in threads
                        else None,
                        "name": thread.name,
                        "cpu_seconds": thread.ttot,
                    }
                )
            record("saved")
        else:
            profile.clear_stats()
            record("running")
            active = True
            cpu_started = time.process_time()
            profile.start(builtins=True, profile_threads=True)

    signal.signal(signal.SIGUSR2, toggle)
    stacks = Path(os.environ.get("KV_STACK_DUMP_DIR", root.parent / "stacks"))
    stacks.mkdir(parents=True, exist_ok=True)
    stack_file = (stacks / f"stacks-{pid}.log").open("a", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
    record("ready")
