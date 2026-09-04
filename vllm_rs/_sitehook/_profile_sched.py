"""Wraps vllm.v1.core.sched.scheduler.Scheduler.update_from_output and
Scheduler.schedule with a cumulative-time accumulator; prints totals in
the EngineCore subprocess on shutdown.

Loaded by sitecustomize.py when VLLM_PROFILE_SCHED=1.
"""
import atexit
import os
import sys
import time


_totals = {"schedule": 0.0, "update_from_output": 0.0, "total_step": 0.0,
           "n_schedule": 0, "n_update": 0}


def install():
    import importlib
    try:
        sched_mod = importlib.import_module("vllm.v1.core.sched.scheduler")
    except Exception as e:
        print(f"[profile_sched] failed to import scheduler: {e}", file=sys.stderr)
        return

    SchedCls = sched_mod.Scheduler
    orig_schedule = SchedCls.schedule
    orig_update = SchedCls.update_from_output

    def wrapped_schedule(self, *a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return orig_schedule(self, *a, **kw)
        finally:
            _totals["schedule"] += time.perf_counter_ns() - t0
            _totals["n_schedule"] += 1

    def wrapped_update(self, *a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return orig_update(self, *a, **kw)
        finally:
            _totals["update_from_output"] += time.perf_counter_ns() - t0
            _totals["n_update"] += 1

    SchedCls.schedule = wrapped_schedule
    SchedCls.update_from_output = wrapped_update

    def report():
        s = _totals["schedule"] / 1e6
        u = _totals["update_from_output"] / 1e6
        ns = _totals["n_schedule"]
        nu = _totals["n_update"]
        print(
            f"[profile_sched pid={os.getpid()}] "
            f"schedule: {s:.1f} ms total / {ns} calls "
            f"({s*1000/max(ns,1):.1f} us/call avg); "
            f"update_from_output: {u:.1f} ms total / {nu} calls "
            f"({u*1000/max(nu,1):.1f} us/call avg)",
            file=sys.stderr,
            flush=True,
        )

    atexit.register(report)
    print(
        f"[profile_sched pid={os.getpid()}] installed scheduler timers",
        file=sys.stderr,
        flush=True,
    )
