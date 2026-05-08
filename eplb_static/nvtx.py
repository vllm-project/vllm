#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end automation for the offline-EPLB tuning workflow.

Given two user-typed commands — one to launch a vLLM server, one to launch
a `vllm bench serve` benchmark — this script walks through every stage of
the manual offline-EPLB tuning loop:

    1. preflight        ─ require 8 idle GPUs; create the output dir.
    2. count_steps      ─ run server (+ --enable-eplb) and the bench once;
                          curl /eplb_step_count to learn how many EPLB
                          steps the bench produces, then derive
                          expert_load_stats_interval = ceil(steps / 50)
                          so a follow-up stats run captures ~50 records.
    3. baseline_stats   ─ rerun bench with stats writing on; produces
                          baseline.jsonl.
    4. baseline_perf    ─ rerun the bench 5x (1 warmup + 4 measured);
                          appends the stdouts into baseline_perf.txt.
    5. baseline_html    ─ moe_report.py baseline.jsonl + baseline_perf.txt
                          → baseline.html.
    6. schedules        ─ tools/eplb/generate_static_mapping.py for X
                          ∈ {0,32,64} → <X>replicas_schedule.jsonl.
    7. per-replica runs ─ for each X: collect stats with the schedule
                          loaded, run perf 5x, generate HTML.

Between every server start the script invokes `killvllm` so leftover DP
workers from previous stages cannot collide with the new launch.

Usage:
    eplb_static/nvtx.py --dir sharegpt
"""

# === FOR AI AGENTS ===
# This script follows the conventions in ~/rules/script-conventions.mdc.
# If you modify this script, keep all comments up to date, comment any new
# logic you add, and preserve the colored output / log file / session
# summary behaviour described in that rule.

import argparse
import fcntl
import json
import math
import os
import pty
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

SCRIPT_NAME = "nvtx"

# ANSI colors. Plain log file gets the same lines without colors.
CYAN = "\033[1;36m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
NC = "\033[0m"

# Mutable globals — convenient because every helper logs and the alternative
# of threading them through every call would balloon the file.
LOG_FP = None
CURRENT_STAGE = "init"
SERVER_PROC = None  # Popen of the active server, if any (for signal cleanup).
SERVER_LOG_FP = None  # File handle for the server's stdout/stderr capture.
SERVER_TEE_THREAD = None  # Daemon thread teeing server output → terminal + log file.

# How many EPLB stats records we want at most in any single .jsonl. The
# bench-step counter from stage 2 is divided by this to get the interval.
TARGET_STATS_RECORDS = 50

# How many bench iterations during the perf stage. First one is a warmup;
# only the remaining MEASURED_RUNS are stable. Total = WARMUP_RUNS + MEASURED_RUNS.
WARMUP_RUNS = 1
MEASURED_RUNS = 4

# Replica counts to sweep in the offline-EPLB stage.
REPLICA_VARIANTS = [0, 32, 64]

# Server readiness timeout. Cold cache + 30B-class MoE warmup typically takes
# 6-9 minutes; allow 15 to give CI-like environments breathing room before we
# call the server "stuck" and abort.
SERVER_READY_TIMEOUT_S = 15 * 60


def _ts() -> str:
    """HH:MM:SS for log line prefixes."""
    return time.strftime("%H:%M:%S")


def _emit(prefix_color: str, label: str, msg: str) -> None:
    """Write one log line to terminal (colored) and log file (plain)."""
    t = _ts()
    plain = f"[{t} {CURRENT_STAGE}] {label}{msg}"
    colored = (
        f"{prefix_color}[{SCRIPT_NAME} {t} {CURRENT_STAGE}]{NC} "
        f"{label}{msg}"
    )
    print(colored, flush=True)
    if LOG_FP is not None:
        LOG_FP.write(plain + "\n")
        LOG_FP.flush()


def log(msg: str) -> None:
    _emit(CYAN, "", msg)


def ok(msg: str) -> None:
    _emit(GREEN, "", msg)


def warn(msg: str) -> None:
    _emit(YELLOW, "WARN: ", msg)


def err(msg: str) -> None:
    _emit(RED, "ERROR: ", msg)


def set_stage(name: str) -> None:
    global CURRENT_STAGE
    CURRENT_STAGE = name
    log(f"--- stage: {name} ---")


def _fmt_elapsed(seconds: float) -> str:
    """`5.3s`, `1m 12s`, `1h 02m 33s` — human-readable per-stage timer."""
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


@contextmanager
def stage_timer(name: str):
    """Log the stage banner, time it, and emit a closing line with the
    elapsed wall-clock so the user can see at a glance where the script
    is spending time. The closing line fires on both success and failure
    paths via ``finally``, so even an aborted stage gets timed."""
    set_stage(name)
    t0 = time.time()
    try:
        yield
    finally:
        log(f"stage `{name}` finished in {_fmt_elapsed(time.time() - t0)}")


# ------------------------------ preflight ------------------------------

def check_gpus() -> tuple[bool, str]:
    """Return (ok, message). Hard fails when nvidia-smi missing — there is
    no way to verify GPU availability without it.
    """
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not in PATH; cannot verify GPUs."
    try:
        gpu_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        return False, f"nvidia-smi failed: {e.stderr.strip() or e}"
    gpus = [r.strip() for r in gpu_out.strip().splitlines() if r.strip()]
    if len(gpus) != 8:
        return False, f"expected 8 GPUs, found {len(gpus)}"

    # Any compute app on any GPU = "busy". This catches lingering vllm
    # workers from a previous run that killvllm somehow missed.
    apps_out = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    active = [r.strip() for r in apps_out.strip().splitlines() if r.strip()]
    if active:
        sample = "; ".join(active[:5])
        more = f" (+{len(active)-5} more)" if len(active) > 5 else ""
        return False, f"GPUs not idle, {len(active)} compute proc(s): {sample}{more}"
    return True, "8 GPUs present and idle"


# ------------------------------ killvllm ------------------------------

def killvllm() -> None:
    """Run the user's `killvllm` shell function/alias. Wrapped via `bash -lc`
    so shell init (~/.bashrc / aliases / functions) is loaded; without this,
    aliases don't resolve in non-interactive subprocesses."""
    log("killvllm")
    try:
        subprocess.run(
            ["bash", "-lc", "killvllm"],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        warn("killvllm timed out after 60s")
    # Brief settle so freed sockets/CUDA contexts don't collide with the next
    # server's bind.
    time.sleep(2)


# ------------------------------ server ------------------------------

def _tee_server_output(stream, log_fp) -> None:
    """Drain the server's combined stdout/stderr, writing each chunk to BOTH
    the terminal (so the user can watch warmup live) AND the per-stage
    server log file (kept for post-mortem after the run). Runs in a daemon
    thread; exits when the pipe closes (= server exited) or read raises."""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_fp.write(chunk)
            log_fp.flush()
    except (OSError, ValueError):
        # Stream closed or already-closed file; nothing to do.
        pass


def start_server(cmd: str, log_path: Path) -> None:
    """Spawn a vllm server in its own process group (so SIGTERM to the group
    propagates to all DP/TP workers). Server stdout+stderr is teed to both
    the terminal (so the user sees warmup progress live) and ``log_path``
    (post-mortem record). The tee runs on a daemon thread."""
    global SERVER_PROC, SERVER_LOG_FP, SERVER_TEE_THREAD
    log(f"server: {cmd}")
    log(f"server log → {log_path} (also streamed to terminal)")
    SERVER_LOG_FP = open(log_path, "ab", buffering=0)
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}
    SERVER_PROC = subprocess.Popen(
        cmd, shell=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        bufsize=0,
    )
    SERVER_TEE_THREAD = threading.Thread(
        target=_tee_server_output,
        args=(SERVER_PROC.stdout, SERVER_LOG_FP),
        daemon=True,
        name="server-tee",
    )
    SERVER_TEE_THREAD.start()


def wait_for_ready(port: int = 8000, timeout_s: int = SERVER_READY_TIMEOUT_S) -> bool:
    """Poll /health every 5s. Returns False on timeout."""
    log(f"waiting for /health on :{port} (timeout {timeout_s}s)")
    deadline = time.time() + timeout_s
    started = time.time()
    last_err = None
    while time.time() < deadline:
        # Detect server crash early — no point waiting for /health from a
        # process that already exited.
        if SERVER_PROC is not None and SERVER_PROC.poll() is not None:
            err(f"server process exited prematurely (code {SERVER_PROC.returncode})")
            return False
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=5,
            ) as resp:
                if resp.status == 200:
                    ok(f"server ready in {int(time.time() - started)}s")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(5)
    err(f"server not ready within {timeout_s}s; last error: {last_err}")
    return False


def stop_server() -> None:
    """SIGTERM the server's process group, wait, escalate to SIGKILL, then
    `killvllm` as belt-and-suspenders. Safe to call when no server is up."""
    global SERVER_PROC, SERVER_LOG_FP, SERVER_TEE_THREAD
    if SERVER_PROC is None:
        killvllm()
        return
    log("stopping server")
    try:
        os.killpg(os.getpgid(SERVER_PROC.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        SERVER_PROC.wait(timeout=30)
    except subprocess.TimeoutExpired:
        warn("server did not exit in 30s after SIGTERM; sending SIGKILL")
        try:
            os.killpg(os.getpgid(SERVER_PROC.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            SERVER_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # Drain remaining server output before closing the log file. The tee
    # thread exits on its own when the pipe closes (= server exited).
    if SERVER_TEE_THREAD is not None:
        SERVER_TEE_THREAD.join(timeout=5)
        SERVER_TEE_THREAD = None
    if SERVER_LOG_FP is not None:
        SERVER_LOG_FP.close()
        SERVER_LOG_FP = None
    SERVER_PROC = None
    killvllm()


def restart_server(cmd: str, log_path: Path) -> None:
    """killvllm → start_server → wait_for_ready. Raises RuntimeError on
    readiness failure (caller is expected to clean up via try/except +
    stop_server in main)."""
    killvllm()
    start_server(cmd, log_path)
    if not wait_for_ready():
        raise RuntimeError("server failed to become ready")


# ------------------------------ HTTP helpers ------------------------------

def get_eplb_step_count(port: int = 8000) -> int:
    """GET /eplb_step_count → integer step count. Endpoint is the one we
    added in this PR; if the server is missing it (older build), this will
    raise HTTPError 404, which is then surfaced verbatim in the summary."""
    with urllib.request.urlopen(
        f"http://localhost:{port}/eplb_step_count", timeout=10,
    ) as resp:
        return int(json.loads(resp.read())["step_count"])


# ------------------------------ command rewriting ------------------------------

def with_enable_eplb(cmd: str) -> str:
    """Append --enable-eplb if not already there. Idempotent so the user is
    free to include it themselves."""
    if "--enable-eplb" in cmd.split():
        return cmd
    return f"{cmd} --enable-eplb"


def with_eplb_config(cmd: str, config: dict) -> str:
    """Append --eplb-config '<json>'. The single-quoted JSON survives bash
    word-splitting; embedded single quotes in JSON are impossible because
    json.dumps emits only `\"`."""
    cmd = with_enable_eplb(cmd)
    return f"{cmd} --eplb-config '{json.dumps(config)}'"


# ------------------------------ benchmark ------------------------------

def run_benchmark(cmd: str, perf_file: Path | None = None) -> tuple[bool, bytes]:
    """Run the user's bench command, mirroring its stdout to our terminal
    LIVE (via a pty so tqdm refreshes work) AND capturing it. Optionally
    appends the captured bytes to `perf_file`. Returns (success, captured).
    """
    log(f"bench: {cmd}")
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}

    # PTY-based pipe: subprocess connected to a pseudo-tty, so vllm bench
    # serve's tqdm progress bar refreshes as expected. Plain `stdout=PIPE`
    # would block-buffer at 4-8KB and we'd see nothing until the bench
    # finished — useless for monitoring a 15s run.
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        cmd, shell=True, env=env,
        stdin=slave, stdout=slave, stderr=slave,
        close_fds=True,
    )
    os.close(slave)

    captured = bytearray()
    try:
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:
                break  # pty closed when the child exited
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            captured.extend(data)
    finally:
        os.close(master)
    proc.wait()
    captured = bytes(captured)
    if proc.returncode != 0:
        err(f"benchmark exited with code {proc.returncode}")
        return False, captured
    if perf_file is not None:
        with open(perf_file, "ab") as f:
            f.write(captured)
    return True, captured


# ------------------------------ stages ------------------------------

def compute_interval(steps: int) -> int:
    """Round up so that the resulting JSONL has at most TARGET_STATS_RECORDS
    eplb_load_stats records when the rearrangement counter doesn't reset
    mid-bench. ``max(1, …)`` guards against pathological inputs."""
    return max(1, math.ceil(steps / TARGET_STATS_RECORDS))


def stage_count_steps(server_cmd: str, bench_cmd: str, dir_path: Path) -> int:
    """Run server with --enable-eplb only, verify /eplb_step_count==0 before
    the bench, run the bench once, return the post-bench step count. The
    interval used by all subsequent collect_stats stages is computed from
    this number."""
    with stage_timer("count_steps"):
        cmd = with_enable_eplb(server_cmd)
        restart_server(cmd, dir_path / "server_count_steps.log")
        try:
            before = get_eplb_step_count()
            if before != 0:
                raise RuntimeError(
                    f"expected eplb_step_count == 0 before bench, got {before}"
                )
            ok("pre-bench eplb_step_count == 0")
            if not run_benchmark(bench_cmd)[0]:
                raise RuntimeError("benchmark failed")
            steps = get_eplb_step_count()
            ok(f"post-bench eplb_step_count = {steps}")
        finally:
            stop_server()
    return steps


def stage_collect_stats(
    server_cmd: str, bench_cmd: str, dir_path: Path,
    name: str, interval: int,
) -> None:
    """Run a single bench with stats writing on. ``name`` controls both the
    output JSONL filename and whether an offline mapping is loaded:
      - "baseline"        → no initial mapping (default identity).
      - "<X>replicas"     → loads <X>replicas_schedule.jsonl.
    """
    with stage_timer(f"collect_stats:{name}"):
        config: dict = {
            "expert_load_stats_path": str(dir_path / f"{name}.jsonl"),
            "expert_load_stats_interval": interval,
        }
        if name != "baseline":
            config["initial_mapping_path"] = str(dir_path / f"{name}_schedule.jsonl")
        cmd = with_eplb_config(server_cmd, config)
        restart_server(cmd, dir_path / f"server_collect_{name}.log")
        try:
            if not run_benchmark(bench_cmd)[0]:
                raise RuntimeError("benchmark failed")
        finally:
            stop_server()


def stage_perf(
    server_cmd: str, bench_cmd: str, dir_path: Path, name: str,
) -> Path:
    """Run the bench WARMUP_RUNS+MEASURED_RUNS times against a fresh
    server and append every stdout to ``<name>_perf.txt``. Returns the
    perf file path.

    Server config differs by ``name``:

      - "baseline"        → `--enable-eplb` only. Identity placement,
                            online rebalancing OFF (it is OFF by default
                            on this branch — `eplb_config.enable_online`
                            defaults to False), no initial mapping.
      - "<X>replicas"     → also passes
                            `--eplb-config '{"initial_mapping_path": ".../<X>replicas_schedule.jsonl"}'`
                            so the offline mapping is loaded. Without
                            this the X-replicas perf is indistinguishable
                            from the baseline perf — same server config.
                            (`num_redundant_experts` is auto-read from
                            the schedule file by `parallel.py`.)
    """
    with stage_timer(f"perf:{name}"):
        if name == "baseline":
            cmd = with_enable_eplb(server_cmd)
        else:
            cmd = with_eplb_config(server_cmd, {
                "initial_mapping_path": str(
                    dir_path / f"{name}_schedule.jsonl"
                ),
            })
        restart_server(cmd, dir_path / f"server_perf_{name}.log")
        perf_file = dir_path / f"{name}_perf.txt"
        try:
            total = WARMUP_RUNS + MEASURED_RUNS
            for i in range(total):
                tag = " (warmup)" if i < WARMUP_RUNS else ""
                log(f"perf bench {i+1}/{total}{tag}")
                t_iter = time.time()
                if not run_benchmark(bench_cmd, perf_file)[0]:
                    raise RuntimeError(f"perf bench iteration {i+1} failed")
                log(
                    f"perf bench {i+1}/{total}{tag} done in "
                    f"{_fmt_elapsed(time.time() - t_iter)}"
                )
        finally:
            stop_server()
    return perf_file


def stage_html_report(
    jsonl: Path, perf_file: Path | None, output: Path,
) -> None:
    """Invoke moe_report.py with `.venv/bin/python` (= sys.executable)."""
    with stage_timer(f"html_report:{output.name}"):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "moe_report.py"),
            str(jsonl), "-o", str(output),
        ]
        if perf_file is not None:
            cmd.extend(["--perf-data", str(perf_file)])
        log("running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def stage_generate_schedule(
    jsonl: Path, output: Path, num_redundant: int, repo_root: Path,
) -> None:
    with stage_timer(f"schedule:{num_redundant}replicas"):
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "eplb" / "generate_static_mapping.py"),
            "--stats-path", str(jsonl),
            "--output", str(output),
            "--num-redundant-experts", str(num_redundant),
        ]
        log("running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)


# ------------------------------ resume infrastructure ------------------------------

# State file lives at <dir>/.nvtx_state.json. It's the single source of truth
# for resume: server/bench commands, the count_steps result, and the list of
# stages that have already completed. When the file is missing on a non-empty
# directory we fall back to "reconstruct" mode (inspect output files,
# re-derive interval from baseline.jsonl).
STATE_FILENAME = ".nvtx_state.json"
LOCK_FILENAME = ".nvtx.lock"

# Module-level lock handle. Kept open for the lifetime of the process so
# the OS-level fcntl lock survives until exit. Closing it (or letting it
# be garbage-collected) releases the lock — that's exactly what we want
# on normal exit / signal cleanup.
_LOCK_FH = None


def acquire_lock(dir_path: Path) -> None:
    """Take an exclusive non-blocking ``fcntl.flock`` on
    ``<dir>/.nvtx.lock`` so two ``nvtx.py`` processes can't trample
    each other's state file. We pick the lock file path so it stays
    alongside the state — ``flock`` is per-fd, so the file just needs
    to exist; we never read/write its contents."""
    global _LOCK_FH
    lock_path = dir_path / LOCK_FILENAME
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        sys.exit(
            f"error: another nvtx.py is already running on {dir_path}\n"
            f"(lock: {lock_path}). Wait for it to finish, or kill it "
            "and remove the lock file if it's truly stale."
        )
    _LOCK_FH = fh


def _validate_stats_jsonl(path: Path) -> bool:
    """A complete EPLB stats JSONL: file exists, every non-empty line
    parses, contains an ``eplb_load_meta`` record AND ≥1
    ``eplb_load_stats`` record. The "every line parses" check is what
    catches a server killed mid-write (the partial last line fails)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    has_meta = False
    has_stats = False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False
                rt = rec.get("record_type")
                if rt == "eplb_load_meta":
                    has_meta = True
                elif rt == "eplb_load_stats":
                    has_stats = True
    except OSError:
        return False
    return has_meta and has_stats


def _validate_schedule_jsonl(path: Path) -> bool:
    """A complete offline mapping JSONL has ≥1 ``eplb_initial_mapping``
    record. ``generate_static_mapping.py`` writes the record in a single
    ``open(..., 'w')`` block, so it's effectively atomic on local
    filesystems — no partial-line concern."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if rec.get("record_type") == "eplb_initial_mapping":
                    return True
    except OSError:
        return False
    return False


def _validate_perf_txt(path: Path) -> bool:
    """A complete perf file ends with the ``=`` × 50 closing bar of the
    final ``Serving Benchmark Result`` block. If the bench was killed
    mid-print, the last block is truncated and there's no closing bar
    near the end — we look at the last ~500 bytes as our test."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 500))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    closing = "=" * 50
    # Inspect the last few lines so trailing blank lines / shell prompts
    # don't fool us.
    return any(closing in ln for ln in tail.splitlines()[-5:])


def _validate_html(path: Path) -> bool:
    """``moe_report.py`` calls ``Path.write_text(html)`` once, so an
    interrupted run leaves either no file or a complete one ending in
    ``</html>``. Check the closing tag in the last 200 bytes."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 200))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return "</html>" in tail.lower()


@dataclass
class Stage:
    """One unit of work in the pipeline.

    ``name`` is what we print in plan output and store in
    ``completed_stages``. ``outputs`` is the set of files this stage
    creates — used in reconstruct mode to detect "already done" without
    a state file. ``run`` is a zero-arg callable that does the work.
    ``is_complete`` returns True iff every output is present AND
    structurally valid (catches partial files from a killed server). The
    default checks just existence — supply the real validator at
    construction time when there's a known shape to verify."""
    name: str
    outputs: list[Path]
    run: Callable[[], None | int]
    is_complete: Callable[[], bool] = field(
        default_factory=lambda: (lambda: True)
    )


def _build_plan(
    server_cmd: str, bench_cmd: str, dir_path: Path, repo_root: Path,
    get_interval: Callable[[], int],
) -> list[Stage]:
    """Construct the linear list of stages. ``get_interval`` is a closure
    so collect_stats stages can read the up-to-date ``interval`` value
    that count_steps populated, even though the plan is built once at the
    start (before count_steps has run)."""
    baseline_jsonl = dir_path / "baseline.jsonl"
    baseline_perf = dir_path / "baseline_perf.txt"
    baseline_html = dir_path / "baseline.html"
    plan: list[Stage] = [
        Stage(
            name="count_steps",
            # No persistent file output; completion is tracked in the
            # state file (interval / step_count fields). The default
            # validator (always True) is overridden in the resume path
            # to require interval being known.
            outputs=[],
            run=lambda: stage_count_steps(server_cmd, bench_cmd, dir_path),
        ),
        Stage(
            name="collect_stats:baseline",
            outputs=[baseline_jsonl],
            run=lambda: stage_collect_stats(
                server_cmd, bench_cmd, dir_path, "baseline", get_interval(),
            ),
            is_complete=lambda p=baseline_jsonl: _validate_stats_jsonl(p),
        ),
        Stage(
            name="perf:baseline",
            outputs=[baseline_perf],
            run=lambda: stage_perf(server_cmd, bench_cmd, dir_path, "baseline"),
            is_complete=lambda p=baseline_perf: _validate_perf_txt(p),
        ),
        Stage(
            name="html_report:baseline.html",
            outputs=[baseline_html],
            run=lambda: stage_html_report(
                baseline_jsonl, baseline_perf, baseline_html,
            ),
            is_complete=lambda p=baseline_html: _validate_html(p),
        ),
    ]
    # Schedules are independent of each other and only depend on baseline.jsonl.
    for x in REPLICA_VARIANTS:
        sched = dir_path / f"{x}replicas_schedule.jsonl"
        plan.append(Stage(
            name=f"schedule:{x}replicas",
            outputs=[sched],
            run=lambda x=x, sched=sched: stage_generate_schedule(
                baseline_jsonl, sched, x, repo_root,
            ),
            is_complete=lambda p=sched: _validate_schedule_jsonl(p),
        ))
    # Per-replica trio: collect (stats with schedule loaded) → perf → html.
    for x in REPLICA_VARIANTS:
        name = f"{x}replicas"
        repl_jsonl = dir_path / f"{name}.jsonl"
        repl_perf = dir_path / f"{name}_perf.txt"
        repl_html = dir_path / f"{name}.html"
        plan.append(Stage(
            name=f"collect_stats:{name}",
            outputs=[repl_jsonl],
            run=lambda name=name: stage_collect_stats(
                server_cmd, bench_cmd, dir_path, name, get_interval(),
            ),
            is_complete=lambda p=repl_jsonl: _validate_stats_jsonl(p),
        ))
        plan.append(Stage(
            name=f"perf:{name}",
            outputs=[repl_perf],
            run=lambda name=name: stage_perf(
                server_cmd, bench_cmd, dir_path, name,
            ),
            is_complete=lambda p=repl_perf: _validate_perf_txt(p),
        ))
        plan.append(Stage(
            name=f"html_report:{name}.html",
            outputs=[repl_html],
            run=lambda name=name, j=repl_jsonl, p=repl_perf, h=repl_html:
                stage_html_report(j, p, h),
            is_complete=lambda p=repl_html: _validate_html(p),
        ))
    return plan


def _read_state(state_path: Path) -> dict | None:
    """Return the parsed state dict or None when missing/corrupt."""
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"state file present but unreadable ({exc}); ignoring")
        return None


def _write_state(state_path: Path, state: dict) -> None:
    """Atomically replace the state file. Use a tmp+rename pair so a crash
    mid-write can't leave a half-written JSON document on disk."""
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(state_path)


def _infer_interval_from_jsonl(path: Path) -> int | None:
    """Recover ``expert_load_stats_interval`` from a stats JSONL by reading
    the first two ``eplb_load_stats`` records and taking the difference of
    their ``step`` fields. vLLM writes records at expert_rearrangement_step
    values 0, interval, 2*interval, … so the first delta IS the interval.
    Returns ``None`` when the file has fewer than two stats records."""
    steps: list[int] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("record_type") == "eplb_load_stats" and "step" in rec:
                    steps.append(int(rec["step"]))
                    if len(steps) >= 2:
                        break
    except OSError:
        return None
    if len(steps) < 2:
        return None
    delta = steps[1] - steps[0]
    return delta if delta > 0 else None


def _expected_filenames(dir_path: Path) -> set[str]:
    """Set of every filename a successful nvtx.py run can produce. Used by
    reconstruct mode to decide whether a non-empty directory looks like
    one of ours (vs random user content) before touching anything."""
    names = {"nvtx_log.txt", STATE_FILENAME}
    for n in (
        "baseline.jsonl", "baseline_perf.txt", "baseline.html",
        "server_count_steps.log",
        "server_collect_baseline.log",
        "server_perf_baseline.log",
    ):
        names.add(n)
    for x in REPLICA_VARIANTS:
        for n in (
            f"{x}replicas.jsonl", f"{x}replicas_schedule.jsonl",
            f"{x}replicas_perf.txt", f"{x}replicas.html",
            f"server_collect_{x}replicas.log",
            f"server_perf_{x}replicas.log",
        ):
            names.add(n)
    return names


def _looks_like_nvtx_dir(dir_path: Path) -> bool:
    """Heuristic for reconstruct mode: at least one entry must match
    ``_expected_filenames``. The previous run's nvtx_log.txt is a strong
    hint; baseline.jsonl etc. are equally clear signals."""
    expected = _expected_filenames(dir_path)
    for entry in dir_path.iterdir():
        if entry.name in expected:
            return True
    return False


def _detect_completed_from_files(plan: list[Stage], interval: int | None) -> int:
    """Walk plan in order; the first stage whose ``is_complete`` returns
    False is where we resume. Special-cases ``count_steps``: no file
    output, so it counts as done iff ``interval`` was recovered (e.g.
    inferred from baseline.jsonl). Otherwise count_steps is forced to
    re-run.

    Crucially this calls ``is_complete`` (the per-output structural
    validator) rather than naive ``exists()``, so a half-written JSONL
    or a perf .txt missing its closing ``=`` × 50 bar is correctly
    classified as "incomplete → re-run from scratch"."""
    for i, stage in enumerate(plan):
        if stage.name == "count_steps":
            if interval is not None:
                continue
            return i
        if not stage.is_complete():
            return i
    return len(plan)  # everything done


# ------------------------------ summary ------------------------------

def session_summary(start_time: float, dir_path: Path | None, status: str) -> None:
    """Cyan-banded summary block printed both to terminal and log file.
    Always called once on exit (success or failure)."""
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    bar = "━" * 64
    log_path = dir_path / "nvtx_log.txt" if dir_path is not None else "n/a"
    lines = [
        bar,
        "  SESSION SUMMARY",
        bar,
        f"  Status:        {status}",
        f"  Output dir:    {dir_path or 'n/a'}",
        f"  Log file:      {log_path}",
        f"  Total runtime: {mins}m {secs}s",
        bar,
    ]
    for ln in lines:
        print(f"{CYAN}{ln}{NC}")
        if LOG_FP is not None:
            LOG_FP.write(ln + "\n")
    if LOG_FP is not None:
        LOG_FP.flush()


# ------------------------------ main ------------------------------

class _Parser(argparse.ArgumentParser):
    """Print full help on any usage error (mirrors script-conventions.mdc
    pattern)."""
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _ask(prompt: str) -> str:
    """Prompt the user for a (possibly multi-line) command. Submission is
    signalled by a single empty line.

    Lines are concatenated **without inserting any separator** — only
    trailing ``\\`` continuations are stripped. This is what makes both
    common paste cases work:

    1. *Terminal soft-wrap paste*: the original command was logically
       one line, but the user's terminal copied it back with a real
       ``\\n`` injected mid-token (e.g. ``--enable-exper\\nt-parallel``).
       Naive space-join produces the broken ``--enable-exper t-parallel``;
       concatenation restores the intended ``--enable-expert-parallel``.

    2. *Proper multi-line with `\\` continuations*: the leading
       whitespace of each subsequent line is preserved by ``input()``,
       so it serves as the natural token separator after the ``\\`` is
       stripped. ``cmd \\\\<NL>    --foo`` round-trips as ``cmd    --foo``.

    A typed-but-not-wrapped multi-line command without continuations
    won't reassemble correctly — but that pattern doesn't appear in
    real shell usage, so we don't try to second-guess it."""
    print(f"{CYAN}[{SCRIPT_NAME}] {prompt}{NC}")
    print(
        f"{CYAN}[{SCRIPT_NAME}] (paste/type the command; "
        f"submit with an empty line):{NC}"
    )
    parts: list[str] = []
    while True:
        try:
            raw = input()
        except EOFError:
            break
        # input() already strips the trailing newline. We DON'T rstrip
        # whitespace here — leading/trailing spaces of each line are
        # what carries the inter-token separator across line boundaries
        # (see case 2 above).
        if not raw.strip():
            if parts:
                break
            print(
                f"{RED}[{SCRIPT_NAME}] empty input; please type a command:{NC}"
            )
            continue
        # Trailing `\` is the shell continuation marker — drop it. Any
        # whitespace that immediately preceded it survives, so a
        # well-formatted ``cmd \\<NL>    --foo`` still concatenates
        # with the right separator.
        s = raw[:-1] if raw.endswith("\\") else raw
        parts.append(s)
    return "".join(parts).strip()


def _confirm(prompt: str) -> bool:
    """Yes/no prompt — only an exact ``y`` / ``Y`` (or ``yes``) counts as
    consent; everything else (Enter, n, anything) is "no". We log the raw
    input verbatim so the audit trail records what the user typed."""
    print(f"{CYAN}[{SCRIPT_NAME}] {prompt} [y/N]{NC}")
    try:
        ans = input().strip().lower()
    except EOFError:
        ans = ""
    log(f"user answered: {ans!r}")
    return ans in {"y", "yes"}


def main() -> None:
    global LOG_FP
    parser = _Parser(
        description=(
            "Automate the end-to-end offline-EPLB tuning loop "
            "(count steps → baseline stats → baseline perf → "
            "schedules for {0,32,64} → per-replica stats+perf → HTML).\n"
            "\n"
            "Resume: if --dir already exists and looks like a previous "
            "nvtx.py run, the script reads/derives state, prints a plan, "
            "and continues from the first incomplete stage after Y/n "
            "confirmation."
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=80,
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help=(
            "Output directory. If it doesn't exist, a fresh run starts. "
            "If it exists with a `.nvtx_state.json` (a previous run's state "
            "file) or recognizable output files, a resume is offered."
        ),
    )
    args = parser.parse_args()

    dir_path: Path = args.dir.resolve()
    state_path = dir_path / STATE_FILENAME

    # ---- mode detection -------------------------------------------------
    # Three possible modes:
    #   fresh        : dir doesn't exist, OR exists but is empty.
    #   resume       : dir + state file → trust state.
    #   reconstruct  : dir non-empty without state file → infer from files.
    # Anything else (dir is a regular file, dir non-empty but unrecognized)
    # exits with a diagnostic before any work happens.
    mode: str
    saved_state: dict | None = None

    if dir_path.exists():
        if not dir_path.is_dir():
            sys.exit(f"error: {dir_path} exists but is not a directory")
        saved_state = _read_state(state_path)
        if saved_state is not None:
            mode = "resume"
        elif any(dir_path.iterdir()):
            # Non-empty without state file. Will validate "is this ours?"
            # below — refuse early so we don't even open the log file
            # inside someone else's directory.
            if not _looks_like_nvtx_dir(dir_path):
                sys.exit(
                    f"error: directory {dir_path} is non-empty but does not "
                    "look like a previous nvtx.py run "
                    f"(no {STATE_FILENAME} and no recognizable output "
                    "files). Refusing to overwrite. Use a fresh path."
                )
            mode = "reconstruct"
        else:
            mode = "fresh"
    else:
        dir_path.mkdir(parents=True)
        mode = "fresh"

    # ---- exclusive lock so two scripts on the same dir can't fight ----
    acquire_lock(dir_path)

    # ---- log file (append mode keeps history across resumes) ------------
    LOG_FP = open(dir_path / "nvtx_log.txt", "a")
    LOG_FP.write(
        f"\n# nvtx.py session started at "
        f"{datetime.now().isoformat(timespec='seconds')} (mode={mode})\n"
    )

    start_time = time.time()

    def _sig_handler(signum, frame):
        warn(f"caught signal {signum}; cleaning up")
        try:
            stop_server()
        except Exception as exc:
            warn(f"stop_server during signal handler raised: {exc}")
        session_summary(start_time, dir_path, f"INTERRUPTED (signal {signum})")
        sys.exit(130)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    set_stage("init")
    log(f"output directory: {dir_path}")
    log(f"mode: {mode}")

    gpus_ok, gpu_msg = check_gpus()
    if not gpus_ok:
        err(gpu_msg)
        session_summary(start_time, dir_path, f"FAILED preflight: {gpu_msg}")
        sys.exit(1)
    ok(gpu_msg)

    # tools/eplb/generate_static_mapping.py lives at <repo>/tools/eplb/.
    # nvtx.py lives at <repo>/eplb_static/. Walk one level up.
    repo_root = Path(__file__).resolve().parent.parent

    # ---- resolve cmds + interval according to mode ---------------------
    server_cmd: str
    bench_cmd: str
    completed_stages: list[str]
    interval: int | None
    step_count: int | None

    if mode == "resume":
        assert saved_state is not None
        server_cmd = saved_state["server_cmd"]
        bench_cmd = saved_state["bench_cmd"]
        completed_stages = list(saved_state.get("completed_stages") or [])
        interval = saved_state.get("interval")
        step_count = saved_state.get("step_count")
        log(f"resumed server cmd: {server_cmd}")
        log(f"resumed bench cmd:  {bench_cmd}")
        log(
            f"resumed step_count={step_count}, interval={interval}, "
            f"completed_stages={len(completed_stages)}"
        )
    elif mode == "reconstruct":
        # Try to recover interval from baseline.jsonl. If it's missing or
        # too short, count_steps will simply re-run.
        baseline_jsonl = dir_path / "baseline.jsonl"
        interval = None
        step_count = None
        if baseline_jsonl.exists():
            interval = _infer_interval_from_jsonl(baseline_jsonl)
            if interval is not None:
                log(
                    f"inferred expert_load_stats_interval={interval} from "
                    f"step deltas in {baseline_jsonl.name}"
                )
            else:
                warn(
                    f"could not infer interval from {baseline_jsonl.name} "
                    "(fewer than 2 stats records); count_steps will re-run"
                )
        completed_stages = []  # will be filled by file inspection below
        set_stage("input")
        log(
            "reconstruct mode: please re-enter the original commands so "
            "the remaining stages run with the same configuration."
        )
        server_cmd = _ask(
            "Enter the `vllm serve` command (without --enable-eplb / "
            "--eplb-config; the script appends those itself):"
        )
        bench_cmd = _ask(
            "Enter the `vllm bench serve` command (HF_HUB_OFFLINE=0 will "
            "be added automatically):"
        )
    else:  # fresh
        completed_stages = []
        interval = None
        step_count = None
        set_stage("input")
        server_cmd = _ask(
            "Enter the `vllm serve` command (without --enable-eplb / "
            "--eplb-config; the script appends those itself):"
        )
        bench_cmd = _ask(
            "Enter the `vllm bench serve` command (HF_HUB_OFFLINE=0 will "
            "be added automatically):"
        )
    log(f"server cmd: {server_cmd}")
    log(f"bench cmd:  {bench_cmd}")

    # ---- build plan, decide first incomplete ---------------------------
    # ``interval`` is captured by reference via the closure. count_steps
    # writes to it (through the outer scope assignment in the loop below).
    interval_ref = {"value": interval}

    def get_interval() -> int:
        v = interval_ref["value"]
        if v is None:
            raise RuntimeError(
                "internal: interval requested before count_steps ran"
            )
        return v

    plan = _build_plan(server_cmd, bench_cmd, dir_path, repo_root, get_interval)

    if mode == "resume":
        # Trust-but-verify: a stage marked done in the state file must
        # ALSO have valid on-disk outputs. If they don't match (someone
        # deleted a file by hand, or the OS killed the writer right
        # before fsync), force a re-run of that stage and trim the
        # completed list to match.
        completed_set = set(completed_stages)
        first_incomplete = len(plan)
        for i, s in enumerate(plan):
            if s.name not in completed_set:
                first_incomplete = i
                break
            if s.name == "count_steps":
                if interval_ref["value"] is None:
                    warn(
                        "state says count_steps done but interval missing; "
                        "will re-run"
                    )
                    first_incomplete = i
                    break
                continue
            if not s.is_complete():
                warn(
                    f"state says `{s.name}` done but outputs are "
                    "missing/invalid; will re-run that stage"
                )
                first_incomplete = i
                break
        # Truncate completed_stages so save_state() reflects reality.
        completed_stages = [s.name for s in plan[:first_incomplete]]
    else:
        # Fresh or reconstruct: derive completion from on-disk files.
        first_incomplete = _detect_completed_from_files(plan, interval_ref["value"])
        completed_stages = [s.name for s in plan[:first_incomplete]]

    # ---- present the plan ----------------------------------------------
    set_stage("plan")
    if first_incomplete >= len(plan):
        ok("all stages already completed; nothing to do")
        # Still write the state file so future runs see a clean record.
        _write_state(state_path, {
            "version": 1,
            "started_at": (saved_state or {}).get(
                "started_at",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "server_cmd": server_cmd,
            "bench_cmd": bench_cmd,
            "step_count": step_count,
            "interval": interval_ref["value"],
            "completed_stages": [s.name for s in plan],
        })
        session_summary(start_time, dir_path, "OK (already completed)")
        return

    log("execution plan:")
    for i, s in enumerate(plan):
        if i < first_incomplete:
            tag = f"{GREEN}[done]   {NC}"
        elif i == first_incomplete:
            tag = f"{YELLOW}[resume] {NC}"
        else:
            tag = f"        "
        # Plan lines go straight to terminal (with color) and to the log
        # file (plain). We bypass log() because we want the index column
        # uncolored even on the log-file side.
        line = f"  {i+1:2d}. {s.name}"
        print(f"{tag}{line}")
        if LOG_FP is not None:
            done = "done" if i < first_incomplete else (
                "resume" if i == first_incomplete else "todo"
            )
            LOG_FP.write(f"  {done:>6}  {line}\n")
    if LOG_FP is not None:
        LOG_FP.flush()

    if first_incomplete < len(plan):
        log(
            f"will resume from stage `{plan[first_incomplete].name}` "
            "(any partial outputs of that stage will be deleted and "
            "re-created)"
        )

    # In fresh mode there's nothing to resume, so skip the prompt — the
    # user already opted in by passing a non-existent --dir.
    if mode != "fresh":
        if not _confirm("proceed with this plan?"):
            log("user declined plan; exiting")
            session_summary(start_time, dir_path, "ABORTED: user declined")
            sys.exit(1)

    # ---- persist state and run remaining stages ------------------------
    started_at = (saved_state or {}).get(
        "started_at", datetime.now().isoformat(timespec="seconds"),
    )

    def save_state() -> None:
        _write_state(state_path, {
            "version": 1,
            "started_at": started_at,
            "server_cmd": server_cmd,
            "bench_cmd": bench_cmd,
            "step_count": step_count,
            "interval": interval_ref["value"],
            "completed_stages": completed_stages,
        })

    save_state()

    try:
        for i in range(first_incomplete, len(plan)):
            stage = plan[i]
            # Wipe any partial output from a previously-aborted attempt
            # at this stage. Without this, a half-written perf .txt or
            # corrupt .jsonl would silently poison the run.
            for out in stage.outputs:
                if out.exists():
                    log(f"removing partial output: {out.name}")
                    out.unlink()
            result = stage.run()
            if stage.name == "count_steps":
                # count_steps returns the step count; convert it to the
                # interval and persist both so resume runs don't redo it.
                assert isinstance(result, int)
                if result < 10:
                    session_summary(
                        start_time, dir_path,
                        f"FAILED: bench produced only {result} EPLB step(s); "
                        "need ≥10. Increase bench workload "
                        "(e.g. --num-prompts) or use a slower model.",
                    )
                    sys.exit(1)
                step_count = result
                interval_ref["value"] = compute_interval(result)
                ok(
                    f"steps={step_count} → "
                    f"expert_load_stats_interval={interval_ref['value']} "
                    f"(targeting ≤{TARGET_STATS_RECORDS} stats records)"
                )
            completed_stages.append(stage.name)
            save_state()

        session_summary(start_time, dir_path, "OK")

    except Exception as exc:
        err(f"{type(exc).__name__}: {exc}")
        try:
            stop_server()
        except Exception as cleanup_exc:
            warn(f"cleanup stop_server failed: {cleanup_exc}")
        save_state()
        session_summary(start_time, dir_path, f"FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
