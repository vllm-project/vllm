#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AMD GPU CI test runner for vLLM -- K8s-hardened.

Runs pytest inside a Docker container on AMD GPU hardware, with proper
handling of GPU lifecycle, exit code propagation, and cleanup.

Designed to run inside Kubernetes pods with Docker socket access, handling
the unique challenges of containerized GPU CI:

  - Cross-PID-namespace GPU zombie detection (via ``rocm-smi --showpids``)
  - File-locked GPU state coordination (safe with parallel agents on same node)
  - Driver-level GPU health validation (VRAM checks, hardware reset fallback)
  - JUnit XML as authoritative exit code source (immune to ``os._exit(0)``
    from vLLM EngineCore / PyTorch NCCL cleanup)
  - Pod-aware multi-node NCCL configuration


Architecture
------------

The script is organized into these layers, each with a single responsibility:

    Output helpers      - Buildkite-friendly logging (section, info, warn, error)
    Shell helper        - Thin wrapper around subprocess.run
    Buildkite layer     - Artifact upload, build annotations, metadata
    Container diagnosis - OOM detection, signal analysis via ``docker inspect``
    GPU VRAM monitoring - Pre/post-test VRAM snapshots via ``rocm-smi``
    JUnit XML parsing   - Failure annotation builder and exit-code validator
    K8s detection       - Pod/node/namespace context for debug logs
    GPU state locking   - flock-based coordination for parallel agents
    GPU management      - Zombie cleanup, health checks, VRAM validation, reset
    Docker management   - Daemon health, disk cleanup, container lifecycle
    Command processing  - VLLM_TEST_COMMANDS source, marker re-quoting, ROCm overrides
    Result validation   - JUnit XML failure count, pytest exit normalization
    Multi-node support  - Bracket-syntax parsing, pod-IP discovery
    Cleanup             - Idempotent teardown (atexit + signal handlers)
    Main                - 10-phase orchestration


The exit-code problem
---------------------

Python libraries and C/C++ extensions can register ``atexit`` hooks or
C-level ``atexit()`` handlers that call ``os._exit(0)`` or ``std::exit(0)``
during interpreter shutdown. When this happens, the exit code that pytest
set (e.g., 1 for test failures) is silently overwritten to 0 before the
process actually terminates.

This affects any test suite whose dependencies register such hooks --
including but not limited to GPU runtimes, distributed communication
libraries, and engine subprocess managers.

Because ``docker wait`` returns the process exit code *after* these hooks
have run, neither ``docker run --rm`` + ``$?`` nor ``docker wait`` alone
will see the real pytest exit code.

This script uses **JUnit XML** as the authoritative source of truth. Pytest
writes the XML file during ``pytest_sessionfinish`` -- before Python's atexit
handlers execute. The XML is written to a bind-mounted volume, so it
survives on the host after the container exits. After ``docker wait``
returns, we parse the XML: if it reports failures but the exit code is 0,
we override the exit code to 1.


Usage
-----
Preferred (quoting preserved)::

    export VLLM_TEST_COMMANDS='pytest -v -s tests/ -m "not slow"'
    python3 run-amd-test.py

Legacy (backward-compatible, inner double-quotes may be stripped)::

    python3 run-amd-test.py "pytest -v -s tests/"

The bash shim ``run-amd-test.sh`` does ``exec python3 run-amd-test.py "$@"``
so existing Buildkite pipeline YAML needs no changes.
"""

from __future__ import annotations

import atexit
import fcntl
import grp
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from pathlib import Path

import regex as re

# ==========================================================================
# Constants
#
# All tunables are grouped here so operators can adjust them without reading
# the rest of the file. Environment variable overrides are noted inline.
# ==========================================================================

# GPU state file written by the AMD GPU driver's userspace tooling.
# The driver writes "clean" after a successful reset. We write "reset"
# to request one. Multiple agents may share this file (hence the lock).
GPU_STATE_FILE = Path("/opt/amdgpu/etc/gpu_state")

# Advisory lock sidecar -- we never write to gpu_state without holding this.
GPU_STATE_LOCK = Path("/opt/amdgpu/etc/gpu_state.lock")

# How long to poll gpu_state for "clean" before giving up.
GPU_CLEAN_TIMEOUT_S = 300

# Poll interval for gpu_state.
GPU_POLL_INTERVAL_S = 3

# Docker cache eviction thresholds. Two limit types: percentage-based and
# absolute GB-based. The STRICTER of the two wins at runtime.
#
# Example: on a 500GB disk, 70% = 350GB used. If DISK_USAGE_THRESHOLD_GB
# is set to 200, eviction starts at 200GB (stricter). On a 100GB disk,
# 70% = 70GB, which is stricter than 200GB, so percentage wins.
#
# Set any value to 0 to disable that specific limit. Setting both PCT and
# GB to 0 disables eviction entirely (a warning is logged at startup).
#
# Override via env vars: VLLM_DISK_THRESHOLD_PCT, VLLM_DISK_THRESHOLD_GB,
# VLLM_DISK_TARGET_PCT, VLLM_DISK_TARGET_GB.

# Percentage thresholds. 0 = disabled (GB-only or no eviction).
DISK_USAGE_THRESHOLD_PCT = int(os.environ.get("VLLM_DISK_THRESHOLD_PCT", "70"))
DISK_USAGE_TARGET_PCT = int(os.environ.get("VLLM_DISK_TARGET_PCT", "50"))

# Absolute thresholds in GB. 0 = disabled (percentage-only or no eviction).
DISK_USAGE_THRESHOLD_GB = int(os.environ.get("VLLM_DISK_THRESHOLD_GB", "0"))
DISK_USAGE_TARGET_GB = int(os.environ.get("VLLM_DISK_TARGET_GB", "0"))

# Eviction policy. Controls the order in which images are removed when
# disk usage exceeds the threshold.
#
#   "lfu" (default) - Least Frequently Used. Images that appear in the
#       fewest running/stopped containers are evicted first. This keeps
#       images that are actively used by multiple jobs (e.g., the ci_base
#       image shared across all PRs) and evicts one-off images first.
#       Better for shared CI nodes with diverse workloads.
#
#   "lru" - Least Recently Used. Images sorted by creation/pull time,
#       oldest evicted first. Simple and predictable. Better for
#       single-tenant nodes where each job uses a different image.
#
# Override via: VLLM_DISK_EVICTION_POLICY=lru or VLLM_DISK_EVICTION_POLICY=lfu
_VALID_EVICTION_POLICIES = ("lfu", "lru")
DISK_EVICTION_POLICY = os.environ.get("VLLM_DISK_EVICTION_POLICY", "lfu").lower()

# Mount point for test results (JUnit XML, container logs).
RESULTS_MOUNT = "/tmp/vllm-ci-results"

# ---------------------------------------------------------------------------
# Persistent cache configuration
#
# Each cache type maps a host-side directory to a container-side mount point
# and an environment variable that tells the code inside the container where
# to find it. Caches persist across CI jobs on the same node, so subsequent
# runs hit warm cache instead of re-downloading.
#
# Adding a new cache:
#   1. Add an entry to CACHES below.
#   2. The host directory is created automatically under CACHE_ROOT.
#   3. The volume mount and env var are injected into docker run.
#
# Override VLLM_CI_CACHE_ROOT to change the host-side base directory
# (default: ~/vllm-ci-cache). Useful when the node has a fast ephemeral
# disk (NVMe) separate from the root partition.
# ---------------------------------------------------------------------------

# Base directory for all caches on the host.
CACHE_ROOT = Path(
    os.environ.get("VLLM_CI_CACHE_ROOT", str(Path.home() / "vllm-ci-cache"))
)

# Cache registry: each entry defines one persistent cache.
#   host_subdir:    directory name under CACHE_ROOT on the host
#   container_path: absolute path inside the container
#   env_var:        environment variable set inside the container
#   description:    human-readable description for logging
# Legacy HF cache path: the original bash script used ~/huggingface (not
# under CACHE_ROOT). We preserve this path so existing warm caches on CI
# nodes are not invalidated. New caches go under CACHE_ROOT.
_HF_CACHE_HOST = Path(os.environ.get("VLLM_HF_CACHE", str(Path.home() / "huggingface")))

CACHES = [
    {
        # host_dir is overridden to _HF_CACHE_HOST below (legacy path).
        "host_subdir": "__hf_legacy__",
        "host_dir_override": str(_HF_CACHE_HOST),
        "container_path": "/root/.cache/huggingface",
        "env_var": "HF_HOME",
        "description": "HuggingFace models and datasets",
    },
    {
        "host_subdir": "modelscope",
        "container_path": "/root/.cache/modelscope",
        "env_var": "MODELSCOPE_CACHE",
        "description": "ModelScope models",
    },
    {
        "host_subdir": "vllm-test-cache",
        "container_path": "/root/.cache/vllm-test-cache",
        "env_var": "VLLM_TEST_CACHE",
        "description": "Test data (dummy models, datasets, tiktoken, GeoTIFFs)",
    },
    {
        "host_subdir": "vllm",
        "container_path": "/root/.cache/vllm",
        "env_var": "VLLM_CACHE_ROOT",
        "description": "vLLM runtime cache (compiled kernels, etc.)",
    },
    {
        "host_subdir": "vllm/media_cache",
        "container_path": "/root/.cache/vllm/media_cache",
        "env_var": "VLLM_MEDIA_CACHE",
        "description": "Media URL cache (images, audio, video from URLs)",
    },
    {
        "host_subdir": "pip",
        "container_path": "/root/.cache/pip",
        "env_var": "PIP_CACHE_DIR",
        "description": "pip download cache (avoids re-downloading wheels)",
    },
    {
        "host_subdir": "ccache",
        "container_path": "/root/.cache/ccache",
        "env_var": "CCACHE_DIR",
        "description": "ccache compiler cache (for per-arch kernel builds)",
    },
]  # type: list[dict[str, str]]

# Pytest exit code 5 = "no tests were collected". We treat this as success
# because shard-based parallelism can legitimately produce empty shards.
PYTEST_NO_TESTS_COLLECTED = 5

# Safety threshold: never SIGKILL PIDs below this (avoids killing init, etc.).
MIN_SYSTEM_PID = 1000

# Remove stopped rocm_* containers older than this (hours).
STALE_CONTAINER_AGE_H = 4

# If ``docker info`` does not respond within this many seconds, abort.
# Increased from 10s to 30s to tolerate slow Docker daemons under load.
DOCKER_HEALTH_TIMEOUT_S = 30

# Number of times to retry ``docker pull`` on failure (network flakes).
DOCKER_PULL_RETRIES = 3
DOCKER_PULL_RETRY_DELAY_S = 10

# Maximum number of PIDs allowed inside the test container.
# Prevents fork-bomb scenarios from killing the K8s node.
CONTAINER_PIDS_LIMIT = 4096

# Shared-memory size passed to ``docker run --shm-size``.
# PyTorch DataLoader workers communicate via /dev/shm.
CONTAINER_SHM_SIZE = "16gb"

# Maximum wall-clock time (seconds) the test container may run.
#
# This must be SHORTER than the Buildkite step's timeout_in_minutes
# (defined in test-amd.yaml) to leave time for post-test work: JUnit XML
# validation, OOM detection, artifact upload, and cleanup. If our timeout
# fires first, we get clean teardown. If Buildkite's fires first, we get
# SIGTERM and can only do emergency cleanup (no JUnit check, no artifacts).
#
# test-amd.yaml values: most steps use 180 min, some use 60, 40, or less.
# We default to 170 min (10 min buffer). Steps with shorter YAML timeouts
# should set VLLM_TEST_TIMEOUT to (yaml_timeout - 10) * 60.
#
# Override: VLLM_TEST_TIMEOUT env var (seconds).
CONTAINER_TIMEOUT_S = int(os.environ.get("VLLM_TEST_TIMEOUT", "10200"))  # 170 min


# ==========================================================================
# Output helpers
#
# Buildkite renders lines starting with "--- " as collapsible section
# headers in the build log. All CI-visible output goes through these
# four functions so the format is consistent and easy to grep.
# ==========================================================================


def section(title):
    # type: (str) -> None
    """Print a Buildkite collapsible section header.

    In the Buildkite web UI, any line beginning with ``--- `` becomes a
    clickable section header that collapses the output below it until the
    next section. We use this to organize the build log into phases.

    Args:
        title: Section name shown in the Buildkite UI.
    """
    print(f"\n--- {title}", flush=True)


def info(msg):
    # type: (str) -> None
    """Print an informational message to stdout.

    All non-error CI output goes through this function so it can be
    filtered or redirected uniformly.

    Args:
        msg: Message text.
    """
    print(msg, flush=True)


def warn(msg):
    # type: (str) -> None
    """Print a warning to stdout (prefixed with WARNING).

    Warnings indicate degraded conditions that do not block the run
    but should be investigated (e.g., high GPU temperature, stale
    containers found).

    Args:
        msg: Warning text.
    """
    print(f"WARNING: {msg}", flush=True)


def error(msg):
    # type: (str) -> None
    """Print an error to stderr (prefixed with ERROR).

    Errors indicate fatal conditions that will cause the script to exit
    non-zero. Sent to stderr so Buildkite highlights them.

    Args:
        msg: Error text.
    """
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


# ==========================================================================
# Shell helper
# ==========================================================================


def sh(cmd, *, check=False, capture=False, timeout=None):
    # type: (str | list[str], ...) -> subprocess.CompletedProcess
    """Run a shell command via subprocess.run with sensible defaults.

    This is the single point of entry for all external command execution.
    Using one function makes it easy to add logging, dry-run support, or
    retry logic later without touching every call site.

    Args:
        cmd:     Command to run. If a string, executed via ``shell=True``.
                 If a list, executed directly (no shell interpolation).
        check:   If True, raise CalledProcessError on non-zero exit.
        capture: If True, capture stdout and stderr (accessible via
                 ``.stdout`` and ``.stderr`` on the returned object).
        timeout: Maximum seconds to wait. Raises TimeoutExpired if exceeded.

    Returns:
        A ``subprocess.CompletedProcess`` instance.
    """
    kwargs = {"text": True}  # type: dict
    if capture:
        kwargs["capture_output"] = True
    if timeout:
        kwargs["timeout"] = timeout
    if isinstance(cmd, str):
        kwargs["shell"] = True
    return subprocess.run(cmd, check=check, **kwargs)


# ==========================================================================
# Timing helper
# ==========================================================================


@contextmanager
def timed(label):
    # type: (str) -> ...
    """Context manager that logs the wall-clock duration of a block.

    Usage::

        with timed("docker pull"):
            sh(["docker", "pull", image])
        # Prints: "docker pull completed in 34.2s"

    Args:
        label: Human-readable name for the operation being timed.
    """
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    info(f"{label} completed in {elapsed:.1f}s")


# ==========================================================================
# Configuration dump
# ==========================================================================


def log_effective_config():
    # type: () -> None
    """Log all effective configuration values at startup.

    This is the single most useful thing for post-mortem debugging:
    when a build fails, the first question is always "what were the
    settings?" Logging them upfront means the answer is always in
    the build log without needing to SSH into the node.
    """
    section("Effective configuration")
    info(f"  GPU_STATE_FILE:            {GPU_STATE_FILE}")
    info(f"  GPU_CLEAN_TIMEOUT_S:       {GPU_CLEAN_TIMEOUT_S}")
    info(f"  DISK_USAGE_THRESHOLD_PCT:  {DISK_USAGE_THRESHOLD_PCT}%")
    info(f"  DISK_USAGE_TARGET_PCT:     {DISK_USAGE_TARGET_PCT}%")
    info(
        f"  DISK_USAGE_THRESHOLD_GB:   {DISK_USAGE_THRESHOLD_GB}GB"
        f"{' (disabled)' if DISK_USAGE_THRESHOLD_GB == 0 else ''}"
    )
    info(
        f"  DISK_USAGE_TARGET_GB:      {DISK_USAGE_TARGET_GB}GB"
        f"{' (disabled)' if DISK_USAGE_TARGET_GB == 0 else ''}"
    )
    info(
        f"  CONTAINER_TIMEOUT_S:       {CONTAINER_TIMEOUT_S}s "
        f"({CONTAINER_TIMEOUT_S // 60}min)"
    )
    info(f"  CONTAINER_PIDS_LIMIT:      {CONTAINER_PIDS_LIMIT}")
    info(f"  CONTAINER_SHM_SIZE:        {CONTAINER_SHM_SIZE}")
    info(f"  DOCKER_PULL_RETRIES:       {DOCKER_PULL_RETRIES}")
    info(f"  DOCKER_HEALTH_TIMEOUT_S:   {DOCKER_HEALTH_TIMEOUT_S}")
    info(f"  DISK_EVICTION_POLICY:      {DISK_EVICTION_POLICY}")
    info(f"  STALE_CONTAINER_AGE_H:     {STALE_CONTAINER_AGE_H}h")
    info(f"  MIN_SYSTEM_PID:            {MIN_SYSTEM_PID}")
    info(f"  CACHE_ROOT:                {CACHE_ROOT}")
    info(
        f"  Caches registered:         {len(CACHES)} "
        f"({', '.join(c['env_var'] for c in CACHES)})"
    )

    # Log env var overrides so operators can see what was customized.
    overrides = []  # type: list[str]
    for var in [
        "VLLM_TEST_TIMEOUT",
        "VLLM_DISK_THRESHOLD_PCT",
        "VLLM_DISK_THRESHOLD_GB",
        "VLLM_DISK_TARGET_PCT",
        "VLLM_DISK_TARGET_GB",
        "VLLM_DISK_EVICTION_POLICY",
        "VLLM_CI_REGISTRY",
    ]:
        val = os.environ.get(var)
        if val is not None:
            overrides.append(f"{var}={val}")
    if overrides:
        info(f"  Env overrides: {', '.join(overrides)}")
    else:
        info("  Env overrides: none (all defaults)")

    # Validate: warn if thresholds are set to 0 (disables eviction silently).
    if DISK_USAGE_THRESHOLD_PCT <= 0 and DISK_USAGE_THRESHOLD_GB <= 0:
        warn(
            "Both disk thresholds are 0 -- disk cache eviction is DISABLED. "
            "The disk may fill up without intervention."
        )
    if DISK_USAGE_TARGET_PCT <= 0 and DISK_USAGE_TARGET_GB <= 0:
        warn(
            "Both disk targets are 0 -- eviction will never stop once started. "
            "This will wipe the entire Docker cache."
        )
    if DISK_EVICTION_POLICY not in _VALID_EVICTION_POLICIES:
        error(
            f"VLLM_DISK_EVICTION_POLICY='{DISK_EVICTION_POLICY}' is not valid. "
            f"Must be one of: {', '.join(_VALID_EVICTION_POLICIES)}"
        )
        sys.exit(1)


# ==========================================================================
# Buildkite integration (artifacts, annotations, metadata)
#
# These functions are no-ops when ``buildkite-agent`` is not on PATH,
# so the script can run outside Buildkite (e.g., local debugging).
# ==========================================================================


def _has_buildkite_agent():
    # type: () -> bool
    """Return True if the ``buildkite-agent`` binary is on PATH."""
    return shutil.which("buildkite-agent") is not None


def upload_artifacts(*globs):
    # type: (*str) -> None
    """Upload files matching glob patterns to Buildkite artifact storage.

    Uploaded artifacts are downloadable from the Buildkite build page and
    persist after the build completes. We upload JUnit XML (for test-result
    dashboards) and container logs (for post-mortem debugging).

    No-op if ``buildkite-agent`` is not available.

    Args:
        globs: One or more glob patterns (e.g., "/tmp/results/*.xml").
    """
    if not _has_buildkite_agent():
        return
    for pattern in globs:
        r = sh(
            ["buildkite-agent", "artifact", "upload", pattern],
            capture=True,
            timeout=60,
        )
        if r.returncode != 0:
            warn(f"Artifact upload failed for '{pattern}'")


def annotate_build(body, style="error", context="test-result"):
    # type: (str, str, str) -> None
    """Post a markdown annotation to the Buildkite build page.

    Annotations appear at the very top of the build page, above the log.
    This makes test failures, OOM kills, and timeouts immediately visible
    without scrolling through thousands of log lines.

    The ``context`` parameter deduplicates: posting with the same context
    replaces the previous annotation rather than appending.

    No-op if ``buildkite-agent`` is not available.

    Args:
        body:    Markdown-formatted annotation body.
        style:   One of "success", "info", "warning", "error".
        context: Deduplication key.
    """
    if not _has_buildkite_agent():
        return
    # Pipe body via stdin to avoid shell-escaping issues in the markdown.
    proc = subprocess.run(
        ["buildkite-agent", "annotate", "--style", style, "--context", context],
        input=body,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        warn("buildkite-agent annotate failed")


def set_buildkite_meta(key, value):
    # type: (str, str) -> None
    """Set a key-value pair in Buildkite build metadata.

    Metadata is queryable via the Buildkite API and from other pipeline
    steps. We use it to expose structured data (exit code, OOM status,
    timeout flag) for dashboards and alerting.

    No-op if ``buildkite-agent`` is not available.

    Args:
        key:   Metadata key (e.g., "test_exit_code").
        value: Metadata value (always a string).
    """
    if not _has_buildkite_agent():
        return
    # Use list form to avoid shell injection via key/value content.
    sh(["buildkite-agent", "meta-data", "set", key, value], timeout=10)


# ==========================================================================
# Container OOM / exit diagnosis
#
# After the container stops but BEFORE we ``docker rm`` it, we inspect its
# state to determine *why* it exited. This catches OOM kills, signal
# deaths, and K8s evictions that would otherwise be invisible.
# ==========================================================================


# Prefix for all diagnostic messages so they are easy to grep in CI logs.
_DIAG_PREFIX = "[run-amd-test.py diagnostics]"


def diagnose_container_exit(container_name, log_file=None):
    # type: (str, Path | None) -> dict[str, object]
    """Inspect a stopped container and produce clear, actionable error messages.

    Must be called BEFORE ``docker rm`` -- once the container is removed,
    its state metadata is gone.

    Checks for:
      - OOM kill (kernel killed the container for exceeding memory).
      - Signal kills (SIGKILL, SIGTERM, SIGSEGV, SIGABRT) with cause analysis.
      - PID limit exhaustion (fork failures from --pids-limit).
      - K8s eviction (SIGKILL without OOM -- kubelet killed the pod).

    All diagnostic messages are prefixed with ``[run-amd-test.py diagnostics]``
    so they can be grepped in Buildkite logs.

    Args:
        container_name: Docker container name or ID.
        log_file:       Path to the container log file on the host. If provided,
                        the log is scanned for resource-limit error patterns
                        (e.g., "fork: Resource temporarily unavailable").

    Returns:
        Dict with keys:
          "oom_killed" (bool), "exit_code" (int), "error" (str),
          "pids_exhausted" (bool), "timed_out" (bool).
    """
    diag = {
        "oom_killed": False,
        "exit_code": -1,
        "error": "",
        "pids_exhausted": False,
        "timed_out": False,
    }  # type: dict[str, object]

    # -- Docker inspect --
    r = sh(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.OOMKilled}} {{.State.ExitCode}} {{.State.Error}}",
            container_name,
        ],
        capture=True,
    )
    if r.returncode != 0:
        warn(
            f"docker inspect {container_name} failed "
            f"(rc={r.returncode}) -- cannot diagnose exit"
        )
        return diag

    info(f"docker inspect state: {r.stdout.strip()}")
    parts = r.stdout.strip().split(None, 2)
    if len(parts) >= 1:
        diag["oom_killed"] = parts[0].lower() == "true"
    if len(parts) >= 2:
        try:
            diag["exit_code"] = int(parts[1])
        except ValueError:
            warn(f"Could not parse exit code from docker inspect: '{parts[1]}'")
    if len(parts) >= 3:
        diag["error"] = parts[2]

    # -- Scan container log for resource-limit patterns --
    if log_file and log_file.is_file():
        try:
            # Read only the last 100KB to avoid loading huge logs into memory.
            size = log_file.stat().st_size
            with open(log_file, errors="replace") as f:
                if size > 100_000:
                    f.seek(size - 100_000)
                tail = f.read()
        except OSError as exc:
            warn(f"Could not read container log {log_file} for diagnosis: {exc}")
            tail = ""

        # PID limit: when --pids-limit is hit, fork/clone syscalls fail.
        # The error appears in the test output as one of these patterns.
        pid_patterns = [
            "fork: Resource temporarily unavailable",
            "Cannot allocate memory",
            "OSError: [Errno 11]",  # EAGAIN from Python os.fork()
            "BlockingIOError: [Errno 11]",  # EAGAIN from subprocess
            "RuntimeError: can't start new thread",
        ]
        for pattern in pid_patterns:
            if pattern in tail:
                diag["pids_exhausted"] = True
                break

    # -- Emit clear, actionable diagnostics --
    code = diag["exit_code"]

    if diag["oom_killed"]:
        error(
            f"{_DIAG_PREFIX} CONTAINER OOM KILLED\n"
            f"  What happened: The kernel OOM killer terminated the container\n"
            f"                 because it exceeded its memory limit.\n"
            f"  Container:     {container_name}\n"
            f"  Exit code:     {code}\n"
            f"  How to fix:\n"
            f"    - Increase --shm-size (currently {CONTAINER_SHM_SIZE})\n"
            f"    - Increase the K8s pod memory limit\n"
            f"    - Reduce batch size or model size in the test\n"
            f"    - Check for memory leaks (compare pre/post VRAM snapshots above)"
        )

    if diag["pids_exhausted"]:
        error(
            f"{_DIAG_PREFIX} PID LIMIT EXHAUSTED\n"
            f"  What happened: The container hit the\n"
            f"    --pids-limit={CONTAINER_PIDS_LIMIT}.\n"
            f"    New process/thread creation failed with\n"
            f"                 'fork: Resource temporarily unavailable' or similar.\n"
            f"  Container:     {container_name}\n"
            f"  Exit code:     {code}\n"
            f"  How to fix:\n"
            f"    - Increase CONTAINER_PIDS_LIMIT in run-amd-test.py\n"
            f"      (currently {CONTAINER_PIDS_LIMIT})\n"
            f"    - Check if test spawns too many workers\n"
            f"    - Check for process leaks (zombie processes accumulating)"
        )

    if isinstance(code, int) and code > 128:
        sig = code - 128
        sig_names = {9: "SIGKILL", 15: "SIGTERM", 6: "SIGABRT", 11: "SIGSEGV"}
        sig_name = sig_names.get(sig, f"signal {sig}")

        if sig == 9 and not diag["oom_killed"]:
            error(
                f"{_DIAG_PREFIX} CONTAINER KILLED BY SIGKILL (no OOM)\n"
                f"  What happened: The container received SIGKILL but was NOT\n"
                f"    flagged as OOM-killed by Docker.\n"
                f"    This usually means:\n"
                f"    - K8s kubelet killed the pod\n"
                f"      (memory limit or eviction)\n"
                f"                 - An external process sent SIGKILL\n"
                f"  Container:     {container_name}\n"
                f"  Exit code:     {code}\n"
                f"  How to fix:\n"
                f"    - Check K8s events: kubectl describe pod <pod-name>\n"
                f"    - Check node memory pressure: kubectl top node\n"
                f"    - Increase the pod's memory request/limit in the K8s manifest"
            )
        elif sig == 11:
            error(
                f"{_DIAG_PREFIX} CONTAINER SEGFAULT (SIGSEGV)\n"
                f"  What happened: The test process crashed\n"
                f"    with a segmentation fault.\n"
                f"  Container:     {container_name}\n"
                f"  Exit code:     {code}\n"
                f"  How to fix:\n"
                f"    - Check C/C++ extension bugs\n"
                f"      (ROCm, PyTorch, custom kernels)\n"
                f"    - Check for stack overflow in recursive code\n"
                f"    - Run with MALLOC_CHECK_=3 or AddressSanitizer for more detail"
            )
        elif sig == 6:
            error(
                f"{_DIAG_PREFIX} CONTAINER ABORTED (SIGABRT)\n"
                f"  What happened: The process called\n"
                f"    abort() or a C++ assertion failed.\n"
                f"  Container:     {container_name}\n"
                f"  Exit code:     {code}\n"
                f"  How to fix:\n"
                f"    - Check the container log for assertion messages\n"
                f"    - This often comes from NCCL, ROCm runtime, or PyTorch C++ code"
            )
        elif not diag["oom_killed"]:
            warn(f"{_DIAG_PREFIX} Container killed by {sig_name} (exit code {code})")

    return diag


# ==========================================================================
# GPU VRAM monitoring
#
# We snapshot VRAM usage before and after the test run. Comparing the two
# reveals GPU memory leaks: if post-test VRAM is significantly higher than
# pre-test, something in the test (or vLLM) is not releasing GPU memory.
# ==========================================================================


def snapshot_gpu_vram(label=""):
    # type: (str) -> dict[int, dict[str, int]]
    """Capture per-GPU VRAM usage via rocm-smi and log it.

    Attempts JSON output first (``--showmemuse --json``); falls back to
    plain-text ``--showmeminfo vram`` if JSON is not supported by the
    installed rocm-smi version.

    Args:
        label: Optional prefix for log lines (e.g., "pre-test", "post-test").

    Returns:
        Dict mapping GPU index to {"used": bytes, "total": bytes}.
        Empty dict if rocm-smi is unavailable.
    """
    prefix = f"[{label}] " if label else ""
    r = sh("rocm-smi --showmemuse --json 2>/dev/null", capture=True)
    if r.returncode != 0:
        # Fallback: non-JSON output for older rocm-smi versions.
        r = sh("rocm-smi --showmeminfo vram 2>/dev/null", capture=True)
        if r.returncode == 0:
            info(f"{prefix}VRAM snapshot:\n{r.stdout.strip()}")
        return {}

    try:
        data = json.loads(r.stdout)
    except (json.JSONDecodeError, ValueError):
        return {}

    result = {}  # type: dict[int, dict[str, int]]
    for key, val in data.items():
        # rocm-smi JSON keys are "card0", "card1", etc.
        if not key.startswith("card"):
            continue
        try:
            idx = int(key.replace("card", ""))
        except ValueError:
            continue
        # Prefer byte-level keys; fall back to 0 if missing.
        used_raw = val.get("VRAM Total Used Memory (B)")
        total_raw = val.get("VRAM Total Memory (B)")
        if used_raw is None or total_raw is None:
            warn(
                f"GPU {idx}: VRAM keys missing from rocm-smi JSON -- "
                f"available keys: {list(val.keys())}"
            )
            continue
        used = int(used_raw)
        total = int(total_raw)
        # Sanity check: used should not exceed total.
        if total > 0 and used > total:
            warn(
                f"GPU {idx}: VRAM used ({used}) > total ({total}) -- "
                f"data may be corrupted"
            )
        result[idx] = {"used": used, "total": total}

    if result:
        for idx, mem in sorted(result.items()):
            used_mb = mem["used"] / (1024 * 1024) if mem["used"] > 1024 else mem["used"]
            total_mb = (
                mem["total"] / (1024 * 1024) if mem["total"] > 1024 else mem["total"]
            )
            info(f"{prefix}GPU {idx}: VRAM {used_mb:.0f}MB / {total_mb:.0f}MB")

    return result


# ==========================================================================
# JUnit XML -> Buildkite failure annotation
#
# When tests fail, we parse the JUnit XML to build a concise markdown
# summary that gets posted as a Buildkite annotation. This puts the
# failure details front-and-center on the build page.
# ==========================================================================


def build_failure_annotation(xml_path):
    # type: (Path) -> str | None
    """Parse JUnit XML and build a Buildkite annotation with failure details.

    Extracts the test name, class, duration, and error message from each
    failed/errored test case. Returns a markdown string suitable for
    ``annotate_build()``, or None if no failures are found.

    Caps output at 20 failures to avoid enormous annotations.

    Args:
        xml_path: Path to the JUnit XML file.

    Returns:
        Markdown string, or None.
    """
    if not xml_path.is_file():
        return None

    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError) as exc:
        warn(f"Could not parse JUnit XML {xml_path}: {exc}")
        return None

    # Walk all <testcase> elements. Works with both <testsuite> (single
    # suite) and <testsuites> (multi-suite) root elements.
    cases = root.iter("testcase")
    failures = []  # type: list[dict[str, str]]

    for tc in cases:
        fail = tc.find("failure")
        err = tc.find("error")
        element = fail if fail is not None else err
        if element is None:
            continue

        name = tc.get("name", "unknown")
        classname = tc.get("classname", "")
        duration = tc.get("time", "?")
        message = (element.get("message") or element.text or "")[:300]

        failures.append(
            {
                "name": f"{classname}::{name}" if classname else name,
                "duration": duration,
                "message": message.strip(),
            }
        )

    if not failures:
        return None

    # Build markdown body.
    lines = [f"### :x: {len(failures)} test failure(s)\n"]
    for f in failures[:20]:
        lines.append(f"**`{f['name']}`** ({f['duration']}s)")
        if f["message"]:
            short_msg = f["message"][:200].replace("\n", "\n> ")
            lines.append(f"> {short_msg}")
        lines.append("")

    if len(failures) > 20:
        lines.append(f"*... and {len(failures) - 20} more. See JUnit XML artifact.*")

    return "\n".join(lines)


# ==========================================================================
# K8s environment detection
#
# When running inside a Kubernetes pod, we log the pod name, node name,
# namespace, and allocated GPU devices. This context is invaluable when
# debugging flaky failures that only happen on specific nodes.
# ==========================================================================


def is_k8s():
    # type: () -> bool
    """Return True if the current process is running inside a K8s pod.

    Detection methods (either is sufficient):
      1. KUBERNETES_SERVICE_HOST env var is set (injected by kubelet).
      2. The ServiceAccount token file exists on disk.
    """
    return (
        os.environ.get("KUBERNETES_SERVICE_HOST") is not None
        or Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()
    )


def log_k8s_context():
    # type: () -> None
    """Log Kubernetes pod context for CI debugging.

    Prints pod name, node name, namespace, and GPU device allocation.
    On bare-metal / VM environments, prints a single line and returns.
    """
    if not is_k8s():
        info("Environment: bare-metal / VM (not K8s)")
        return

    pod = os.environ.get("HOSTNAME", "unknown")
    node = os.environ.get("NODE_NAME", os.environ.get("KUBE_NODE_NAME", "unknown"))
    ns = "unknown"
    with suppress(OSError):
        ns = (
            Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
            .read_text()
            .strip()
        )

    info(f"Environment: Kubernetes pod={pod} node={node} namespace={ns}")

    # The AMD K8s device plugin sets AMD_VISIBLE_DEVICES or GPU_DEVICE_ORDINAL
    # to indicate which GPUs were allocated to this pod.
    allocated = os.environ.get(
        "AMD_VISIBLE_DEVICES", os.environ.get("GPU_DEVICE_ORDINAL", "")
    )
    if allocated:
        info(f"K8s device-plugin allocated GPUs: {allocated}")


# ==========================================================================
# GPU state file locking
#
# On a shared K8s node, multiple Buildkite agents (in separate pods) may
# read/write /opt/amdgpu/etc/gpu_state concurrently. Without coordination,
# agent A could read "clean", start a test, and agent B could simultaneously
# write "reset" -- corrupting A's assumption about GPU state.
#
# We use POSIX advisory file locking (fcntl.flock) on a sidecar .lock file.
# Advisory locks are respected by all processes that use them; processes
# that don't (e.g., the GPU driver itself) are unaffected.
# ==========================================================================


@contextmanager
def _gpu_state_lock(blocking=True, timeout=30.0):
    # type: (bool, float) -> ...
    """Context manager: acquire an advisory flock on the GPU state file.

    Uses a sidecar ``.lock`` file so the state file itself stays clean
    (no lock metadata mixed with "clean"/"reset" content).

    Args:
        blocking: If True, wait up to ``timeout`` seconds for the lock.
                  If False, try once and warn if it fails.
        timeout:  Maximum seconds to wait (only used when blocking=True).

    Yields:
        Nothing. The lock is held for the duration of the ``with`` block.

    Notes:
        - If the lock cannot be acquired, we warn and proceed (fail-open).
          This is intentional: a stuck lock file should not block all CI.
        - The lock file is created if it does not exist.
    """
    lock_path = GPU_STATE_LOCK
    fd = None
    try:
        # Only attempt lock if the parent directory already exists.
        # In K8s, /opt/amdgpu/etc/ does not exist and we should not
        # create it (the filesystem may be read-only).
        if not lock_path.parent.is_dir():
            warn(f"Lock directory {lock_path.parent} does not exist -- skipping lock")
            yield
            return
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        acquired = False
        if blocking:
            # Try to acquire the lock, polling every 0.5s until the deadline.
            # Check the deadline BEFORE each attempt, not after.
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.5)
            if not acquired:
                warn(
                    f"Timed out waiting for GPU state lock "
                    f"after {timeout}s -- proceeding without it"
                )
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                warn("GPU state lock held by another process -- proceeding without it")
        yield
    except OSError as exc:
        warn(f"Cannot acquire GPU state lock ({lock_path}): {exc}")
        yield
    finally:
        if fd is not None:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ==========================================================================
# GPU Management (driver-level, cross-PID-namespace aware)
#
# GPU zombie detection is the single most important pre-flight check.
# A zombie is a process from a previous CI job that crashed without
# releasing its GPU resources. If we start a new test on a GPU that
# still has allocated VRAM, the test will OOM or produce wrong results.
#
# In bare-metal setups, ``fuser /dev/kfd`` finds these zombies. But in
# K8s, each pod has its own PID namespace, so fuser only sees processes
# in the *current* pod. A zombie from a *previous* pod is invisible.
#
# ``rocm-smi --showpids`` queries the AMD kernel driver directly and
# returns PIDs in the *host* PID namespace, regardless of which container
# they came from. This is the only reliable detection method in K8s.
# ==========================================================================


def _device_pids(device):
    # type: (str) -> list[int]
    """Return PIDs holding a device file open, via ``fuser``.

    Limitation: in K8s, fuser only sees PIDs in the current PID namespace.
    Use ``rocm_smi_gpu_pids()`` for cross-namespace visibility.

    Args:
        device: Device path (e.g., "/dev/kfd", "/dev/dri/renderD128").

    Returns:
        List of integer PIDs. Empty list if fuser fails or finds nothing.
    """
    r = sh(f"fuser {device} 2>/dev/null", capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    return [int(p) for p in r.stdout.split() if p.strip().isdigit()]


def _kill_pid(pid):
    # type: (int) -> None
    """Send SIGKILL to a process after safety checks.

    Safety:
      - Refuses to kill PIDs <= MIN_SYSTEM_PID (avoids init, kernel threads).
      - Refuses to kill our own PID (avoids suicide).
      - Logs the process name (from /proc/PID/comm) before killing.
      - Silently ignores ProcessLookupError (already dead) and
        PermissionError (different user -- e.g., root vs agent).

    Args:
        pid: Process ID to kill.
    """
    if pid <= MIN_SYSTEM_PID or pid == os.getpid():
        return
    try:
        name = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        name = "unknown"
    info(f"  Killing orphan GPU process {pid} ({name})")
    with suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


def rocm_smi_gpu_pids():
    # type: () -> list[int]
    """Get PIDs using GPUs via ``rocm-smi --showpids``.

    Unlike ``fuser /dev/kfd``, this queries the AMD kernel driver directly
    and returns PIDs in the host PID namespace. In K8s, this sees processes
    from ALL containers on the node, not just the current pod.

    This is the primary zombie-detection method for K8s environments.

    Returns:
        List of integer PIDs with PID > MIN_SYSTEM_PID.
        Empty list if rocm-smi is unavailable or reports no processes.
    """
    r = sh("rocm-smi --showpids 2>/dev/null", capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []

    pids = []  # type: list[int]
    for line in r.stdout.splitlines():
        # Output format varies by rocm-smi version. Common formats:
        #   "  PID  12345  ..."     (tabular)
        #   "12345"                 (one PID per line)
        # We take the first numeric token on each line as the PID.
        for token in line.split():
            if token.isdigit():
                pid = int(token)
                if pid > MIN_SYSTEM_PID:
                    pids.append(pid)
                break  # only first numeric token per line
    return pids


def rocm_smi_check_vram():
    # type: () -> bool
    """Verify that no GPUs have allocated VRAM.

    Parses ``rocm-smi --showmeminfo vram`` output, looking for "Used"
    lines with non-zero values.

    Returns:
        True if all GPUs report zero VRAM usage (or if rocm-smi fails,
        in which case we return True optimistically to avoid blocking CI).
    """
    r = sh("rocm-smi --showmeminfo vram 2>/dev/null", capture=True)
    if r.returncode != 0:
        warn("rocm-smi --showmeminfo failed -- cannot verify VRAM")
        return True  # optimistic: don't block CI on tooling failure

    for line in r.stdout.splitlines():
        lower = line.lower()
        if "used" in lower:
            for token in line.split():
                cleaned = token.rstrip("BMKGbmkg")
                if cleaned.isdigit() and int(cleaned) > 0:
                    warn(f"GPU VRAM still in use: {line.strip()}")
                    return False
    return True


def rocm_smi_hard_reset():
    # type: () -> bool
    """Attempt a hardware-level GPU reset via ``rocm-smi --gpureset``.

    This is the last resort when soft reset (writing "reset" to gpu_state)
    and zombie cleanup have both failed to free GPU memory. It resets the
    GPU at the PCIe level, which clears all VRAM and kills any remaining
    GPU contexts.

    Returns:
        True if the reset succeeded, False otherwise.
    """
    info("Attempting hardware GPU reset via rocm-smi...")
    r = sh("rocm-smi --gpureset 2>&1", capture=True)
    if r.returncode == 0:
        info("rocm-smi --gpureset succeeded")
        return True
    warn(f"rocm-smi --gpureset failed (rc={r.returncode}): {r.stdout.strip()}")
    return False


def rocm_smi_validate_health():
    # type: () -> None
    """Pre-flight GPU health validation.

    Checks three things before allowing tests to run:

    1. GPU enumeration (``--showid``): Confirms GPUs are visible. Fails
       hard if not -- likely means the K8s device plugin did not allocate
       any GPUs to this pod.

    2. Temperature (``--showtemp``): Warns if any GPU exceeds 90C. Does
       not fail -- thermal throttling degrades performance but tests may
       still pass.

    3. ECC errors (``--showrasinfo``): Warns on uncorrectable errors.
       Does not fail -- hardware errors are the infra team's problem, not
       the test's.
    """
    section("Validating GPU health")

    # 1. Enumeration
    r = sh("rocm-smi --showid 2>/dev/null", capture=True)
    if r.returncode != 0:
        error("rocm-smi --showid failed -- GPUs may not be accessible")
        error("Check that the K8s device plugin allocated GPUs to this pod")
        sys.exit(1)
    info(r.stdout.strip())

    # 2. Temperature
    r = sh("rocm-smi --showtemp 2>/dev/null", capture=True)
    if r.returncode == 0:
        info(r.stdout.strip())
        for line in r.stdout.splitlines():
            for token in line.split():
                try:
                    temp = float(token)
                    if temp > 90.0:
                        warn(f"GPU temperature {temp}C > 90C -- throttling risk")
                except ValueError:
                    continue

    # 3. ECC errors
    r = sh("rocm-smi --showrasinfo 2>/dev/null", capture=True)
    if r.returncode == 0 and r.stdout.strip():
        for line in r.stdout.splitlines():
            lower = line.lower()
            if "uncorrectable" in lower:
                for token in line.split():
                    if token.isdigit() and int(token) > 0:
                        warn(f"GPU has uncorrectable ECC errors: {line.strip()}")
                        break


def kill_gpu_zombies():
    # type: () -> None
    """Find and kill leftover GPU processes from previous CI runs.

    Uses four detection methods, in order of increasing scope:

    1. ``fuser /dev/kfd`` and ``fuser /dev/dri/renderD*``:
       Finds processes holding GPU device files open. Only sees the
       current PID namespace (useless for cross-container zombies in K8s).

    2. ``rocm-smi --showpids``:
       Queries the AMD kernel driver directly. Sees ALL processes using
       GPUs on the node, regardless of PID namespace. This is the primary
       detection method for K8s.

    3. ``docker ps --filter name=rocm_``:
       Finds running containers with the ``rocm_`` prefix (our naming
       convention). These are containers from a previous CI job that
       were not cleaned up (e.g., the agent crashed).

    4. ``docker ps -a --filter status=exited/dead/created``:
       Removes stopped containers that were never cleaned up. Prevents
       the container list from growing indefinitely.

    After cleanup, verifies VRAM was actually released. If not, attempts
    a hardware GPU reset as a last resort.
    """
    section("Pre-flight: checking for leftover GPU processes")
    found = False

    # Method 1: fuser (same PID namespace only)
    for pid in _device_pids("/dev/kfd"):
        found = True
        _kill_pid(pid)

    dri = Path("/dev/dri")
    if dri.is_dir():
        for dev in sorted(dri.glob("renderD*")):
            for pid in _device_pids(str(dev)):
                found = True
                _kill_pid(pid)

    # Method 2: rocm-smi (cross PID namespace -- the one that matters in K8s)
    driver_pids = rocm_smi_gpu_pids()
    if driver_pids:
        found = True
        warn(f"rocm-smi reports GPU processes (driver-level): {driver_pids}")
        for pid in driver_pids:
            _kill_pid(pid)

    # Method 3: running stale containers
    r = sh("docker ps -q --filter name=rocm_", capture=True)
    if r.returncode == 0 and r.stdout.strip():
        found = True
        warn("Found stale rocm_ containers, removing...")
        for cid in r.stdout.strip().splitlines():
            sh(f"docker rm -f {cid} 2>/dev/null || true")

    # Method 4: stopped/dead/created containers
    r = sh(
        "docker ps -a -q --filter name=rocm_ "
        "--filter 'status=exited' "
        "--filter 'status=dead' "
        "--filter 'status=created'",
        capture=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        for cid in r.stdout.strip().splitlines():
            sh(f"docker rm -f {cid} 2>/dev/null || true")

    if found:
        info("Cleaned up leftover processes, waiting for GPU memory release...")
        time.sleep(5)

        # Verify VRAM was actually released.
        if not rocm_smi_check_vram():
            warn("VRAM still allocated after cleanup -- attempting hardware reset")
            rocm_smi_hard_reset()
            time.sleep(3)
            if not rocm_smi_check_vram():
                error("VRAM still allocated after hardware reset -- GPUs may be stuck")
                error(
                    "This node may need manual intervention "
                    "(reboot or amdgpu driver reload)"
                )
    else:
        info("No leftover GPU processes found")


def _gpu_state_file_available():
    # type: () -> bool
    """Return True if the GPU state file exists and is readable.

    In Kubernetes pods, /opt/amdgpu/etc/gpu_state does not exist because
    GPU lifecycle is managed by the AMD device plugin and kubelet, not by
    host-level userspace tooling. On bare-metal nodes with the AMD GPU
    driver's userspace stack installed, the file is present.

    All gpu_state interactions should be gated on this check to avoid
    blocking for 300s polling a nonexistent file.
    """
    return GPU_STATE_FILE.exists() and os.access(str(GPU_STATE_FILE), os.R_OK)


def wait_for_clean_gpus(timeout=GPU_CLEAN_TIMEOUT_S):
    # type: (int) -> None
    """Block until the GPU state file contains 'clean'.

    After writing "reset" to the state file, the AMD GPU driver's userspace
    tooling performs the actual reset and writes "clean" when done. We poll
    the file every GPU_POLL_INTERVAL_S seconds.

    File access is protected by ``_gpu_state_lock`` to prevent races with
    parallel Buildkite agents on the same node.

    **Skipped in K8s** (or any environment where the state file does not
    exist). In that case, GPU readiness is validated via ``rocm-smi``
    instead (see ``rocm_smi_validate_health``).

    Args:
        timeout: Maximum seconds to wait. Exits the script if exceeded.
    """
    if not _gpu_state_file_available():
        info("GPU state file not found (expected in K8s) -- skipping state-file wait")
        return

    section(f"Waiting for clean GPU state (timeout: {timeout}s)")
    deadline = time.monotonic() + timeout
    while True:
        try:
            with _gpu_state_lock(blocking=True, timeout=5.0):
                content = GPU_STATE_FILE.read_text().strip()
                if "clean" in content:
                    info('GPUs state is "clean"')
                    return
        except OSError as exc:
            warn(f"Cannot read {GPU_STATE_FILE}: {exc}")
        if time.monotonic() >= deadline:
            error(f"GPUs did not reach clean state within {timeout}s")
            sys.exit(1)
        time.sleep(GPU_POLL_INTERVAL_S)


def reset_gpus():
    # type: () -> None
    """Request a GPU reset via the state file and wait for completion.

    Writes "reset" to the GPU state file (under lock), then calls
    ``wait_for_clean_gpus`` to block until the driver confirms the
    reset is complete.

    **Skipped in K8s** (or any environment where the state file does not
    exist). In K8s, GPU reset is handled by the device plugin. If a GPU
    is truly stuck, the pod should be evicted and rescheduled.
    """
    if not _gpu_state_file_available():
        info("GPU state file not found (expected in K8s) -- skipping state-file reset")
        return

    section("Resetting GPUs")
    try:
        with _gpu_state_lock():
            GPU_STATE_FILE.write_text("reset\n")
    except OSError as exc:
        warn(f"Cannot write {GPU_STATE_FILE}: {exc}")
    wait_for_clean_gpus()


# ==========================================================================
# Persistent cache management
#
# CI jobs download large files: model weights (HF), test datasets, compiled
# kernels, pip wheels. Without caching, every job re-downloads everything
# over the network. With persistent caches on the host, subsequent jobs on
# the same node hit warm cache.
#
# The cache system is designed to work with:
#   - Multi-tier Docker images (PR 36949): ci_base caches stable deps,
#     per-commit images are thin. Host caches complement this by caching
#     data that changes per-model (HF weights) not per-build.
#   - Media URL caching (PR 37123): VLLM_MEDIA_CACHE caches fetched media
#     files by SHA-256 hash, avoiding repeated downloads in tests.
#   - Ephemeral storage: set VLLM_CI_CACHE_ROOT to a fast local disk
#     (NVMe /scratch) instead of shared NFS for better I/O.
#   - Per-arch builds: the ccache entry caches compiled kernels so that
#     ROCm arch-specific builds (gfx90a, gfx942, gfx950) can share
#     compilation results across jobs.
# ==========================================================================


def setup_caches():
    # type: () -> list[dict[str, str]]
    """Create host-side cache directories and return the active cache list.

    For each entry in CACHES:
      1. Creates the host directory (CACHE_ROOT / host_subdir) if it does
         not exist.
      2. Logs the directory path, size, and file count for visibility.

    Returns:
        The CACHES list (unchanged), for use by ``build_cache_docker_args``.
    """
    section("Persistent cache setup")
    info(f"Cache root: {CACHE_ROOT}")
    if not CACHE_ROOT.exists():
        info(f"  Creating cache root: {CACHE_ROOT}")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    for cache in CACHES:
        # Use override path if specified (e.g., legacy HF cache at ~/huggingface),
        # otherwise use CACHE_ROOT / host_subdir.
        override = cache.get("host_dir_override")
        host_dir = Path(override) if override else CACHE_ROOT / cache["host_subdir"]
        host_dir.mkdir(parents=True, exist_ok=True)

        # Report cache size and file count for visibility.
        try:
            r = sh(f"du -sh '{host_dir}' 2>/dev/null | cut -f1", capture=True)
            size = r.stdout.strip() if r.returncode == 0 else "?"
            r2 = sh(f"find '{host_dir}' -type f 2>/dev/null | wc -l", capture=True)
            count = r2.stdout.strip() if r2.returncode == 0 else "?"
        except (OSError, subprocess.SubprocessError):
            size, count = "?", "?"

        try:
            is_warm = count != "?" and int(count) > 0
            status = "warm" if is_warm else "cold"
        except ValueError:
            status = "unknown"
        env = cache["env_var"]
        info(f"  {env:25s} {str(host_dir):45s} {size:>8s} ({count} files) [{status}]")

    return CACHES


def build_cache_docker_args():
    # type: () -> list[str]
    """Build docker run arguments for all persistent cache mounts.

    Returns a list of ``-v`` and ``-e`` flags to pass to ``docker run``:
      - ``-v host_dir:container_path`` for each cache
      - ``-e ENV_VAR=container_path`` for each cache

    This is the single place where cache mounts are defined, so adding
    a new cache type only requires editing the CACHES list.

    Returns:
        List of docker CLI arguments (strings).
    """
    args = []  # type: list[str]
    for cache in CACHES:
        override = cache.get("host_dir_override")
        host_dir = Path(override) if override else CACHE_ROOT / cache["host_subdir"]
        args += ["-v", f"{host_dir}:{cache['container_path']}"]
        args += ["-e", f"{cache['env_var']}={cache['container_path']}"]
    return args


def log_cache_stats_diff(label):
    # type: (str) -> None
    """Log cache sizes for post-mortem comparison (e.g., pre-test vs post-test).

    Called before and after test execution. By diffing the two snapshots,
    you can see which caches grew (new downloads) and by how much.

    Args:
        label: Tag for the snapshot (e.g., "pre-test", "post-test").
    """
    info(f"Cache stats [{label}]:")
    for cache in CACHES:
        override = cache.get("host_dir_override")
        host_dir = Path(override) if override else CACHE_ROOT / cache["host_subdir"]
        if not host_dir.exists():
            continue
        try:
            r = sh(f"du -sh '{host_dir}' 2>/dev/null | cut -f1", capture=True)
            size = r.stdout.strip() if r.returncode == 0 else "?"
        except (OSError, subprocess.SubprocessError):
            size = "?"
        info(f"  {cache['env_var']:25s} {size}")


# ==========================================================================
# Docker helpers
# ==========================================================================


def check_docker_health():
    # type: () -> None
    """Verify the Docker daemon is responsive before running anything.

    In K8s, the Docker socket is typically bind-mounted from the host node.
    If the host daemon is overloaded, restarting, or the mount is stale,
    ``docker`` commands will hang indefinitely. This function applies a
    short timeout to fail fast with a clear error message.

    Also checks the storage driver: overlay2 is strongly recommended for
    CI workloads (devicemapper and vfs have known performance issues).
    """
    section("Checking Docker daemon health")
    try:
        r = sh(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture=True,
            timeout=DOCKER_HEALTH_TIMEOUT_S,
        )
        if r.returncode != 0:
            error("Docker daemon is not responding")
            error(
                "Check that /var/run/docker.sock is "
                "mounted and the Docker daemon is running"
            )
            sys.exit(1)
        info(f"Docker daemon version: {r.stdout.strip()}")
    except subprocess.TimeoutExpired:
        error(f"Docker daemon did not respond within {DOCKER_HEALTH_TIMEOUT_S}s")
        error("The Docker socket may be stale or the daemon overloaded")
        sys.exit(1)

    r = sh("docker info --format '{{.Driver}}'", capture=True)
    if r.returncode == 0:
        driver = r.stdout.strip()
        if driver != "overlay2":
            warn(
                f"Docker storage driver is '{driver}' -- overlay2 is recommended for CI"
            )


def check_infra_health():
    # type: () -> None
    """Pre-flight infrastructure health checks.

    Runs a battery of checks to catch common infra problems early, before
    we spend 10 minutes pulling an image or 30 minutes running tests only
    to discover the node was degraded from the start.

    Checks performed:
      1. DNS resolution -- can we resolve hostnames? Catches broken DNS
         in K8s (kube-dns/coredns down, pod DNS policy misconfigured).
      2. Docker registry reachability -- can we reach the image registry?
         Catches network partition, proxy issues, registry outages.
      3. Available memory -- is the node under memory pressure? If free
         memory is very low, tests will OOM or the kubelet may evict us.
      4. Disk I/O latency -- is the disk responsive? Slow NFS mounts,
         degraded network storage, or worn-out SSDs cause mysterious
         test timeouts.
      5. Pod restart count -- was this pod recently restarted? Repeated
         restarts suggest a flaky node or a resource limit that keeps
         getting hit.

    None of these checks are fatal (we warn, not exit) because the test
    might still pass on degraded infra. But the warnings make post-mortem
    much faster.
    """
    section("Infrastructure health checks")

    # -- 1. DNS resolution --
    # Test with a well-known hostname. If DNS is broken, docker pull will
    # hang for minutes before failing with an opaque error.
    dns_hosts = ["ghcr.io", "docker.io", "huggingface.co"]
    for host in dns_hosts:
        r = sh(f"getent hosts {host} 2>/dev/null", capture=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            warn(
                f"{_DIAG_PREFIX} DNS resolution failed for '{host}'. "
                f"Docker pull and model downloads may fail. "
                f"Check pod DNS policy and kube-dns/coredns health."
            )
            break
    else:
        info("DNS resolution: OK")

    # -- 2. Docker registry reachability --
    # Try to reach the registry API. We don't need to authenticate --
    # a TCP connection or HTTP response is enough to confirm the network
    # path is open.
    registry = os.environ.get("VLLM_CI_REGISTRY", "docker.io")
    r = sh(
        f"curl -sf --connect-timeout 10 --max-time 15 "
        f"-o /dev/null -w '%{{http_code}}' https://{registry}/v2/ 2>/dev/null",
        capture=True,
    )
    if r.returncode != 0:
        warn(
            f"{_DIAG_PREFIX} Cannot reach Docker registry '{registry}'. "
            f"docker pull will likely fail. "
            f"Check network connectivity, proxy settings, and firewall rules."
        )
    else:
        info(f"Docker registry ({registry}): reachable")

    # -- 3. Available memory --
    r = sh("cat /proc/meminfo 2>/dev/null", capture=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith("MemAvailable:"):
                try:
                    avail_kb = int(line.split()[1])
                    avail_gb = avail_kb / (1024 * 1024)
                    info(f"Available memory: {avail_gb:.1f} GB")
                    if avail_gb < 8:
                        warn(
                            f"{_DIAG_PREFIX} Low available memory: {avail_gb:.1f} GB. "
                            f"Tests may OOM or the kubelet may evict this pod. "
                            f"Check node memory pressure: kubectl top node"
                        )
                except (ValueError, IndexError):
                    pass
                break

    # -- 4. Disk I/O latency --
    # Write a small file and measure time. Healthy disks complete in <50ms.
    # Degraded NFS or network storage can take seconds.
    try:
        test_file = Path(tempfile.gettempdir()) / ".disk_io_check"
        start = time.monotonic()
        test_file.write_bytes(b"x" * 4096)
        test_file.unlink()
        latency_ms = (time.monotonic() - start) * 1000
        if latency_ms > 500:
            warn(
                f"{_DIAG_PREFIX} Disk I/O latency: {latency_ms:.0f}ms (>500ms). "
                f"Storage may be degraded (slow NFS, worn SSD, network storage issue). "
                f"Test performance will be affected."
            )
        elif latency_ms > 100:
            warn(f"Disk I/O latency: {latency_ms:.0f}ms (elevated but usable)")
        else:
            info(f"Disk I/O latency: {latency_ms:.0f}ms (OK)")
    except OSError:
        warn("Could not perform disk I/O check")

    # -- 5. Pod restart count (K8s only) --
    if is_k8s():
        # The Downward API can expose restart count, but it's not always
        # configured. Check the container's start time as a proxy: if the
        # container was created very recently (< 60s ago), we may have just
        # been restarted.
        r = sh("cat /proc/1/stat 2>/dev/null", capture=True)
        if r.returncode == 0:
            # /proc/1/stat field 22 is the start time in clock ticks.
            # We can also check uptime more simply:
            r2 = sh("cat /proc/uptime 2>/dev/null", capture=True)
            if r2.returncode == 0:
                try:
                    uptime_s = float(r2.stdout.split()[0])
                    if uptime_s < 120:
                        warn(
                            f"{_DIAG_PREFIX} Pod uptime is only {uptime_s:.0f}s. "
                            f"This pod was recently (re)started. If tests fail, "
                            f"check for repeated restarts (kubectl describe pod) "
                            f"which may indicate a flaky node or resource limit."
                        )
                    else:
                        info(f"Pod uptime: {uptime_s:.0f}s")
                except (ValueError, IndexError):
                    pass


def docker_pull_with_retry(image, retries=DOCKER_PULL_RETRIES):
    # type: (str, int) -> None
    """Pull a Docker image with retry and exponential backoff.

    Network flakes (DNS hiccups, registry rate limits, transient TCP resets)
    are the most common cause of CI failures that are NOT test failures.
    A simple retry eliminates most of these.

    On each retry, logs a clear diagnostic message with the attempt number
    and the error from Docker.

    Args:
        image:   Full image name (e.g., "rocm/vllm-ci:abc123").
        retries: Maximum number of attempts (default: DOCKER_PULL_RETRIES).

    Raises:
        SystemExit: If all retries are exhausted.
    """
    for attempt in range(1, retries + 1):
        info(f"docker pull {image} (attempt {attempt}/{retries})")
        r = sh(["docker", "pull", image], capture=True)
        if r.returncode == 0:
            info("Pull succeeded")
            return

        # Pull failed -- diagnose and maybe retry.
        stderr = (r.stderr or r.stdout or "").strip()
        if attempt < retries:
            delay = DOCKER_PULL_RETRY_DELAY_S * attempt  # linear backoff
            warn(
                f"{_DIAG_PREFIX} docker pull failed (attempt {attempt}/{retries})\n"
                f"  Image:  {image}\n"
                f"  Error:  {stderr[:300]}\n"
                f"  Action: Retrying in {delay}s..."
            )
            time.sleep(delay)
        else:
            error(
                f"{_DIAG_PREFIX} docker pull FAILED after {retries} attempts\n"
                f"  Image:  {image}\n"
                f"  Error:  {stderr[:500]}\n"
                f"  Common causes:\n"
                f"    - Network issue (check DNS, proxy, firewall)\n"
                f"    - Registry rate limit (wait and retry, or use a mirror)\n"
                f"    - Image does not exist (check BUILDKITE_COMMIT)\n"
                f"    - Registry outage (check status page)"
            )
            sys.exit(1)


def _get_disk_info(path):
    # type: (str) -> tuple[int | None, int | None, int | None]
    """Return (usage_pct, used_gb, total_gb) for the partition containing ``path``.

    Uses ``df`` to read partition stats. Returns None for any value that
    cannot be parsed.
    """
    # df output: Filesystem 1K-blocks Used Available Use% Mounted-on
    r = sh(f"df '{path}' | tail -1", capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None, None, None

    parts = r.stdout.strip().split()
    try:
        total_kb = int(parts[1])
        used_kb = int(parts[2])
        pct_str = parts[4].rstrip("%")
        return int(pct_str), used_kb // (1024 * 1024), total_kb // (1024 * 1024)
    except (ValueError, IndexError):
        return None, None, None


def _should_evict(used_pct, used_gb, threshold_pct, threshold_gb):
    # type: (int | None, int | None, int, int) -> bool
    """Return True if disk usage exceeds the stricter of two thresholds.

    Compares percentage-based and GB-based limits and triggers eviction if
    EITHER is exceeded. This means the stricter limit always wins:

      - Small disk (100GB): 70% = 70GB. If threshold_gb=200, percentage
        triggers first (at 70GB). GB limit is irrelevant.
      - Large disk (2TB): 70% = 1.4TB. If threshold_gb=200, GB limit
        triggers first (at 200GB). Percentage is irrelevant.

    Args:
        used_pct:      Current usage as percentage (0-100).
        used_gb:       Current usage in GB.
        threshold_pct: Percentage threshold (0 = disabled).
        threshold_gb:  GB threshold (0 = disabled).

    Returns:
        True if eviction should start.
    """
    if threshold_pct > 0 and used_pct is not None and used_pct > threshold_pct:
        return True
    return threshold_gb > 0 and used_gb is not None and used_gb > threshold_gb


def _below_target(used_pct, used_gb, target_pct, target_gb):
    # type: (int | None, int | None, int, int) -> bool
    """Return True if disk usage is below BOTH targets (safe to stop evicting).

    We require both to be satisfied so we don't stop evicting when one
    limit is met but the other is still exceeded.

    Args:
        used_pct:   Current usage as percentage.
        used_gb:    Current usage in GB.
        target_pct: Percentage target (0 = disabled / always satisfied).
        target_gb:  GB target (0 = disabled / always satisfied).
    """
    pct_ok = (target_pct <= 0) or (used_pct is not None and used_pct <= target_pct)
    gb_ok = (target_gb <= 0) or (used_gb is not None and used_gb <= target_gb)
    return pct_ok and gb_ok


def cleanup_stale_containers():
    # type: () -> None
    """Remove stopped rocm_* containers older than STALE_CONTAINER_AGE_H.

    Stopped containers consume disk (their writable layer persists) and
    pollute ``docker ps -a`` output. We only remove containers that are:
      - Named ``rocm_*`` (our naming convention)
      - In a terminal state (exited, dead, or created-but-never-started)
      - Older than STALE_CONTAINER_AGE_H

    Running containers are never touched.
    """
    section("Cleaning stale containers")

    # List stopped rocm_* containers.
    r = sh(
        "docker ps -a --format '{{.ID}} {{.Names}} {{.Status}} {{.CreatedAt}}' "
        "--filter name=rocm_ "
        "--filter 'status=exited' "
        "--filter 'status=dead' "
        "--filter 'status=created'",
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        info("No stale containers found")
        return

    # Parse creation times and remove containers older than the threshold.
    now = time.time()
    threshold_s = STALE_CONTAINER_AGE_H * 3600
    removed = 0
    for line in r.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        cid = parts[0]

        # Get creation timestamp from docker inspect.
        r2 = sh(
            f"docker inspect --format '{{{{.Created}}}}' {cid} 2>/dev/null",
            capture=True,
        )
        if r2.returncode != 0:
            continue

        # Docker timestamps are ISO 8601: "2026-03-27T08:12:34.123456789Z"
        # We strip nanoseconds (Python can't parse them) and the trailing Z
        # (we assume UTC, which Docker always uses).
        raw_ts = r2.stdout.strip()
        try:
            # Drop nanosecond fractional part and trailing Z.
            clean = raw_ts.split(".")[0].rstrip("Z")
            # Python 3.7+ fromisoformat handles "YYYY-MM-DDTHH:MM:SS".
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
            created_epoch = dt.timestamp()
            age_s = now - created_epoch
            if age_s < threshold_s:
                continue
        except (ValueError, OverflowError, AttributeError) as exc:
            warn(
                f"  Skipping container {cid}: could not parse timestamp "
                f"'{raw_ts}': {exc}"
            )
            continue

        age_h = age_s / 3600
        info(f"  Removing stale container {cid} (age: {age_h:.1f}h)")
        sh(f"docker rm -f {cid} 2>/dev/null || true")
        removed += 1

    info(f"Removed {removed} stale container(s)")


def cleanup_docker_disk():
    # type: () -> None
    """LRU-based Docker cache management.

    Instead of ``docker system prune --all`` (which wipes every cached
    layer and forces cold pulls), this function implements an LRU eviction
    strategy:

    1. Check disk usage of Docker's data-root partition.
    2. If below DISK_USAGE_THRESHOLD_PCT -- do nothing.
    3. If above -- evict images in LRU order (least recently used first)
       until usage drops below DISK_USAGE_TARGET_PCT.

    This preserves recently-pulled image layers in the cache, so the next
    ``docker pull`` only downloads the delta. On a typical CI node, this
    reduces pull time from ~10 minutes (cold) to ~30 seconds (warm).

    Eviction order:
      a. Dangling images (unreferenced layers from failed builds).
      b. Unused volumes.
      c. Images sorted by last-used time, oldest first.

    The currently-running image (if any) is never evicted.
    """
    section("Docker cache management (LRU)")

    # Short-circuit: if all thresholds are disabled, skip entirely.
    if DISK_USAGE_THRESHOLD_PCT <= 0 and DISK_USAGE_THRESHOLD_GB <= 0:
        info("Disk eviction disabled (both threshold PCT and GB are 0)")
        return

    r = sh("docker info -f '{{.DockerRootDir}}'", capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        error("Failed to determine Docker root directory")
        sys.exit(1)

    docker_root = r.stdout.strip()
    used_pct, used_gb, total_gb = _get_disk_info(docker_root)
    if used_pct is None:
        warn("Could not parse disk usage -- skipping cache management")
        return

    info(f"Docker root: {docker_root}")
    info(f"  Disk: {used_gb}GB / {total_gb}GB ({used_pct}%)")
    info(
        f"  Thresholds: {DISK_USAGE_THRESHOLD_PCT}% or {DISK_USAGE_THRESHOLD_GB}GB "
        f"(whichever is stricter)"
    )
    info(f"  Targets:    {DISK_USAGE_TARGET_PCT}% or {DISK_USAGE_TARGET_GB}GB")

    if not _should_evict(
        used_pct, used_gb, DISK_USAGE_THRESHOLD_PCT, DISK_USAGE_THRESHOLD_GB
    ):
        info("Disk usage within limits -- no eviction needed")
        return

    info("Disk usage exceeds threshold -- starting LRU eviction")

    # Step A: Remove dangling images (broken/partial builds). These are
    # never useful and can be large. Always safe.
    info("  Removing dangling images...")
    sh("docker image prune -f", capture=True)

    # Step B: Remove unused volumes. Orphaned volumes accumulate from
    # containers that were removed without --volumes.
    info("  Removing unused volumes...")
    sh("docker volume prune -f", capture=True)

    # Check if that was enough.
    used_pct, used_gb, _ = _get_disk_info(docker_root)
    if _below_target(used_pct, used_gb, DISK_USAGE_TARGET_PCT, DISK_USAGE_TARGET_GB):
        info(f"  Target reached after dangling cleanup ({used_gb}GB, {used_pct}%)")
        return

    # Step C: Evict images according to the configured policy.
    info(f"  Eviction policy: {DISK_EVICTION_POLICY}")

    # List all images with metadata.
    img_fmt = "'{{.ID}}\t{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}'"
    r = sh(
        f"docker image ls --format {img_fmt} --no-trunc",
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        warn("Could not list Docker images")
        return

    # Parse image list.
    # Each entry: (img_id, name, created, size)
    raw_images = []  # type: list[tuple[str, str, str, str]]
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            raw_images.append((parts[0], parts[1], parts[2], parts[3]))

    # Protect the current job's image from eviction.
    current_commit = os.environ.get("BUILDKITE_COMMIT", "")
    protected_names = set()  # type: set[str]
    if current_commit:
        protected_names.add(f"rocm/vllm-ci:{current_commit}")

    # Filter out protected and base images.
    candidates = []  # type: list[tuple[str, str, str, str]]
    for img_id, name, created, size in raw_images:
        if name in protected_names:
            info(f"  Protected (current job): {name}")
            continue
        if name.startswith("rocm/") and ":latest" in name:
            info(f"  Protected (base image): {name}")
            continue
        candidates.append((img_id, name, created, size))

    # Sort candidates by eviction priority (evict first = index 0).
    if DISK_EVICTION_POLICY == "lfu":
        # LFU: Least Frequently Used.
        # Count how many containers (running + stopped) reference each image.
        # Images with fewer container references are evicted first -- they
        # are used by fewer jobs and are less valuable to keep cached.
        # Ties are broken by creation time (oldest first).
        info("  Counting container references per image...")
        r_ps = sh(
            "docker ps -a --format '{{.Image}}' --no-trunc",
            capture=True,
        )
        ref_counts = {}  # type: dict[str, int]
        if r_ps.returncode == 0 and r_ps.stdout.strip():
            for img_ref in r_ps.stdout.strip().splitlines():
                ref_counts[img_ref] = ref_counts.get(img_ref, 0) + 1

        # Build sort key: (reference_count, created_time) ascending.
        # Lowest ref count first; among equal counts, oldest first.
        def lfu_key(item):
            # type: (tuple[str, str, str, str]) -> tuple[int, str]
            img_id, name, created, size = item
            count = ref_counts.get(name, 0)
            # Also check by ID in case containers reference by ID not name.
            count = max(count, ref_counts.get(img_id, 0))
            return (count, created)

        candidates.sort(key=lfu_key)

        # Log the ranking for transparency.
        for img_id, name, created, size in candidates[:10]:
            count = ref_counts.get(name, ref_counts.get(img_id, 0))
            info(f"    {name:60s} refs={count}  created={created}  size={size}")
        if len(candidates) > 10:
            info(f"    ... and {len(candidates) - 10} more")

    else:
        # LRU: Least Recently Used.
        # Sort by creation time ascending (oldest first).
        # CreatedAt is a proxy for last-used time -- Docker updates it on pull.
        candidates.sort(key=lambda item: item[2])  # sort by created

    # Evict one at a time, checking disk after each removal.
    evicted = 0
    for img_id, name, created, size in candidates:
        policy_detail = ""
        if DISK_EVICTION_POLICY == "lfu":
            count = ref_counts.get(name, ref_counts.get(img_id, 0))
            policy_detail = f", refs={count}"
        info(f"  Evicting: {name} ({size}, created {created}{policy_detail})")
        sh(f"docker image rm -f {img_id} 2>/dev/null || true")
        evicted += 1

        used_pct, used_gb, _ = _get_disk_info(docker_root)
        if _below_target(
            used_pct, used_gb, DISK_USAGE_TARGET_PCT, DISK_USAGE_TARGET_GB
        ):
            info(
                f"  Target reached after evicting {evicted} image(s) "
                f"({used_gb}GB, {used_pct}%)"
            )
            return

    # Last resort: aggressive prune if policy eviction was not enough.
    used_pct, used_gb, _ = _get_disk_info(docker_root)
    if not _below_target(
        used_pct, used_gb, DISK_USAGE_TARGET_PCT, DISK_USAGE_TARGET_GB
    ):
        warn(
            f"Disk still at {used_gb}GB ({used_pct}%) after evicting "
            f"{evicted} images ({DISK_EVICTION_POLICY}). "
            f"Falling back to aggressive prune."
        )
        sh("docker builder prune -f --keep-storage=10GB 2>/dev/null || true")
        sh("docker system prune -f --filter 'until=24h' 2>/dev/null || true")
        used_pct, used_gb, _ = _get_disk_info(docker_root)
        info(f"  Disk after aggressive prune: {used_gb}GB ({used_pct}%)")


def run_container(
    *,
    image,  # type: str
    name,  # type: str
    commands,  # type: str
    render_gid,  # type: str
    results_dir,  # type: Path
    render_devices,  # type: str
    rdma,  # type: bool
):
    # type: (...) -> int
    """Run test commands inside a Docker container and return a validated exit code.

    This is the core of the script. The execution flow has 10 steps:

     0. VRAM snapshot (pre-test baseline)
     1. ``docker run --detach`` to start the container in the background
     2. ``docker logs -f`` piped through ``tee`` for real-time Buildkite
        output AND a log file (for artifact upload)
     3. ``docker wait`` with a timeout watchdog to get the exit code
     4. Wait for log streaming to flush
     5. VRAM snapshot (post-test -- compare with step 0 to detect leaks)
     6. ``docker inspect`` for OOM / signal diagnosis (before docker rm)
     7. Parse JUnit XML -- the AUTHORITATIVE exit code source (see module
        docstring for why docker's exit code cannot be trusted)
     8. Post a Buildkite annotation if tests failed
     9. Upload JUnit XML and container log as Buildkite artifacts
    10. Set Buildkite metadata (exit code, OOM status, timeout flag)

    Why ``--detach`` + ``docker wait`` instead of blocking ``docker run``?

    ``docker run`` (blocking) returns ``$?`` which is the container's exit
    code. But with ``--rm``, there is a known race condition in Docker/
    containerd where the exit code is lost during container cleanup.
    ``docker wait`` queries containerd's event stream and is not affected
    by this race.

    Why ``tee`` instead of just ``docker logs -f`` to stdout?

    We need the logs in TWO places: stdout (for Buildkite real-time display)
    and a file (for artifact upload / post-mortem).  ``tee`` does both.

    Args:
        image:          Docker image name (e.g., "rocm/vllm-ci:<commit>").
        name:           Container name (unique per job).
        commands:       Shell command string to run inside the container.
        render_gid:     Numeric GID of the "render" group (for GPU access).
        results_dir:    Host path to bind-mount for JUnit XML and logs.
        render_devices: Space-separated ``--device`` flags from Buildkite metadata.
        rdma:           True if /dev/infiniband exists (enables RDMA passthrough).

    Cache mounts are injected automatically from the CACHES registry via
    ``build_cache_docker_args()``. See the "Persistent cache configuration"
    section at the top of this file.

    Returns:
        Validated exit code (0 = pass, non-zero = fail).
    """
    docker_cmd = [
        "docker",
        "run",
        "--detach",
        "--device",
        "/dev/kfd",
    ]  # type: list[str]

    # Render devices are passed as a space-separated string of --device flags
    # from the Buildkite agent's metadata (set by the node's agent config).
    # Validate each token: must start with --device or be a /dev/ path.
    if render_devices:
        tokens = render_devices.split()
        for token in tokens:
            if not (token.startswith("--device") or token.startswith("/dev/")):
                warn(
                    f"Unexpected render_devices token: '{token}' "
                    f"(expected --device or /dev/ path)"
                )
        docker_cmd.extend(tokens)

    # RDMA passthrough for ibverbs-based tests (e.g., test_moriio_connector).
    if rdma:
        docker_cmd += ["--device", "/dev/infiniband", "--cap-add=IPC_LOCK"]

    docker_cmd += [
        "--network=host",
        f"--shm-size={CONTAINER_SHM_SIZE}",
        "--group-add",
        render_gid,
        f"--pids-limit={CONTAINER_PIDS_LIMIT}",
        # IPC namespace: share the host's IPC namespace so that PyTorch
        # multiprocessing workers (torchrun, torch.distributed.launch) can
        # communicate via shared memory. Without this, multi-GPU tests that
        # use NCCL or Gloo backends fail with "Bus error" or "Connection
        # refused" when workers try to open /dev/shm segments from siblings.
        "--ipc=host",
        # Pass-through env vars (values come from the Buildkite agent env).
        "-e",
        "HF_TOKEN",
        "-e",
        "AWS_ACCESS_KEY_ID",
        "-e",
        "AWS_SECRET_ACCESS_KEY",
        "-e",
        "BUILDKITE_PARALLEL_JOB",
        "-e",
        "BUILDKITE_PARALLEL_JOB_COUNT",
        # Results mount (JUnit XML, container logs).
        "-v",
        f"{results_dir}:{RESULTS_MOUNT}",
        # Container-only env vars.
        "-e",
        "PYTHONPATH=..",
        "-e",
        "PYTORCH_ROCM_ARCH=",
        # NCCL tuning for ROCm multi-GPU tests.
        # NCCL_DEBUG=WARN: log NCCL warnings (INFO is too noisy for CI).
        # NCCL_CUMEM_HOST_ENABLE=0: workaround for NCCL host memory issue
        #   (see https://github.com/NVIDIA/nccl/issues/1838).
        "-e",
        "NCCL_DEBUG=WARN",
        "-e",
        "NCCL_CUMEM_HOST_ENABLE=0",
        # Tell pytest to write JUnit XML to the bind-mounted results dir.
        # This is the KEY to the exit-code fix: the XML is written BEFORE
        # Python's atexit handlers run, and it persists on the host.
        "-e",
        f"PYTEST_ADDOPTS=--junitxml={RESULTS_MOUNT}/results.xml",
    ]

    # Persistent cache mounts -- all caches defined in the CACHES registry.
    docker_cmd += build_cache_docker_args()

    docker_cmd += [
        "--name",
        name,
        image,
        "/bin/bash",
        "-c",
        commands,
    ]

    # -- Step 0: Pre-test baselines --
    snapshot_gpu_vram("pre-test")
    log_cache_stats_diff("pre-test")

    # -- Step 1: Start container in background --
    # Log the full docker command for reproducibility. Truncate at the
    # bash -c boundary to avoid logging the entire test command twice.
    cmd_summary = (
        docker_cmd[: docker_cmd.index("/bin/bash")]
        if "/bin/bash" in docker_cmd
        else docker_cmd
    )
    info(f"Docker run command: {' '.join(cmd_summary)} /bin/bash -c '<commands>'")

    r = sh(docker_cmd, check=True, capture=True)
    container_id = r.stdout.strip()
    info(f"Container started: {container_id[:12]} (full ID: {container_id})")

    # -- Step 2: Stream logs to stdout (Buildkite) AND a file (artifact) --
    log_file = results_dir / "container.log"
    info(f"Container log will be saved to: {log_file}")
    log_fd = open(log_file, "w")  # noqa: SIM115
    log_proc = subprocess.Popen(
        ["docker", "logs", "-f", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tee_proc = subprocess.Popen(
        ["tee", str(log_file)],
        stdin=log_proc.stdout,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # Close our copy of the pipe so log_proc gets SIGPIPE if tee dies.
    log_proc.stdout.close()

    # -- Step 3: Wait for container exit with timeout watchdog --
    timed_out = False
    wait_start = time.monotonic()
    try:
        r = sh(["docker", "wait", name], capture=True, timeout=CONTAINER_TIMEOUT_S)
        wait_elapsed = time.monotonic() - wait_start
        info(
            f"docker wait returned after {wait_elapsed:.1f}s "
            f"(raw output: '{r.stdout.strip()}')"
        )
        try:
            exit_code = int(r.stdout.strip())
        except (ValueError, AttributeError):
            warn(
                f"docker wait returned non-integer: "
                f"'{r.stdout.strip()}' -- defaulting to 1"
            )
            exit_code = 1
    except subprocess.TimeoutExpired:
        wait_elapsed = time.monotonic() - wait_start
        timed_out = True
        warn(
            f"Container exceeded timeout after {wait_elapsed:.1f}s "
            f"(limit: {CONTAINER_TIMEOUT_S}s) -- killing"
        )
        sh(f"docker kill {name} 2>/dev/null || true")
        exit_code = 124  # matches GNU timeout convention

    # -- Step 4: Flush log streaming --
    info("Flushing container log stream...")
    try:
        tee_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        warn("Log stream flush timed out after 30s -- killing tee process")
        tee_proc.kill()
        tee_proc.wait()
    log_fd.close()

    info(f"Container exit code (docker wait): {exit_code}")

    # -- Step 5: Post-test snapshots (compare with step 0 for leaks/growth) --
    snapshot_gpu_vram("post-test")
    log_cache_stats_diff("post-test")

    # -- Step 6: Diagnose exit (OOM, signals, PID limit) BEFORE docker rm --
    diag = diagnose_container_exit(name, log_file=log_file)

    if diag["oom_killed"]:
        annotate_build(
            "### :boom: Container OOM Killed\n\n"
            "The test container ran out of memory.\n"
            "See `[run-amd-test.py diagnostics]` in the build log for details.",
            style="error",
            context="oom-kill",
        )
        if exit_code == 0:
            exit_code = 137

    if diag["pids_exhausted"]:
        annotate_build(
            "### :no_entry: PID Limit Exhausted "
            f"(--pids-limit={CONTAINER_PIDS_LIMIT})\n\n"
            "The container could not fork new processes/threads.\n"
            "See `[run-amd-test.py diagnostics]` in the build log for details.",
            style="error",
            context="pids-exhausted",
        )
        if exit_code == 0:
            exit_code = 1

    if timed_out:
        timeout_min = CONTAINER_TIMEOUT_S // 60
        error(
            f"{_DIAG_PREFIX} CONTAINER TIMEOUT\n"
            f"  What happened: The test container did not finish within\n"
            f"                 {CONTAINER_TIMEOUT_S}s ({timeout_min} min).\n"
            f"  Container:     {name}\n"
            f"  Exit code:     124 (timeout)\n"
            f"  How to fix:\n"
            f"    - Check if the test is hung (deadlock, infinite loop)\n"
            f"    - Increase timeout_in_minutes in test-amd.yaml (currently\n"
            f"      controls the Buildkite step timeout; our container timeout\n"
            f"      is set 10 min shorter as a buffer)\n"
            f"    - Override with VLLM_TEST_TIMEOUT=<seconds> env var\n"
            f"    - Current timeout: "
            f"{CONTAINER_TIMEOUT_S}s ({timeout_min} min)"
        )
        annotate_build(
            f"### :alarm_clock: Test Timeout ({timeout_min} min)\n\n"
            "The test container exceeded the maximum allowed runtime.\n"
            "See `[run-amd-test.py diagnostics]` in the build log for details.",
            style="error",
            context="timeout",
        )

    # -- Step 7: JUnit XML -- the AUTHORITATIVE result source --
    # Pytest writes the XML during pytest_sessionfinish, BEFORE Python's
    # atexit handlers run. Libraries that register atexit hooks calling
    # os._exit(0) overwrite pytest's exit code, but the XML is already
    # on disk (bind-mounted to the host) at that point.
    xml_path = results_dir / "results.xml"
    if exit_code == 0:
        failures = parse_junit_failures(xml_path)
        if failures is not None and failures > 0:
            error(
                f"{_DIAG_PREFIX} EXIT CODE OVERRIDE (atexit hook detected)\n"
                f"  What happened: The container exited 0 but JUnit XML reports\n"
                f"                 {failures} failure(s)/error(s). This means a\n"
                f"                 library's atexit hook called os._exit(0) and\n"
                f"                 overwrote pytest's real exit code.\n"
                f"  Container:     {name}\n"
                f"  JUnit XML:     {xml_path}\n"
                f"  Action:        Overriding exit code from 0 to 1."
            )
            exit_code = 1

    # -- Step 8: Buildkite annotation with failure details --
    if exit_code != 0:
        annotation = build_failure_annotation(xml_path)
        if annotation:
            annotate_build(annotation, style="error", context="test-failures")

    # -- Step 9: Upload artifacts for post-mortem --
    section("Uploading artifacts")
    if xml_path.is_file():
        info(f"JUnit XML: {xml_path} ({xml_path.stat().st_size} bytes)")
    if log_file.is_file():
        info(f"Container log: {log_file} ({log_file.stat().st_size} bytes)")
    upload_artifacts(f"{results_dir}/*.xml", f"{results_dir}/*.log")

    # -- Step 10: Build metadata for downstream steps / dashboards --
    set_buildkite_meta("test_exit_code", str(exit_code))
    set_buildkite_meta("test_oom_killed", str(diag["oom_killed"]))
    if timed_out:
        set_buildkite_meta("test_timed_out", "true")

    return exit_code


# ==========================================================================
# Command processing
#
# Test commands arrive via one of two paths:
#   1. VLLM_TEST_COMMANDS env var (preferred -- preserves all quoting)
#   2. Positional args (legacy -- inner double-quotes get stripped)
#
# After sourcing, commands go through two transforms:
#   1. re_quote_pytest_markers: fixes mangled -m/-k quoting
#   2. apply_rocm_overrides: adds --ignore flags for ROCm-unsupported tests
# ==========================================================================


def get_commands(argv):
    # type: (list[str]) -> str
    """Resolve test commands from VLLM_TEST_COMMANDS env var or argv.

    Preference order:
      1. VLLM_TEST_COMMANDS environment variable (preserves quoting).
      2. Positional arguments joined with spaces (legacy, lossy).

    Exits with code 1 if no commands are provided.

    Args:
        argv: sys.argv (program name + positional args).

    Returns:
        Command string to pass to ``bash -c`` inside the container.
    """
    env = os.environ.get("VLLM_TEST_COMMANDS", "")
    if env:
        info("Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)")
        return env

    if len(argv) > 1:
        commands = " ".join(argv[1:])
        info("Commands sourced from positional args (legacy mode)")
        warn(
            "Inner double-quotes may have been stripped by the calling shell.\n"
            "  Prefer: VLLM_TEST_COMMANDS='...' python3 run-amd-test.py"
        )
        return commands

    error("No test commands provided.")
    error(f"  Preferred:  VLLM_TEST_COMMANDS='...' python3 {argv[0]}")
    error(f'  Legacy:     python3 {argv[0]} "commands here"')
    sys.exit(1)


# --------------------------------------------------------------------------
# Pytest marker re-quoting
#
# When commands transit Buildkite -> YAML -> shell -> $* -> bash -c, quotes
# around multi-word -m/-k expressions get stripped:
#
#   pytest -m 'not cpu_test' tests/   becomes   pytest -m not cpu_test tests/
#
# Pytest then interprets "cpu_test" as a file path, not part of the marker.
# This function detects unquoted multi-word expressions after -m/-k and
# re-wraps them in single quotes.
# --------------------------------------------------------------------------

_ENV_VAR_ASSIGN = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def _is_marker_boundary(word):
    # type: (str) -> bool
    """Return True if ``word`` cannot be part of a pytest marker expression.

    Used by ``re_quote_pytest_markers`` to detect where a marker expression
    ends. Boundary tokens include:

      - Command separators: &&  ||  ;  |  (backslash)
      - Long flags:         --ignore, --shard-id, etc.
      - Short flags:        -v, -s, -k, -m  (single letter after dash)
      - File paths:         anything containing /
      - Test files:         anything ending in .py or containing .py::
      - Env assignments:    FOO=bar (uppercase variable name before =)

    Args:
        word: A single whitespace-delimited token.

    Returns:
        True if this token is a boundary (not part of a marker expression).
    """
    if word in ("&&", "||", ";", "|", "\\"):
        return True
    if word.startswith("--"):
        return True
    if len(word) == 2 and word[0] == "-" and word[1].isalpha():
        return True
    if "/" in word:
        return True
    if word.endswith(".py") or ".py::" in word:
        return True
    return bool(_ENV_VAR_ASSIGN.match(word))


def re_quote_pytest_markers(command):
    # type: (str) -> str
    """Re-quote multi-word pytest -m/-k expressions that lost quotes in transit.

    Algorithm:
      1. Flatten line continuations and split on whitespace.
      2. Scan for -m or -k tokens.
      3. Collect subsequent tokens until a boundary is reached.
      4. If the collected expression is multi-word or contains parentheses,
         wrap it in single quotes.
      5. If it already contains a single quote (already quoted upstream),
         pass it through unchanged.

    Single-word markers (e.g., ``-m hybrid_model``) pass through unquoted.

    Args:
        command: Full command string (may contain && chains, exports, etc.).

    Returns:
        Command string with marker expressions properly quoted.
    """
    flat = command.replace("\\\n", " ").replace("\n", " ")
    words = flat.split()
    result = []  # type: list[str]
    i = 0

    while i < len(words):
        word = words[i]

        if word not in ("-m", "-k"):
            result.append(word)
            i += 1
            continue

        # Found -m or -k: collect marker tokens until a boundary.
        result.append(word)
        i += 1
        tokens = []  # type: list[str]

        while i < len(words):
            w = words[i]
            # Already contains a single quote -> already quoted upstream.
            if "'" in w:
                if tokens:
                    result.append(" ".join(tokens))
                    tokens = []
                result.append(w)
                i += 1
                break
            if _is_marker_boundary(w):
                break  # don't consume the boundary token
            tokens.append(w)
            i += 1

        if tokens:
            expr = " ".join(tokens)
            # Strip wrapping double-quotes left by shell mangling.
            if len(expr) > 1 and expr[0] == '"' and expr[-1] == '"':
                expr = expr[1:-1]
            # Re-wrap if multi-word or contains grouping parens.
            if " " in expr or "(" in expr:
                result.append(f"'{expr}'")
            else:
                result.append(expr)

    return " ".join(result)


# --------------------------------------------------------------------------
# ROCm-specific test overrides
#
# Some tests are not yet supported or behave differently on AMD/ROCm
# hardware. Rather than maintaining skip markers in every test file
# (which would require upstream changes), we inject --ignore flags and
# -k filters at the command level.
#
# All overrides are defined in two dictionaries:
#   _APPEND_IGNORES:  pattern -> files to --ignore (appended to command end)
#   _INLINE_IGNORES:  pattern -> files to --ignore (inserted after directory)
# --------------------------------------------------------------------------

_APPEND_IGNORES = {
    " kernels/core": [
        "kernels/core/test_fused_quant_layernorm.py",
        "kernels/core/test_permute_cols.py",
    ],
    " kernels/attention": [
        "kernels/attention/test_attention_selector.py",
        "kernels/attention/test_encoder_decoder_attn.py",
        "kernels/attention/test_flash_attn.py",
        "kernels/attention/test_flashinfer.py",
        "kernels/attention/test_prefix_prefill.py",
        "kernels/attention/test_cascade_flash_attn.py",
        "kernels/attention/test_mha_attn.py",
        "kernels/attention/test_lightning_attn.py",
        "kernels/attention/test_attention.py",
    ],
    " kernels/quantization": [
        "kernels/quantization/test_int8_quant.py",
        "kernels/quantization/test_machete_mm.py",
        "kernels/quantization/test_block_fp8.py",
        "kernels/quantization/test_block_int8.py",
        "kernels/quantization/test_marlin_gemm.py",
        "kernels/quantization/test_cutlass_scaled_mm.py",
        "kernels/quantization/test_int8_kernel.py",
    ],
    " kernels/mamba": [
        "kernels/mamba/test_mamba_mixer2.py",
        "kernels/mamba/test_causal_conv1d.py",
        "kernels/mamba/test_mamba_ssm_ssd.py",
    ],
    " kernels/moe": [
        "kernels/moe/test_moe.py",
        "kernels/moe/test_cutlass_moe.py",
    ],
    " entrypoints/serve": [
        "entrypoints/serve/lora/test_lora_adapters.py",
    ],
}  # type: dict[str, list[str]]

_INLINE_IGNORES = {
    " entrypoints/openai ": [
        "entrypoints/openai/chat_completion/test_audio.py",
        "entrypoints/openai/completion/test_shutdown.py",
        "entrypoints/openai/test_completion.py",
        "entrypoints/openai/models/test_models.py",
        "entrypoints/openai/test_return_tokens_as_ids.py",
        "entrypoints/openai/chat_completion/test_root_path.py",
        "entrypoints/openai/completion/test_prompt_validation.py",
    ],
    " entrypoints/llm ": [
        "entrypoints/llm/test_chat.py",
        "entrypoints/llm/test_accuracy.py",
        "entrypoints/llm/test_init.py",
        "entrypoints/llm/test_prompt_validation.py",
    ],
}  # type: dict[str, list[str]]


def apply_rocm_overrides(command):
    # type: (str) -> str
    """Rewrite a pytest command with ROCm-specific --ignore flags and -k filters.

    Three types of overrides:

    1. Model registry -k filter: Excludes model classes that are not yet
       supported on ROCm (e.g., Mamba variants that need CUDA-only kernels).

    2. Append ignores: If the command mentions a test directory (e.g.,
       ``kernels/core``), append ``--ignore=`` flags for unsupported test
       files. These go at the end of the command.

    3. Inline ignores: If the command mentions a directory as a positional
       arg followed by a space (e.g., ``entrypoints/openai ``), insert
       ``--ignore=`` flags immediately after the directory name. This is
       needed when the directory is followed by other flags.

    Args:
        command: Full pytest command string.

    Returns:
        Modified command string with ROCm overrides applied.
    """
    # Model registry filter.
    needle = "pytest -v -s models/test_registry.py"
    if needle in command:
        command = command.replace(
            needle,
            f"{needle} -k 'not BambaForCausalLM and not GritLM "
            f"and not Mamba2ForCausalLM and not Zamba2ForCausalLM'",
        )

    # Append --ignore flags at end of command.
    for pattern, files in _APPEND_IGNORES.items():
        if pattern in command:
            flags = " ".join(f"--ignore={f}" for f in files)
            command = f"{command} {flags}"

    # Insert --ignore flags after a directory token.
    for pattern, files in _INLINE_IGNORES.items():
        if pattern in command:
            flags = " ".join(f"--ignore={f}" for f in files)
            command = command.replace(pattern, f"{pattern}{flags} ")

    return command


# ==========================================================================
# Result validation
# ==========================================================================


def parse_junit_failures(xml_path):
    # type: (Path) -> int | None
    """Parse a JUnit XML file and return the total number of failures + errors.

    Handles both single-suite (``<testsuite>``) and multi-suite
    (``<testsuites>``) root elements.

    Args:
        xml_path: Path to the JUnit XML file.

    Returns:
        Integer count of failures + errors, or None if the file does not
        exist or cannot be parsed.
    """
    if not xml_path.is_file():
        return None
    try:
        root = ET.parse(str(xml_path)).getroot()
        if root.tag == "testsuites":
            return sum(
                int(ts.get("failures", 0)) + int(ts.get("errors", 0)) for ts in root
            )
        return int(root.get("failures", 0)) + int(root.get("errors", 0))
    except Exception as exc:
        warn(f"Could not parse {xml_path}: {exc}")
        return None


def normalize_pytest_exit(code):
    # type: (int) -> int
    """Normalize special pytest exit codes.

    Exit code 5 means "no tests were collected". In shard-based parallel
    test execution, some shards may legitimately have no tests assigned.
    We treat this as success to avoid false-negative CI failures.

    Args:
        code: Raw pytest exit code.

    Returns:
        Normalized exit code (0 for success, non-zero for failure).
    """
    if code == PYTEST_NO_TESTS_COLLECTED:
        info("Pytest exit code 5 (no tests collected) -- treating as success.")
        return 0
    return code


# ==========================================================================
# Multi-node support (K8s pod-aware)
#
# Multi-node tests use a bracket syntax in the command string:
#   prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]
#
# This function parses the syntax and delegates to run-multi-node-test.sh,
# which creates a Docker network, starts one container per node, and
# initializes a Ray cluster for distributed testing.
# ==========================================================================


def is_multi_node(commands):
    # type: (str) -> bool
    """Return True if the command string represents a multi-node job.

    Detection methods:
      1. NUM_NODES env var > 1 (set explicitly by the pipeline YAML).
      2. Bracket syntax: ``[...] && [...]`` in the command string.

    Args:
        commands: Full command string.
    """
    if int(os.environ.get("NUM_NODES", "1")) > 1:
        return True
    return bool(re.search(r"\[.*\].*&&.*\[.*\]", commands))


def _get_pod_ip():
    # type: () -> str | None
    """Discover this pod's IP address for NCCL multi-node communication.

    In K8s, ``--network=host`` on ``docker run`` gives the container the
    POD's network namespace, not the host node's. For multi-node NCCL,
    each node needs to know the head node's IP. The pod IP is the correct
    value to use for MASTER_ADDR.

    Discovery methods (in order of preference):
      1. POD_IP env var (requires K8s downward API in the pod spec).
      2. ``hostname -I`` (first non-loopback IP).

    Returns:
        IP address string, or None if discovery fails.
    """
    pod_ip = os.environ.get("POD_IP", "")
    if pod_ip:
        return pod_ip

    r = sh("hostname -I 2>/dev/null | awk '{print $1}'", capture=True)
    if r.returncode == 0 and r.stdout.strip():
        ip = r.stdout.strip()
        if ip and not ip.startswith("127."):
            return ip

    return None


def _cleanup_multi_node():
    # type: () -> None
    """Clean up multi-node Docker containers and network.

    Stops all nodeN containers and removes the docker-net network.
    Called after multi-node tests complete (success or failure) and
    also registered for cleanup on unexpected exit.
    """
    max_nodes = int(os.environ.get("NUM_NODES", "2"))
    info(f"Multi-node cleanup: stopping {max_nodes} node containers")
    for n in range(max_nodes):
        info(f"  Stopping and removing node{n}")
        sh(f"docker stop node{n} 2>/dev/null || true")
        sh(f"docker rm -f node{n} 2>/dev/null || true")
    info("  Removing docker-net network")
    sh("docker network rm docker-net 2>/dev/null || true")


def _inject_junit_into_multi_node_cmd(cmd, node_idx, results_host_dir):
    # type: (str, int, Path) -> str
    """Wrap a multi-node test command to produce JUnit XML.

    The original multi-node bash script (run-multi-node-test.sh) runs
    commands via ``docker exec``, which has the same torch atexit exit-code
    problem as ``docker run``. We inject PYTEST_ADDOPTS to produce JUnit
    XML inside the container, then copy it out after the test completes.

    Each node gets a unique XML filename to avoid collisions:
      node0 -> results_node0.xml
      node1 -> results_node1.xml

    Args:
        cmd:              Original pytest command string for this node.
        node_idx:         Node index (0 = head, 1+ = workers).
        results_host_dir: Host directory for results (bind-mounted or copied).

    Returns:
        Modified command string with JUnit XML output enabled.
    """
    # The XML path is inside the container. We'll copy it out after the
    # test using ``docker cp``.
    xml_path = f"/tmp/results_node{node_idx}.xml"
    # Inject PYTEST_ADDOPTS before the command. If the command already
    # sets PYTEST_ADDOPTS, this prepends (pytest merges them).
    return f"export PYTEST_ADDOPTS='--junitxml={xml_path}' && {cmd}"


def run_multi_node(commands, image, results_dir):
    # type: (str, str, Path) -> int
    """Parse bracket syntax and run multi-node tests with full safety features.

    Multi-node tests use run-multi-node-test.sh which:
      1. Creates a Docker network (192.168.10.0/24).
      2. Starts N containers (one per node) in detached mode.
      3. Initializes a Ray cluster across the containers.
      4. Runs per-node commands via ``docker exec``.

    The ``docker exec`` exit code has the same torch atexit problem as
    ``docker run``: PyTorch's atexit hook can call os._exit(0) and mask
    pytest's real exit code.

    To handle this, we:
      - Inject PYTEST_ADDOPTS with --junitxml into each per-node command.
      - After execution, ``docker cp`` the JUnit XMLs from each node
        container to the host results directory.
      - Parse all XMLs and override the exit code if any report failures.

    Expected command format::

        prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]

    Args:
        commands:    Full command string with bracket syntax.
        image:       Docker image name.
        results_dir: Host directory for JUnit XML and log artifacts.

    Returns:
        Validated exit code.
    """
    section("Multi-node job detected")

    m = re.match(r"^(.*)\[(.*?)]\s*&&\s*\[(.*?)]$", commands)
    if not m:
        error("Failed to parse multi-node bracket syntax.")
        error("Expected: prefix ; [cmd1, cmd2] && [cmd1, cmd2]")
        error(f"Got: {commands}")
        return 111

    prefix = m.group(1).replace(";", "").strip()
    info(f"PREFIX: {prefix}")

    node0_cmds = [c.strip().strip('"') for c in m.group(2).split(",")]
    node1_cmds = [c.strip().strip('"') for c in m.group(3).split(",")]

    if len(node0_cmds) != len(node1_cmds):
        warn(
            f"node0 has {len(node0_cmds)} commands, "
            f"node1 has {len(node1_cmds)} -- pairing by index"
        )

    # In K8s: set MASTER_ADDR to this pod's IP for NCCL discovery.
    pod_ip = _get_pod_ip()
    existing_master = os.environ.get("MASTER_ADDR")
    if pod_ip and not existing_master:
        info(f"K8s multi-node: setting MASTER_ADDR={pod_ip} (via pod IP discovery)")
        os.environ["MASTER_ADDR"] = pod_ip
    elif existing_master:
        info(f"MASTER_ADDR already set to {existing_master} -- not overriding")
    else:
        warn(
            "Could not discover pod IP and MASTER_ADDR is not set -- "
            "multi-node NCCL communication may fail"
        )

    num_nodes = int(os.environ.get("NUM_NODES", "2"))
    info(f"NUM_NODES: {num_nodes}")

    # VRAM snapshot before multi-node tests.
    snapshot_gpu_vram("multi-node-pre-test")

    # Log parsed per-node commands for debugging.
    info(f"Node 0 commands ({len(node0_cmds)}):")
    for i, cmd in enumerate(node0_cmds):
        info(f"  [{i}] {cmd}")
    info(f"Node 1 commands ({len(node1_cmds)}):")
    for i, cmd in enumerate(node1_cmds):
        info(f"  [{i}] {cmd}")

    # Inject JUnit XML output into each per-node command.
    modified_node0 = []  # type: list[str]
    modified_node1 = []  # type: list[str]
    for i, (cmd0, cmd1) in enumerate(zip(node0_cmds, node1_cmds)):
        mod0 = _inject_junit_into_multi_node_cmd(cmd0, 0, results_dir)
        mod1 = _inject_junit_into_multi_node_cmd(cmd1, 1, results_dir)
        modified_node0.append(mod0)
        modified_node1.append(mod1)
        info(f"  Pair [{i}] node0 (with JUnit): {mod0[:120]}...")
        info(f"  Pair [{i}] node1 (with JUnit): {mod1[:120]}...")

    # Build the composite command that run-multi-node-test.sh will execute.
    composite = "(command rocm-smi || true)"
    for cmd0, cmd1 in zip(modified_node0, modified_node1):
        # shlex.quote() prevents shell injection if commands ever contain
        # metacharacters (single quotes, backticks, $(), etc.).
        step = (
            "./.buildkite/scripts/run-multi-node-test.sh "
            f"/vllm-workspace/tests 2 2 {shlex.quote(image)} "
            f"{shlex.quote(cmd0)} {shlex.quote(cmd1)}"
        )
        info(f"COMMANDS: {step}")
        composite = f"{composite} && {step}"

    # Execute with timeout.
    timed_out = False
    exec_start = time.monotonic()
    try:
        r = subprocess.run(
            ["/bin/bash", "-c", composite],
            timeout=CONTAINER_TIMEOUT_S,
        )
        exit_code = r.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        warn(f"Multi-node test exceeded timeout ({CONTAINER_TIMEOUT_S}s)")
        exit_code = 124
        annotate_build(
            f"### :alarm_clock: Multi-node Test Timeout ({CONTAINER_TIMEOUT_S}s)\n\n"
            "Set `VLLM_TEST_TIMEOUT` env var to override (default: 7200s).",
            style="error",
            context="timeout",
        )
    exec_elapsed = time.monotonic() - exec_start

    info(f"Multi-node composite exit code: {exit_code} (ran for {exec_elapsed:.1f}s)")

    # Copy JUnit XMLs from node containers to the host results directory.
    # The containers may still exist if run-multi-node-test.sh's trap hasn't
    # fired yet, or if it used --rm and they're already gone.
    info("Collecting JUnit XML from node containers...")
    total_failures = 0
    for node_idx in range(num_nodes):
        container_xml = f"/tmp/results_node{node_idx}.xml"
        host_xml = results_dir / f"results_node{node_idx}.xml"
        info(f"  docker cp node{node_idx}:{container_xml} -> {host_xml}")
        r = sh(
            f"docker cp node{node_idx}:{container_xml} {host_xml} 2>/dev/null",
            capture=True,
        )
        if r.returncode == 0 and host_xml.is_file():
            xml_size = host_xml.stat().st_size
            failures = parse_junit_failures(host_xml)
            if failures is not None:
                info(
                    f"  Node {node_idx}: JUnit XML ({xml_size}B) {failures} failure(s)"
                )
                total_failures += failures
            else:
                warn(f"  Node {node_idx}: JUnit XML ({xml_size}B) could not be parsed")
        else:
            # CompletedProcess always has .stderr when capture=True.
            stderr_msg = (r.stderr or "").strip() or "container may have been removed"
            warn(f"  Could not copy JUnit XML from node{node_idx}: {stderr_msg}")

    # JUnit XML validation: override exit code if any node reported failures.
    if exit_code == 0 and total_failures > 0:
        warn(
            f"Multi-node composite exited 0 but JUnit XML reports "
            f"{total_failures} total failure(s) -- overriding exit code to 1."
        )
        exit_code = 1

    # Buildkite annotation for multi-node failures.
    if exit_code != 0:
        # Try to build annotation from whichever node XML exists.
        for node_idx in range(int(os.environ.get("NUM_NODES", "2"))):
            host_xml = results_dir / f"results_node{node_idx}.xml"
            annotation = build_failure_annotation(host_xml)
            if annotation:
                annotate_build(
                    f"**Node {node_idx}:**\n\n{annotation}",
                    style="error",
                    context=f"test-failures-node{node_idx}",
                )

    # OOM detection for each node container (before cleanup removes them).
    for node_idx in range(int(os.environ.get("NUM_NODES", "2"))):
        diag = diagnose_container_exit(f"node{node_idx}")
        if diag["oom_killed"]:
            annotate_build(
                f"### :boom: Node {node_idx} OOM Killed\n\n"
                "Consider reducing per-node memory usage or increasing node memory.",
                style="error",
                context=f"oom-kill-node{node_idx}",
            )
            if exit_code == 0:
                exit_code = 137

    # VRAM snapshot after multi-node tests.
    snapshot_gpu_vram("multi-node-post-test")

    # Upload multi-node artifacts.
    upload_artifacts(f"{results_dir}/*.xml")

    # Set metadata.
    set_buildkite_meta("test_exit_code", str(exit_code))
    if timed_out:
        set_buildkite_meta("test_timed_out", "true")

    # Cleanup multi-node containers and network.
    _cleanup_multi_node()

    return exit_code


# ==========================================================================
# Cleanup (runs on ANY exit: normal, exception, signal)
#
# Registered via atexit.register() and signal handlers for SIGTERM/SIGINT.
# The _done flag makes it idempotent: if the signal handler calls run()
# and then sys.exit() triggers atexit, the second call is a no-op.
# ==========================================================================


class _Cleanup:
    """Idempotent resource cleanup, registered with atexit and signal handlers.

    Tracks the container name, image name, results directory, and commit
    hash so it can clean up even if main() is interrupted mid-execution.

    Cleanup steps (in order):
      1. Remove the test container (``docker rm -f``).
      2. Kill GPU zombie processes via fuser (same PID namespace).
      3. Kill GPU zombie processes via rocm-smi (cross PID namespace).
      4. Reset GPU state file (write "reset", locked).
      5. Remove stale containers from this commit.
      6. Remove the Docker image.
      7. Remove the results directory from disk.
      8. Set Buildkite metadata to confirm cleanup completed.
    """

    def __init__(self):
        # type: () -> None
        self.container = None  # type: str | None
        self.image = None  # type: str | None
        self.results_dir = None  # type: Path | None
        self.commit = os.environ.get("BUILDKITE_COMMIT", "")  # type: str
        self.is_multi_node = False  # type: bool
        self._done = False  # type: bool

    def run(self):
        # type: () -> None
        """Execute all cleanup steps. Idempotent (safe to call multiple times)."""
        if self._done:
            return
        self._done = True

        section("Cleanup: tearing down test environment")

        # 1. Remove test container(s).
        if self.container:
            info(f"  [1/8] Removing test container: {self.container}")
            sh(f"docker rm -f {self.container} 2>/dev/null || true")
        else:
            info("  [1/8] No container to remove (not started)")
        if self.is_multi_node:
            info("  [1/8] Cleaning up multi-node containers and network")
            _cleanup_multi_node()

        # 2. Kill GPU zombies (fuser -- same PID namespace).
        info("  [2/8] Checking for GPU zombies (fuser)")
        kfd_pids = _device_pids("/dev/kfd")
        if kfd_pids:
            info(f"         Found {len(kfd_pids)} process(es) on /dev/kfd")
        for pid in kfd_pids:
            _kill_pid(pid)
        dri = Path("/dev/dri")
        if dri.is_dir():
            for dev in sorted(dri.glob("renderD*")):
                for pid in _device_pids(str(dev)):
                    _kill_pid(pid)

        # 3. Kill GPU zombies (rocm-smi -- cross PID namespace).
        info("  [3/8] Checking for GPU zombies (rocm-smi, cross-namespace)")
        smi_pids = rocm_smi_gpu_pids()
        if smi_pids:
            info(f"         Found {len(smi_pids)} process(es) via rocm-smi")
        for pid in smi_pids:
            _kill_pid(pid)

        # 4. Reset GPU state (locked, non-blocking).
        if _gpu_state_file_available() and os.access(str(GPU_STATE_FILE), os.W_OK):
            info("  [4/8] Resetting GPU state file")
            try:
                with _gpu_state_lock(blocking=False):
                    GPU_STATE_FILE.write_text("reset\n")
                    info("         GPU state reset requested")
            except OSError as exc:
                warn(f"         Could not reset GPU state: {exc}")
        else:
            info("  [4/8] GPU state file not available -- skipping")

        # 5. Remove stale containers from this commit.
        if self.commit:
            info(f"  [5/8] Removing stale containers for commit {self.commit[:12]}")
            r = sh(
                f"docker ps -a --filter name=rocm_{self.commit} -q",
                capture=True,
            )
            stale = (
                r.stdout.strip().splitlines()
                if r.returncode == 0 and r.stdout.strip()
                else []
            )
            if stale:
                info(f"         Found {len(stale)} container(s) to remove")
                for cid in stale:
                    sh(f"docker rm -f {cid} 2>/dev/null || true")
            else:
                info("         None found")
        else:
            info("  [5/8] No commit hash -- skipping stale container cleanup")

        # 6. Remove Docker image.
        if self.image:
            info(f"  [6/8] Removing Docker image: {self.image}")
            sh(f"docker image rm -f {self.image} 2>/dev/null || true")
        else:
            info("  [6/8] No image to remove")

        # 7. Remove results directory (artifacts already uploaded).
        if self.results_dir and self.results_dir.exists():
            info(f"  [7/8] Removing results directory: {self.results_dir}")
            shutil.rmtree(self.results_dir, ignore_errors=True)
        else:
            info("  [7/8] No results directory to remove")

        # 8. Record completion in Buildkite metadata.
        info("  [8/8] Recording cleanup_completed in Buildkite metadata")
        set_buildkite_meta("cleanup_completed", "true")

        info("Cleanup complete")


_cleanup = _Cleanup()


def _on_signal(signum, _frame):
    # type: (int, object) -> None
    """Signal handler for SIGTERM and SIGINT.

    Runs cleanup and exits with 128+signum (POSIX convention).
    """
    _cleanup.run()
    sys.exit(128 + signum)


# ==========================================================================
# Main -- 10-phase orchestration
#
# Each phase is a logical unit that can succeed or fail independently.
# The order matters: GPU cleanup must happen before GPU reset, which must
# happen before container execution, etc.
# ==========================================================================


def main():
    # type: () -> None
    """Entry point: orchestrate the full test run in 10 phases.

    Phase  1: Environment     -- Detect K8s, log pod context.
    Phase  2: Infra health    -- Docker daemon, network, DNS, memory, disk I/O.
    Phase  3: GPU pre-flight  -- Kill zombies, wait for clean state.
    Phase  4: GPU health      -- Validate temperature, ECC, enumeration.
    Phase  5: Docker disk     -- Prune if partition is full.
    Phase  6: GPU reset       -- Write "reset" to state file, wait for "clean".
    Phase  7: Image pull      -- Pull the CI Docker image (with retry).
    Phase  8: Commands        -- Source, re-quote, apply ROCm overrides.
    Phase  9: Execute         -- Run container (single-node or multi-node).
    Phase 10: Exit            -- Normalize pytest exit code and exit.
    """
    # -- Register cleanup for any exit path --
    atexit.register(_cleanup.run)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # -- Phase 1: Environment + config --
    section("Environment")
    log_k8s_context()
    log_effective_config()

    # -- Phase 2: Infrastructure health --
    with timed("Infrastructure health checks"):
        check_docker_health()
        check_infra_health()

    # -- Phase 3: GPU pre-flight --
    with timed("GPU pre-flight"):
        kill_gpu_zombies()
        wait_for_clean_gpus()

    section("ROCm info")
    sh("rocminfo")

    # -- Phase 4: GPU health --
    rocm_smi_validate_health()

    # -- Phase 5: Docker housekeeping --
    with timed("Docker housekeeping"):
        cleanup_stale_containers()
        cleanup_docker_disk()

    # -- Phase 6: GPU reset --
    with timed("GPU reset"):
        reset_gpus()

    # -- Phase 7: Image pull --
    section("Pulling container")
    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        error("BUILDKITE_COMMIT is not set")
        sys.exit(1)

    image = f"rocm/vllm-ci:{commit}"
    container = f"rocm_{commit}_{os.urandom(5).hex()}"
    info(f"Image: {image}")
    info(f"Container name: {container}")

    _cleanup.image = image
    _cleanup.container = container
    _cleanup.commit = commit

    with timed(f"docker pull {image}"):
        docker_pull_with_retry(image)

    # -- Phase 8: Commands --
    section("Preparing test commands")
    commands = get_commands(sys.argv)
    info(f"Raw commands: {commands}")

    commands = re_quote_pytest_markers(commands)
    info(f"After re-quoting: {commands}")

    commands = apply_rocm_overrides(commands)
    info(f"Final commands: {commands}")

    # -- Phase 8b: Validate runtime environment --
    try:
        render_gid = str(grp.getgrnam("render").gr_gid)
        info(f"Render group GID: {render_gid}")
    except KeyError:
        error("'render' group not found -- required for GPU access")
        sys.exit(1)

    rdma = Path("/dev/infiniband").is_dir()
    info(
        "RDMA devices detected, enabling passthrough"
        if rdma
        else "No RDMA devices found, RDMA tests will be skipped"
    )

    render_devices = os.environ.get("BUILDKITE_AGENT_META_DATA_RENDER_DEVICES", "")
    if render_devices:
        info(f"Render devices: {render_devices}")
    else:
        warn("BUILDKITE_AGENT_META_DATA_RENDER_DEVICES is empty")

    # Set up all persistent caches (HF, ModelScope, test data, pip, ccache, etc.).
    setup_caches()

    # -- Phase 9: Execute --
    results_dir = Path(tempfile.mkdtemp(prefix=f"vllm-ci-{container[:20]}-"))
    _cleanup.results_dir = results_dir
    info(f"Results directory: {results_dir}")

    multi = is_multi_node(commands)
    info(f"Execution mode: {'multi-node' if multi else 'single-node'}")

    if multi:
        _cleanup.is_multi_node = True
        with timed("Multi-node test execution"):
            exit_code = run_multi_node(commands, image, results_dir)
    else:
        section("Single-node job")
        with timed("Single-node test execution"):
            exit_code = run_container(
                image=image,
                name=container,
                commands=commands,
                render_gid=render_gid,
                results_dir=results_dir,
                render_devices=render_devices,
                rdma=rdma,
            )

    # -- Phase 10: Exit --
    final_code = normalize_pytest_exit(exit_code)
    info(f"Final exit code: {final_code} (raw: {exit_code})")
    sys.exit(final_code)


if __name__ == "__main__":
    main()
