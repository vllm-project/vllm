#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AMD GPU CI test runner for vLLM -- K8s-hardened.

Runs pytest inside a Docker container on AMD GPU hardware, with proper
handling of GPU lifecycle, exit code propagation, and cleanup.

Designed to run inside Kubernetes pods with Docker socket access, handling
the unique challenges of containerized GPU CI:

  - Cross-PID-namespace GPU zombie detection (via ``amd-smi``/``rocm-smi``)
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
    GPU VRAM monitoring - Pre/post-test VRAM snapshots via ``amd-smi``/``rocm-smi``
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

This script uses **JUnit XML** as the source of truth. Pytest writes
the XML file during ``pytest_sessionfinish`` -- before Python's atexit
handlers execute. The XML is written to a bind-mounted volume, so it
survives on the host after the container exits. After ``docker wait``
returns, we parse the XML: if it reports failures but the exit code is 0,
we override the exit code to 1.


Usage
-----
Preferred (quoting preserved):

    export VLLM_TEST_COMMANDS='pytest -v -s tests/ -m "not slow"'
    python3 run-amd-test.py

Legacy (backward-compatible, inner double-quotes may be stripped):

    python3 run-amd-test.py "pytest -v -s tests/"

The bash shim (run-amd-test.sh) does "exec python3 run-amd-test.py $@"
so existing Buildkite pipeline YAML needs no changes.
"""

from __future__ import annotations

import atexit
import fcntl
import glob as _glob
import grp
import json
import os
import re  # noqa: UP027
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

# ==========================================================================
# Constants
#
# All tunables are grouped here so operators can adjust them without reading
# the rest of the file. Environment variable overrides are noted inline.
# ==========================================================================

# --------------------------------------------------------------------------
# Master feature switches
#
# Each major subsystem can be disabled with a single env var. Use these
# as kill switches when something breaks in production. All default to
# "1" (enabled). Set to "0" to disable.
#
# VLLM_ROCM_CI_GPU_PREFLIGHT    GPU zombie cleanup, health validation, VRAM checks.
# VLLM_ROCM_CI_INFRA_CHECKS     DNS, registry, memory, disk I/O, pod uptime checks.
# VLLM_ROCM_CI_CACHE_ENABLED    All persistent cache mounts and eviction.
# VLLM_ROCM_CI_CACHE_EVICTION   L1 and L2 cache eviction (mounts still work).
# VLLM_ROCM_CI_DOCKER_EVICTION  Docker image LFU/LRU eviction.
# VLLM_ROCM_CI_DIAGNOSTICS      OOM/signal/PID-limit detection and annotations.
# VLLM_ROCM_CI_JUNIT_OVERRIDE   JUnit XML exit-code override (the core fix).
#                                Disabling this means atexit hooks can hide failures.
# --------------------------------------------------------------------------

ENABLE_GPU_PREFLIGHT = os.environ.get("VLLM_ROCM_CI_GPU_PREFLIGHT", "1") == "1"
ENABLE_INFRA_CHECKS = os.environ.get("VLLM_ROCM_CI_INFRA_CHECKS", "1") == "1"
ENABLE_CACHE = os.environ.get("VLLM_ROCM_CI_CACHE_ENABLED", "1") == "1"
ENABLE_CACHE_EVICTION = os.environ.get("VLLM_ROCM_CI_CACHE_EVICTION", "1") == "1"
ENABLE_DOCKER_EVICTION = os.environ.get("VLLM_ROCM_CI_DOCKER_EVICTION", "1") == "1"
ENABLE_DIAGNOSTICS = os.environ.get("VLLM_ROCM_CI_DIAGNOSTICS", "1") == "1"
ENABLE_JUNIT_OVERRIDE = os.environ.get("VLLM_ROCM_CI_JUNIT_OVERRIDE", "1") == "1"

# Docker image resolution:
#   1. Use DOCKER_IMAGE_NAME when the pipeline sets it explicitly.
#   2. Otherwise fall back to the shared commit-tagged AMD image.
DEFAULT_IMAGE_REPO = "rocm/vllm-ci"

# --------------------------------------------------------------------------
# GPU architecture registry
#
# Every ROCm GPU architecture that CI is allowed to run on must be listed
# here. This is a safety gate: if a pipeline step targets an architecture
# that is not in this registry, the script exits immediately with a clear
# error pointing to this line.
#
# Why this exists:
#   - Prevents silent misconfiguration. A typo in DOCKER_IMAGE_NAME
#     (e.g., "gfx94" instead of "gfx942") would pull a nonexistent image
#     and fail late with a confusing Docker error. The registry catches it
#     upfront.
#   - Prevents running on unsupported hardware. If a new GPU architecture
#     is added to the fleet without updating the CI runner, tests may
#     produce wrong results (wrong compiled kernels, wrong NCCL config).
#     The registry forces a conscious decision to support a new arch.
#   - Documents exactly which architectures are tested. Anyone reading
#     this file knows the full set.
#
# To add a new architecture:
#   1. Add it to SUPPORTED_ROCM_ARCHS below.
#   2. Verify the architecture string matches what PYTORCH_ROCM_ARCH and
#      rocm-smi report (e.g., "gfx942", not "mi300x").
# --------------------------------------------------------------------------
SUPPORTED_ROCM_ARCHS = frozenset(
    {
        "gfx90a",  # MI210, MI250, MI250X
        "gfx942",  # MI300X, MI300A, MI325X
        "gfx950",  # MI350X, MI355X
    }
)


def _validate_image_arch(image_name):
    # type: (str) -> None
    """Validate that an arch-suffixed image tag targets a supported architecture.

    If DOCKER_IMAGE_NAME is set to a per-arch tag, it is expected to end with
    "-{arch}" (for example, "rocm/vllm-ci:abc123-gfx942"). This function
    extracts that suffix and checks it against SUPPORTED_ROCM_ARCHS.

    Shared multi-arch commit tags (for example, "rocm/vllm-ci:abc123") have no
    arch suffix, so there is nothing to validate and this function returns.

    Args:
        image_name: Full Docker image name from DOCKER_IMAGE_NAME.

    Raises:
        SystemExit: If the arch suffix is present but not supported.
    """
    if ":" not in image_name:
        return

    tag = image_name.rsplit(":", 1)[1]
    if "-" not in tag:
        return

    arch = tag.rsplit("-", 1)[1]
    if arch not in SUPPORTED_ROCM_ARCHS:
        error(
            f"GPU architecture '{arch}' (from image tag '{image_name}') "
            f"is not in the supported architecture registry.\n"
            f"\n"
            f"  Registered architectures: "
            f"{', '.join(sorted(SUPPORTED_ROCM_ARCHS))}\n"
            f"\n"
            f"  This check exists at SUPPORTED_ROCM_ARCHS in:\n"
            f"    {__file__}:{_SUPPORTED_ROCM_ARCHS_LINE}\n"
            f"\n"
            f"  If '{arch}' is a valid new architecture, add it to\n"
            f"  SUPPORTED_ROCM_ARCHS. If this is a typo, fix\n"
            f"  DOCKER_IMAGE_NAME in the pipeline YAML."
        )
        sys.exit(1)

    info(f"Architecture '{arch}' is registered and supported")


# Line number of SUPPORTED_ROCM_ARCHS for error messages. This avoids
# hardcoding a line number that goes stale on every edit.
_SUPPORTED_ROCM_ARCHS_LINE = "?"  # type: str
try:
    with open(__file__) as _f:
        for _i, _line in enumerate(_f, 1):
            if _line.strip().startswith("SUPPORTED_ROCM_ARCHS"):
                _SUPPORTED_ROCM_ARCHS_LINE = str(_i)
                break
except OSError:
    pass

# --------------------------------------------------------------------------
# Hard reset switches
#
# These are one-shot destructive operations. Set the env var to "1" on a
# single build to wipe the corresponding state, then remove the env var.
# The test still runs after the reset -- this is a pre-test cleanup, not
# an abort.
#
# VLLM_ROCM_CI_RESET_CACHE_COUNTS   Wipe .access_counts (frequency log).
#                                    Use when counts are corrupted or you
#                                    want a fresh frequency baseline.
#
# VLLM_ROCM_CI_RESET_CACHE_L1       Wipe ALL L1 (local) cache data.
#                                    Next test starts cold. Use when cache
#                                    is corrupted or causing test failures.
#
# VLLM_ROCM_CI_RESET_CACHE_L2       Wipe ALL L2 (NFS/PVC) cache data.
#                                    Affects all pods on all nodes. Use
#                                    with caution. Every pod starts cold
#                                    until caches are rebuilt.
#
# VLLM_ROCM_CI_RESET_DOCKER         Run docker system prune --all --force.
#                                    Wipes every image, container, volume.
#                                    Next build does a full cold pull.
#
# VLLM_ROCM_CI_RESET_OVERLAY        Remove overlay workdirs and merged
#                                    mounts. Fixes stale overlay state
#                                    after unclean shutdown.
# --------------------------------------------------------------------------

RESET_CACHE_COUNTS = os.environ.get("VLLM_ROCM_CI_RESET_CACHE_COUNTS", "0") == "1"
RESET_CACHE_L1 = os.environ.get("VLLM_ROCM_CI_RESET_CACHE_L1", "0") == "1"
RESET_CACHE_L2 = os.environ.get("VLLM_ROCM_CI_RESET_CACHE_L2", "0") == "1"
RESET_DOCKER = os.environ.get("VLLM_ROCM_CI_RESET_DOCKER", "0") == "1"
RESET_OVERLAY = os.environ.get("VLLM_ROCM_CI_RESET_OVERLAY", "0") == "1"

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
DISK_USAGE_THRESHOLD_PCT = int(os.environ.get("VLLM_DISK_THRESHOLD_PCT", "0"))
DISK_USAGE_TARGET_PCT = int(os.environ.get("VLLM_DISK_TARGET_PCT", "0"))

# Absolute thresholds in GB. 0 = disabled (percentage-only or no eviction).
DISK_USAGE_THRESHOLD_GB = int(os.environ.get("VLLM_DISK_THRESHOLD_GB", "512"))
DISK_USAGE_TARGET_GB = int(os.environ.get("VLLM_DISK_TARGET_GB", "256"))

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
CONTAINER_REPO_ROOT = "/vllm-workspace"

# ---------------------------------------------------------------------------
# Two-tier cache configuration (L1 ephemeral + L2 persistent)
#
# In K8s, storage comes in two flavors with opposite tradeoffs:
#
#   Ephemeral (emptyDir, local NVMe)   Fast I/O, but wiped on pod death.
#   Persistent (hostPath, PVC/NFS)     Survives pod restarts, but slower I/O.
#
# Using only ephemeral means every pod starts cold (downloads 12GB+ of
# model weights). Using only persistent means slow reads on every cache
# hit if the backend is NFS.
#
# The solution is a two-tier cache using OverlayFS:
#
#   L1: CACHE_ROOT (ephemeral, fast) -- the "upper" overlay layer.
#     All writes land here. Reads that hit L1 are served from fast
#     local storage. This is the read-write layer.
#
#   L2: CACHE_BACKING_ROOT (persistent, slow) -- the "lower" overlay layer.
#     Read-only from the overlay's perspective. When a file is not in L1,
#     the kernel transparently reads it from L2 (NFS). No application
#     changes needed -- tools like huggingface_hub see a single merged
#     directory and don't know about the tiers.
#
#   Merged view: mounted at CACHE_OVERLAY_ROOT, bind-mounted into the
#     container. The test sees one directory with the union of L1 and L2.
#
# OverlayFS gives us:
#   - L1 cache hits: served from NVMe (fast)
#   - L2 cache hits: served from NFS (slow but no internet download)
#   - Cache misses: downloaded from internet, written to L1
#   - After test: rsync L1 delta back to L2 so next pod has it
#
# When OverlayFS is not available (no kernel support, no privileges),
# we fall back to seed-based sync: copy the most recently used files
# from L2 to L1 at startup, and rsync new files back at cleanup.
#
# Access frequency tracking:
#   We maintain a lightweight access log at CACHE_ROOT/.access_counts
#   that records how many times each cache path has been requested
#   across jobs. This is used to prioritize seeding: high-frequency
#   files are seeded first when OverlayFS is not available. The log
#   is persisted to backing on cleanup.
#
# If CACHE_BACKING_ROOT is not set, the system degrades to single-tier:
# CACHE_ROOT is both L1 and L2. This is the right default for hostPath
# mounts (persistent + fast local disk).
#
# Tier behavior matrix:
#
#   CACHE_ROOT        CACHE_BACKING_ROOT  Behavior
#   ----------------------------------------------------------------
#   ~/vllm-ci-cache   (not set)           Single-tier. hostPath style.
#   /scratch/cache    /mnt/nfs/cache      Two-tier overlay (if supported)
#                                         or seed-based fallback.
#   /tmp/cache        (not set)           Single-tier ephemeral. Cold
#                                         start every pod. Works but slow.
#
# Adding a new cache:
#   1. Add an entry to CACHES below.
#   2. The host directory is created automatically under CACHE_ROOT.
#   3. The volume mount and env var are injected into docker run.
#   4. If backing is configured, overlay or seed happens automatically.
#
# Kubernetes-managed model/test caches:
#   HF_HOME, HF_DATASETS_CACHE, MODELSCOPE_CACHE, VLLM_TEST_CACHE,
#   VLLM_CACHE_ROOT, and VLLM_MEDIA_CACHE are expected to be set on the outer
#   Buildkite pod. This runner bind-mounts that existing host cache tree into
#   the inner Docker container, but does not budget, seed, or evict it.
#
# Override env vars for runner-managed caches:
#   VLLM_CI_CACHE_ROOT         Hot tier path (default: ~/vllm-ci-cache)
#   VLLM_CI_CACHE_BACKING_ROOT Warm tier path (default: unset = single-tier)
#   VLLM_CACHE_MAX_<ENV_VAR>   Override one cache's L1 budget in GB.
#   VLLM_CACHE_L2_MAX_DAYS     Default L2 stale-file eviction age in days.
#   VLLM_CACHE_L1_FS_MIN_HEADROOM_GB
#                              Minimum free space to leave on the shared L1
#                              filesystem after seeding or sync work. This is
#                              a filesystem-wide brake, not a per-cache limit.
#   VLLM_CACHE_L1_FS_MAX_UTIL_PCT
#                              Maximum allowed utilization of the shared L1
#                              filesystem for cache growth. Seeding and
#                              watchdog sync stop early if this cap is hit.
#
# Safety:
#   Runner-managed cache roots in CACHES must not overlap one another.
#   Overlapping host cache roots make per-cache budgets double-count disk usage,
#   which can defeat eviction guardrails and trigger node or pod disk pressure.
# ---------------------------------------------------------------------------

# Hot tier: fast local storage. Container mounts point here.
DEFAULT_CACHE_ROOT = Path.home() / "vllm-ci-cache"
TMP_CACHE_ROOT = Path(tempfile.gettempdir()) / f"vllm-ci-cache-{os.getuid()}"
CACHE_ROOT = Path(
    os.environ.get(
        "VLLM_CI_CACHE_ROOT",
        str(DEFAULT_CACHE_ROOT),
    )
)

# L2 (warm tier): persistent shared storage. Set to a PVC or NFS mount.
# None = single-tier mode (no backing store, CACHE_ROOT is everything).
_backing_env = os.environ.get("VLLM_CI_CACHE_BACKING_ROOT", "")
CACHE_BACKING_ROOT = Path(_backing_env) if _backing_env else None

# Overlay merged view: when OverlayFS is used, this is the mount point
# that the container sees. It merges L1 (upper) and L2 (lower).
# OverlayFS also needs a workdir (same filesystem as upper).
CACHE_OVERLAY_ROOT = CACHE_ROOT.parent / "vllm-ci-cache-merged"
CACHE_OVERLAY_WORK = CACHE_ROOT.parent / "vllm-ci-cache-work"

# Access frequency log file. Tracks how many times each relative path
# has been accessed across jobs. Used to prioritize seeding when
# OverlayFS is not available.
ACCESS_LOG_FILE = CACHE_ROOT / ".access_counts"


def _set_cache_root(path):
    # type: (Path) -> None
    """Rebind all derived cache paths when the hot tier root changes."""
    global CACHE_ROOT, CACHE_OVERLAY_ROOT, CACHE_OVERLAY_WORK, ACCESS_LOG_FILE
    CACHE_ROOT = path
    CACHE_OVERLAY_ROOT = CACHE_ROOT.parent / "vllm-ci-cache-merged"
    CACHE_OVERLAY_WORK = CACHE_ROOT.parent / "vllm-ci-cache-work"
    ACCESS_LOG_FILE = CACHE_ROOT / ".access_counts"


def _probe_writable_dir(path):
    # type: (Path) -> Exception | None
    """Return None when a directory is creatable and writable, else the error."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=str(path),
            prefix=".cache-write-probe-",
        ):
            pass
        return None
    except Exception as exc:
        return exc


def normalize_cache_root():
    # type: () -> None
    """Prefer the configured cache root, but fall back to a writable local path."""
    global ENABLE_CACHE

    if not ENABLE_CACHE:
        return

    configured_root = CACHE_ROOT
    configured_err = _probe_writable_dir(configured_root)
    if configured_err is None:
        return

    warn(f"Configured cache root {configured_root} is not writable: {configured_err}")

    fallback_roots = []
    for candidate in [DEFAULT_CACHE_ROOT, TMP_CACHE_ROOT]:
        if candidate not in fallback_roots and candidate != configured_root:
            fallback_roots.append(candidate)

    for fallback in fallback_roots:
        fallback_err = _probe_writable_dir(fallback)
        if fallback_err is None:
            warn(f"Falling back to cache root: {fallback}")
            _set_cache_root(fallback)
            os.environ["VLLM_CI_CACHE_ROOT"] = str(fallback)
            return

    warn("No writable cache root found; disabling persistent caches for this run")
    ENABLE_CACHE = False


# Cache registry: each entry defines one persistent cache.
#
#   host_subdir:    directory name under CACHE_ROOT on the host
#   container_path: absolute path inside the container
#   env_var:        environment variable set inside the container
#   description:    human-readable description for logging
#   max_gb:         L1 (ephemeral/local) size budget in GB. This controls:
#                     - LRU eviction: when L1 exceeds max_gb, oldest-accessed
#                       files are evicted from L1 until under the limit.
#                     - Seed budget: at pod startup, at most max_gb is copied
#                       from L2 (NFS) into L1.
#                   0 = unlimited (bounded only by available disk space).
#
#   l2_max_gb:      L2 (NFS/PVC) size budget in GB. 0 = unlimited.
#                   Even though persistent storage is large, stale data
#                   accumulates (old model versions, abandoned experiments).
#                   When exceeded, L2 is trimmed by time-based eviction.
#
#   l2_max_days:    L2 time-based eviction threshold in days. Files in L2
#                   not accessed (atime) for longer than this are evicted,
#                   regardless of L2 size. 0 = disabled (size-only eviction).
#                   Default: 14 days.
#
# Override any cache's max_gb at runtime via env var:
#   VLLM_CACHE_MAX_<ENV_VAR>=<gb>
#   Example: VLLM_CACHE_MAX_PIP_CACHE_DIR=5
#
# Kubernetes-managed cache tree. These long-lived model/test caches continue to
# live under the outer Buildkite pod's HF tree; the runner only bind-mounts the
# tree into the inner Docker container and injects the container-side env vars.
K8S_CACHE_HOST_ROOT = Path(os.environ.get("HF_HOME", str(Path.home() / "huggingface")))
K8S_CACHE_CONTAINER_ROOT = "/root/.cache/huggingface"
K8S_MANAGED_CACHE_ENVS = (
    ("HF_HOME", K8S_CACHE_CONTAINER_ROOT),
    ("HF_DATASETS_CACHE", f"{K8S_CACHE_CONTAINER_ROOT}/datasets"),
    ("MODELSCOPE_CACHE", f"{K8S_CACHE_CONTAINER_ROOT}/modelscope"),
    ("VLLM_TEST_CACHE", f"{K8S_CACHE_CONTAINER_ROOT}/vllm-test-cache"),
    ("VLLM_CACHE_ROOT", f"{K8S_CACHE_CONTAINER_ROOT}/vllm-cache"),
    ("VLLM_MEDIA_CACHE", f"{K8S_CACHE_CONTAINER_ROOT}/vllm-cache/media_cache"),
)

# Default L2 time-based eviction: files not accessed in this many days
# are evicted from NFS/PVC. Override per-cache or globally via env var.
L2_DEFAULT_MAX_DAYS = int(os.environ.get("VLLM_CACHE_L2_MAX_DAYS", "21"))

# Filesystem-wide L1 safety rails. Per-cache max_gb limits are necessary but
# not sufficient because all caches share the same underlying host filesystem.
# These caps keep seed/sync work from consuming the last free space on NVMe and
# reduce the risk of kubelet disk-pressure evictions.
#   VLLM_CACHE_L1_FS_MIN_HEADROOM_GB keeps a minimum amount of free space.
#   VLLM_CACHE_L1_FS_MAX_UTIL_PCT   caps overall filesystem utilization.
L1_FS_MIN_HEADROOM_GB = float(os.environ.get("VLLM_CACHE_L1_FS_MIN_HEADROOM_GB", "100"))
L1_FS_MAX_UTIL_PCT = float(os.environ.get("VLLM_CACHE_L1_FS_MAX_UTIL_PCT", "85"))

CACHES = [
    {
        # pip respects PIP_CACHE_DIR. Used by the 46 "pip install"
        # commands in test-amd.yaml.
        "host_subdir": "pip",
        "container_path": "/root/.cache/pip",
        "env_var": "PIP_CACHE_DIR",
        "max_gb": 32,
        "l2_max_gb": 64,
        "l2_max_days": 7,  # old wheel versions pile up fast
        "description": "pip download cache",
    },
    {
        # uv respects UV_CACHE_DIR. Used by the 25 "uv pip install"
        # commands in test-amd.yaml.
        "host_subdir": "uv",
        "container_path": "/root/.cache/uv",
        "env_var": "UV_CACHE_DIR",
        "max_gb": 64,
        "l2_max_gb": 256,
        "l2_max_days": 7,
        "description": "uv package manager cache",
    },
    {
        # ROCm CI runtime images compile with ccache when source builds happen
        # inside the test container (for example editable installs or plugin
        # builds). Keep this on the host so repeated jobs reuse object files.
        "host_subdir": "ccache",
        "container_path": "/root/.cache/ccache",
        "env_var": "CCACHE_DIR",
        "max_gb": 80,
        "l2_max_gb": 256,
        "l2_max_days": 21,
        "description": "ccache compiler cache (ROCm source builds)",
    },
]  # type: list[dict]

# Pytest exit code 5 = "no tests were collected". We treat this as success
# because shard-based parallelism can legitimately produce empty shards.
PYTEST_NO_TESTS_COLLECTED = 5

# Safety threshold: never SIGKILL PIDs below this (avoids killing init, etc.).
MIN_SYSTEM_PID = 1000

# Remove stopped rocm_* containers older than this (hours).
STALE_CONTAINER_AGE_H = 4

# If ``docker info`` does not respond within this many seconds, abort.
DOCKER_HEALTH_TIMEOUT_S = 120

# Timeout for rocm-smi commands. The AMD driver can hang indefinitely
# if the GPU is in a bad state (post-reset, memory pressure). Without
# a timeout, the entire CI script hangs silently with no log output.
ROCM_SMI_TIMEOUT_S = 60

# Number of times to retry ``docker pull`` on failure (network flakes).
DOCKER_PULL_RETRIES = 3
DOCKER_PULL_RETRY_DELAY_S = 30

# ---------------------------------------------------------------------------
# Local image cache (NVMe).
#
# Set by the Buildkite agent hooks when running on K8s nodes with NVMe.
# The base-tar-updater DaemonSet keeps ci_base.tar and base.tar up to date
# in this directory. The hooks load them into DinD before this script runs.
#
# When set, Phase 7 uses a tiered strategy:
#   Tier 0: Load per-commit tar from NVMe (zero network, ~10s)
#   Tier 1: Assemble from ci_base + wheel artifact (~25s, ~50MB download)
#   Tier 2: docker pull with pre-loaded layers (~40-60s, ~1-3GB delta)
#   Tier 3: Cold docker pull (~350s, full image)
#
# When empty, Phase 7 uses the standard docker pull with retry.
LOCAL_IMAGE_CACHE = os.environ.get("VLLM_LOCAL_IMAGE_CACHE", "")

# CI base image for local assembly (Tier 1). The hooks pre-load this
# from NVMe tar into DinD. Must match what Dockerfile.rocm uses as
# the FROM image for the test stage.
CI_BASE_IMAGE = os.environ.get(
    "VLLM_CI_BASE_IMAGE",
    os.environ.get("CI_BASE_IMAGE", "rocm/vllm-dev:ci_base"),
)

# Maximum number of PIDs allowed inside the test container.
#
# Docker's ``--pids-limit`` caps the number of tasks in the container. On
# Linux, that budget covers both processes and threads, so thread-heavy
# runtimes such as Ray can hit it even when they do not spawn thousands of
# child processes. Keep a finite default to avoid fork-bombing the node, but
# leave more headroom than the previous 4096-task cap.
#
# Override env var:
#   VLLM_CI_DOCKER_PIDS_LIMIT   Positive integer, or ``-1`` for unlimited.
#
# Official docs:
#   https://docs.docker.com/reference/cli/docker/container/run/
CONTAINER_PIDS_LIMIT = os.environ.get(
    "VLLM_CI_DOCKER_PIDS_LIMIT",
    "16384",
).strip()

# Docker IPC/shm configuration for the single-node test container.
#
# Docker documents these knobs separately:
#   - ``--shm-size`` sets the size of the container's ``/dev/shm`` tmpfs.
#   - ``--ipc=host`` makes the container use the host system's IPC namespace.
#     In that mode, the host's shared memory namespace is used instead of a
#     private container ``/dev/shm`` mount.
#   - Docker's daemon default IPC mode may be ``private`` or ``shareable``
#     depending on daemon version/configuration, so this runner sets an
#     explicit mode for deterministic CI behavior.
#
# Official docs:
#   https://docs.docker.com/engine/containers/run/
#   https://docs.docker.com/reference/cli/docker/container/run/
#
# For single-node jobs the test workload runs entirely inside ONE container,
# so the default should be ``private``: all worker processes inside that
# container already share the same IPC namespace, and ``--shm-size`` then
# governs the container's private ``/dev/shm`` as intended.
#
# Override env vars:
#   VLLM_CI_DOCKER_IPC_MODE   One of: private, shareable, host
#   VLLM_CI_DOCKER_SHM_SIZE   ``/dev/shm`` size when IPC mode is private/shareable
_VALID_CONTAINER_IPC_MODES = frozenset({"private", "shareable", "host"})
CONTAINER_IPC_MODE = (
    os.environ.get("VLLM_CI_DOCKER_IPC_MODE", "private").strip().lower()
)
CONTAINER_SHM_SIZE = os.environ.get("VLLM_CI_DOCKER_SHM_SIZE", "16gb").strip().lower()

# Buildkite Test Engine configuration for pytest-based AMD CI jobs.
#
# Buildkite's Python collector docs say that installing
# ``buildkite-test-collector`` is sufficient for pytest jobs, as long as the
# tests run with access to the Test Engine token and the CI metadata. Their CI
# environment docs further note that containerized jobs must explicitly pass
# the token plus the Buildkite build/job variables into the container.
#
# Official docs:
#   https://buildkite.com/docs/test-engine/python-collectors
#   https://buildkite.com/docs/test-engine/test-collection/ci-environments
#
# Runner policy:
#   - Enable Test Engine only on nightly builds (``NIGHTLY=1``).
#   - Require the Test Suite token (``BUILDKITE_ANALYTICS_TOKEN``).
#   - Forward the standard Buildkite build metadata when present.
_TEST_ENGINE_TOKEN_ENV_VAR = "BUILDKITE_ANALYTICS_TOKEN"
_TEST_ENGINE_METADATA_ENV_VARS = (
    "BUILDKITE_BUILD_ID",
    "BUILDKITE_JOB_ID",
    "BUILDKITE_BUILD_NUMBER",
    "BUILDKITE_BRANCH",
    "BUILDKITE_COMMIT",
    "BUILDKITE_MESSAGE",
    "BUILDKITE_BUILD_URL",
)
_TEST_ENGINE_BUILDKITE_ENV_VARS = (
    _TEST_ENGINE_TOKEN_ENV_VAR,
) + _TEST_ENGINE_METADATA_ENV_VARS

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

# --------------------------------------------------------------------------
# Health watchdog knobs
#
# The watchdog is a background thread that samples system health (memory,
# disk, VRAM, container status) and, when two-tier cache is enabled,
# incrementally rsyncs new model files from L1 (NVMe) to L2 (NFS) during
# the test. This means models are available to other nodes before the test
# finishes, and disk pressure is relieved mid-test rather than only at
# cleanup.
#
# VLLM_ROCM_CI_WATCHDOG_INTERVAL           Sampling interval (seconds).
#                                           How often the watchdog checks
#                                           memory, disk, VRAM, and container
#                                           status. Default: 120.
#
# VLLM_ROCM_CI_WATCHDOG_SYNC_INTERVAL      Normal cache sync interval (seconds).
#                                           How often the watchdog rsyncs new
#                                           files from L1 to L2 during the test.
#                                           Only active when CACHE_BACKING_ROOT
#                                           is set. Default: 300.
#
# VLLM_ROCM_CI_WATCHDOG_SYNC_PRESSURE_INTERVAL
#                                           Pressure cache sync interval (seconds).
#                                           Used when L1 usage exceeds the
#                                           pressure threshold. Default: 10.
#
# VLLM_ROCM_CI_WATCHDOG_CACHE_PRESSURE_PCT L1 usage threshold (% of max_gb)
#                                           that triggers pressure mode, causing
#                                           more frequent syncs. Default: 80.
#
# VLLM_CI_POD_MEMORY_WARN_PCT               Warn when current cgroup/container
#                                           memory usage crosses this percent
#                                           of the pod limit. This is a logging
#                                           threshold only; the outer timeout
#                                           still owns hard-stop behavior.
# --------------------------------------------------------------------------

WATCHDOG_INTERVAL_S = int(os.environ.get("VLLM_ROCM_CI_WATCHDOG_INTERVAL", "120"))
WATCHDOG_SYNC_INTERVAL_S = int(
    os.environ.get("VLLM_ROCM_CI_WATCHDOG_SYNC_INTERVAL", "300")
)
WATCHDOG_SYNC_PRESSURE_INTERVAL_S = int(
    os.environ.get("VLLM_ROCM_CI_WATCHDOG_SYNC_PRESSURE_INTERVAL", "10")
)
WATCHDOG_CACHE_PRESSURE_PCT = int(
    os.environ.get("VLLM_ROCM_CI_WATCHDOG_CACHE_PRESSURE_PCT", "80")
)
POD_MEMORY_WARN_PCT = int(os.environ.get("VLLM_CI_POD_MEMORY_WARN_PCT", "90"))

# Runtime cache state. If setup_caches() fails, container cache mounts must be
# disabled so tests still run without inheriting a partial cache layout.
CACHE_RUNTIME_FAILED = False
OVERLAY_ACTIVE_SUBDIRS = set()  # type: set[str]


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
    if timeout is not None:
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
    exc_occurred = False
    try:
        yield
    except BaseException:
        exc_occurred = True
        raise
    finally:
        elapsed = time.monotonic() - start
        status = "FAILED after" if exc_occurred else "completed in"
        info(f"{label} {status} {elapsed:.1f}s")


@contextmanager
def best_effort(label):
    # type: (str) -> ...
    """Run a block that must never crash the script.

    If the block raises, logs a WARNING with the label and exception,
    then continues. Use this instead of bare ``suppress(Exception)``
    so failures are visible in the build log.
    """
    try:
        yield
    except Exception as exc:
        warn(f"{label} failed: {exc}")


# ==========================================================================
# Configuration dump
# ==========================================================================


def _container_uses_private_shm(ipc_mode):
    # type: (str) -> bool
    """Return True when Docker gives the container its own ``/dev/shm``.

    Docker's ``private`` and ``shareable`` IPC modes both create a private IPC
    namespace for the container. In those modes, ``--shm-size`` controls the
    container's ``/dev/shm`` tmpfs. ``host`` is different: it joins the Docker
    host's IPC namespace, so host-side ``/dev/shm`` capacity governs.
    """
    return ipc_mode in {"private", "shareable"}


def _parse_container_pids_limit(limit):
    # type: (str) -> int | None
    """Parse the Docker PIDs limit env var.

    Docker accepts positive integers and ``-1`` (unlimited). ``0`` is not a
    useful runtime value for this runner and is treated as invalid.
    """
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return None
    if parsed == -1 or parsed > 0:
        return parsed
    return None


def _format_container_pids_limit(limit):
    # type: (str) -> str
    """Render the configured PID budget in operator-friendly form."""
    parsed = _parse_container_pids_limit(limit)
    if parsed == -1:
        return "unlimited (-1)"
    return limit if parsed is None else str(parsed)


def validate_container_pids_config():
    # type: () -> None
    """Validate the Docker PID budget configuration."""
    if _parse_container_pids_limit(CONTAINER_PIDS_LIMIT) is None:
        error(
            "Unsupported VLLM_CI_DOCKER_PIDS_LIMIT="
            f"'{CONTAINER_PIDS_LIMIT}'. Expected a positive integer or -1."
        )
        sys.exit(1)


def validate_container_ipc_config():
    # type: () -> None
    """Validate the single-node Docker IPC/shared-memory configuration."""
    if CONTAINER_IPC_MODE not in _VALID_CONTAINER_IPC_MODES:
        error(
            f"Unsupported VLLM_CI_DOCKER_IPC_MODE='{CONTAINER_IPC_MODE}'. "
            f"Expected one of: {', '.join(sorted(_VALID_CONTAINER_IPC_MODES))}"
        )
        sys.exit(1)

    if _container_uses_private_shm(CONTAINER_IPC_MODE) and not CONTAINER_SHM_SIZE:
        error(
            "VLLM_CI_DOCKER_SHM_SIZE must be non-empty when "
            f"VLLM_CI_DOCKER_IPC_MODE={CONTAINER_IPC_MODE}"
        )
        sys.exit(1)


def _is_nightly_build():
    # type: () -> bool
    """Return True when the current Buildkite run is a nightly build."""
    return os.environ.get("NIGHTLY", "0").strip() == "1"


def get_test_engine_config():
    # type: () -> dict[str, object]
    """Resolve the effective Buildkite Test Engine configuration.

    The pytest collector itself runs inside the test container. This helper
    decides whether the runner should forward the required Buildkite metadata
    into that container.
    """
    token = os.environ.get(_TEST_ENGINE_TOKEN_ENV_VAR, "").strip()
    missing_metadata = [
        name
        for name in _TEST_ENGINE_METADATA_ENV_VARS
        if not os.environ.get(name, "").strip()
    ]

    if not _is_nightly_build():
        return {
            "enabled": False,
            "reason": "disabled because NIGHTLY!=1",
            "missing_metadata": missing_metadata,
        }

    if not token:
        return {
            "enabled": False,
            "reason": f"disabled because {_TEST_ENGINE_TOKEN_ENV_VAR} is not set",
            "missing_metadata": missing_metadata,
        }

    return {
        "enabled": True,
        "reason": "enabled for nightly build",
        "missing_metadata": missing_metadata,
    }


def build_test_engine_docker_env_args():
    # type: () -> list[str]
    """Return ``docker run -e`` args for Buildkite Test Engine collection."""
    config = get_test_engine_config()
    if not config["enabled"]:
        return []

    docker_env = []  # type: list[str]
    for name in _TEST_ENGINE_BUILDKITE_ENV_VARS:
        if os.environ.get(name, "").strip():
            docker_env += ["-e", name]

    # Some collectors use ``CI`` as an additional hint that they are running
    # under automation. Preserve the agent value when present; otherwise set
    # a conservative default.
    docker_env += ["-e", f"CI={os.environ.get('CI', 'true')}"]
    return docker_env


def build_test_engine_shell_exports():
    # type: () -> str
    """Return shell exports for in-container pytest Test Engine collection."""
    config = get_test_engine_config()
    if not config["enabled"]:
        return ""

    assignments = []  # type: list[str]
    for name in _TEST_ENGINE_BUILDKITE_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            assignments.append(f"{name}={shlex.quote(value)}")

    assignments.append(f"CI={shlex.quote(os.environ.get('CI', 'true'))}")
    return "export {} && ".format(" ".join(assignments)) if assignments else ""


def build_git_root_shell_exports():
    # type: () -> str
    """Return a shell export for the repo root inside test containers."""
    return f'export GIT_ROOT="${{GIT_ROOT:-{CONTAINER_REPO_ROOT}}}" && '


def execute_hard_resets():
    # type: () -> None
    """Execute any hard-reset operations requested via env vars.

    Hard resets are destructive, one-shot operations that wipe state.
    They run at the very beginning of the script (Phase 1), before
    any other setup, so that subsequent phases see a clean slate.

    Each reset is logged prominently so it is obvious in the build
    log that a destructive operation occurred.
    """
    any_reset = any(
        [
            RESET_CACHE_COUNTS,
            RESET_CACHE_L1,
            RESET_CACHE_L2,
            RESET_DOCKER,
            RESET_OVERLAY,
        ]
    )
    if not any_reset:
        return

    section("HARD RESET requested")

    if RESET_CACHE_COUNTS:
        warn("Resetting cache access counts (VLLM_ROCM_CI_RESET_CACHE_COUNTS=1)")
        for log_path in [ACCESS_LOG_FILE]:
            if log_path.is_file():
                info(f"  Deleting {log_path}")
                os.unlink(str(log_path))
        if CACHE_BACKING_ROOT is not None:
            backing_log = CACHE_BACKING_ROOT / ".access_counts"
            if backing_log.is_file():
                info(f"  Deleting {backing_log}")
                os.unlink(str(backing_log))
        info("  Access counts reset to zero")

    if RESET_CACHE_L1:
        warn("Wiping ALL L1 cache data (VLLM_ROCM_CI_RESET_CACHE_L1=1)")
        if CACHE_ROOT.exists():
            info(f"  rm -rf {CACHE_ROOT}")
            shutil.rmtree(str(CACHE_ROOT), ignore_errors=True)
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        info("  L1 cache wiped -- next test starts cold")

    if RESET_CACHE_L2:
        warn("Wiping ALL L2 cache data (VLLM_ROCM_CI_RESET_CACHE_L2=1)")
        if CACHE_BACKING_ROOT is not None and CACHE_BACKING_ROOT.exists():
            if os.access(str(CACHE_BACKING_ROOT), os.W_OK):
                info(f"  rm -rf {CACHE_BACKING_ROOT}")
                shutil.rmtree(str(CACHE_BACKING_ROOT), ignore_errors=True)
                CACHE_BACKING_ROOT.mkdir(parents=True, exist_ok=True)
                info("  L2 cache wiped -- all pods start cold")
            else:
                error(f"  Cannot wipe L2: {CACHE_BACKING_ROOT} is read-only")
        else:
            info("  L2 not configured -- nothing to wipe")

    if RESET_OVERLAY:
        warn("Resetting overlay state (VLLM_ROCM_CI_RESET_OVERLAY=1)")
        for d in [CACHE_OVERLAY_ROOT, CACHE_OVERLAY_WORK]:
            if d.exists():
                # Unmount any active overlays first.
                still_mounted = False
                for sub in d.iterdir():
                    if sub.is_mount():
                        sh(f"umount '{sub}' 2>/dev/null || true")
                        if sub.is_mount():
                            # Lazy unmount as fallback for busy mounts.
                            sh(f"umount -l '{sub}' 2>/dev/null || true")
                            if sub.is_mount():
                                warn(
                                    f"  {sub} is still mounted after "
                                    f"lazy unmount -- skipping rmtree "
                                    f"to avoid deleting backing data"
                                )
                                still_mounted = True
                if still_mounted:
                    continue
                info(f"  rm -rf {d}")
                shutil.rmtree(str(d), ignore_errors=True)
        info("  Overlay state reset")

    if RESET_DOCKER:
        warn("Wiping ALL Docker data (VLLM_ROCM_CI_RESET_DOCKER=1)")
        sh("docker system prune --all --force --volumes", capture=True)
        info("  Docker fully pruned -- next pull is cold")


def log_effective_config():
    # type: () -> None
    """Log all effective configuration values at startup.

    This is the single most useful thing for post-mortem debugging:
    when a build fails, the first question is always "what were the
    settings?" Logging them upfront means the answer is always in
    the build log without needing to SSH into the node.
    """
    section("Effective configuration")

    # Feature switches first -- most important for debugging.
    def _on_off(val):
        # type: (bool) -> str
        return "ON" if val else "OFF"

    info("  Feature switches:")
    info(f"    GPU_PREFLIGHT:           {_on_off(ENABLE_GPU_PREFLIGHT)}")
    info(f"    INFRA_CHECKS:            {_on_off(ENABLE_INFRA_CHECKS)}")
    info(f"    CACHE:                   {_on_off(ENABLE_CACHE)}")
    info(f"    CACHE_EVICTION:          {_on_off(ENABLE_CACHE_EVICTION)}")
    info(f"    DOCKER_EVICTION:         {_on_off(ENABLE_DOCKER_EVICTION)}")
    info(f"    DIAGNOSTICS:             {_on_off(ENABLE_DIAGNOSTICS)}")
    info(f"    JUNIT_OVERRIDE:          {_on_off(ENABLE_JUNIT_OVERRIDE)}")
    docker_image_override = os.environ.get("DOCKER_IMAGE_NAME", "").strip() or "(unset)"
    info(f"    DOCKER_IMAGE_NAME:       {docker_image_override}")
    test_engine = get_test_engine_config()
    info(f"    TEST_ENGINE:             {_on_off(test_engine['enabled'])}")

    any_off = not all(
        [
            ENABLE_GPU_PREFLIGHT,
            ENABLE_INFRA_CHECKS,
            ENABLE_CACHE,
            ENABLE_CACHE_EVICTION,
            ENABLE_DOCKER_EVICTION,
            ENABLE_DIAGNOSTICS,
            ENABLE_JUNIT_OVERRIDE,
        ]
    )
    if any_off:
        warn(
            "One or more features are DISABLED via env vars. "
            "This may hide failures or skip cleanup."
        )
    info(f"  Test Engine:               {test_engine['reason']}")
    if test_engine["missing_metadata"]:
        warn(
            "Buildkite Test Engine is missing Buildkite metadata env vars: "
            f"{', '.join(test_engine['missing_metadata'])}. "
            "Uploads can still work, but branch/commit/build links may be incomplete."
        )

    # Hard resets.
    resets_active = [
        name
        for name, val in [
            ("CACHE_COUNTS", RESET_CACHE_COUNTS),
            ("CACHE_L1", RESET_CACHE_L1),
            ("CACHE_L2", RESET_CACHE_L2),
            ("DOCKER", RESET_DOCKER),
            ("OVERLAY", RESET_OVERLAY),
        ]
        if val
    ]
    if resets_active:
        warn(f"  Hard resets ACTIVE: {', '.join(resets_active)}")
    else:
        info("  Hard resets: none")

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
    info(
        "  CONTAINER_PIDS_LIMIT:      "
        f"{_format_container_pids_limit(CONTAINER_PIDS_LIMIT)}"
    )
    info(f"  CONTAINER_REPO_ROOT:       {CONTAINER_REPO_ROOT}")
    info(f"  CONTAINER_IPC_MODE:        {CONTAINER_IPC_MODE}")
    if _container_uses_private_shm(CONTAINER_IPC_MODE):
        info(f"  CONTAINER_SHM_SIZE:        {CONTAINER_SHM_SIZE}")
    else:
        info(
            "  CONTAINER_SHM_SIZE:        "
            f"{CONTAINER_SHM_SIZE} (ignored with --ipc={CONTAINER_IPC_MODE})"
        )
        warn(
            "Single-node Docker jobs are using --ipc=host, so /dev/shm comes "
            "from the Docker host IPC namespace and is not sized by "
            "VLLM_CI_DOCKER_SHM_SIZE."
        )
    info(f"  DOCKER_PULL_RETRIES:       {DOCKER_PULL_RETRIES}")
    info(f"  DOCKER_HEALTH_TIMEOUT_S:   {DOCKER_HEALTH_TIMEOUT_S}")
    info(f"  DISK_EVICTION_POLICY:      {DISK_EVICTION_POLICY}")
    info(f"  STALE_CONTAINER_AGE_H:     {STALE_CONTAINER_AGE_H}h")
    info(f"  MIN_SYSTEM_PID:            {MIN_SYSTEM_PID}")
    info(f"  CACHE_ROOT:                {CACHE_ROOT}")
    backing = CACHE_BACKING_ROOT or "(not set -- single-tier)"
    info(f"  CACHE_BACKING_ROOT:        {backing}")
    info(
        f"  Caches registered:         {len(CACHES)} "
        f"({', '.join(c['env_var'] for c in CACHES)})"
    )
    local_cache = LOCAL_IMAGE_CACHE or "(not set -- no tiered image acquisition)"
    info(f"  LOCAL_IMAGE_CACHE:         {local_cache}")
    info(f"  CI_BASE_IMAGE:             {CI_BASE_IMAGE}")
    info(f"  WATCHDOG_INTERVAL_S:       {WATCHDOG_INTERVAL_S}s")
    info(f"  WATCHDOG_SYNC_INTERVAL_S:  {WATCHDOG_SYNC_INTERVAL_S}s")
    info(
        f"  WATCHDOG_SYNC_PRESSURE_S:  {WATCHDOG_SYNC_PRESSURE_INTERVAL_S}s "
        f"(at >{WATCHDOG_CACHE_PRESSURE_PCT}% L1 usage)"
    )
    info(f"  POD_MEMORY_WARN_PCT:       {POD_MEMORY_WARN_PCT}%")
    info(f"  L1_FS_MAX_UTIL_PCT:        {L1_FS_MAX_UTIL_PCT}%")
    info(f"  L1_FS_MIN_HEADROOM_GB:     {L1_FS_MIN_HEADROOM_GB}GB")

    # Log env var overrides so operators can see what was customized.
    overrides = []  # type: list[str]
    for var in [
        "VLLM_ROCM_CI_GPU_PREFLIGHT",
        "VLLM_ROCM_CI_INFRA_CHECKS",
        "VLLM_ROCM_CI_CACHE_ENABLED",
        "VLLM_ROCM_CI_CACHE_EVICTION",
        "VLLM_ROCM_CI_DOCKER_EVICTION",
        "VLLM_ROCM_CI_DIAGNOSTICS",
        "VLLM_ROCM_CI_JUNIT_OVERRIDE",
        "VLLM_CI_DOCKER_IPC_MODE",
        "VLLM_CI_DOCKER_SHM_SIZE",
        "VLLM_ROCM_CI_RESET_CACHE_COUNTS",
        "VLLM_ROCM_CI_RESET_CACHE_L1",
        "VLLM_ROCM_CI_RESET_CACHE_L2",
        "VLLM_ROCM_CI_RESET_DOCKER",
        "VLLM_ROCM_CI_RESET_OVERLAY",
        "VLLM_TEST_TIMEOUT",
        "VLLM_DISK_THRESHOLD_PCT",
        "VLLM_DISK_THRESHOLD_GB",
        "VLLM_DISK_TARGET_PCT",
        "VLLM_DISK_TARGET_GB",
        "VLLM_DISK_EVICTION_POLICY",
        "VLLM_CI_CACHE_ROOT",
        "VLLM_CI_CACHE_BACKING_ROOT",
        "VLLM_CACHE_L2_MAX_DAYS",
        "VLLM_CI_REGISTRY",
        "DOCKER_IMAGE_NAME",
        "VLLM_LOCAL_IMAGE_CACHE",
        "VLLM_CI_BASE_IMAGE",
        "VLLM_ROCM_CI_WATCHDOG_INTERVAL",
        "VLLM_ROCM_CI_WATCHDOG_SYNC_INTERVAL",
        "VLLM_ROCM_CI_WATCHDOG_SYNC_PRESSURE_INTERVAL",
        "VLLM_ROCM_CI_WATCHDOG_CACHE_PRESSURE_PCT",
        "VLLM_CI_POD_MEMORY_WARN_PCT",
        "VLLM_CACHE_L1_FS_MAX_UTIL_PCT",
        "VLLM_CACHE_L1_FS_MIN_HEADROOM_GB",
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
        try:
            r = sh(
                ["buildkite-agent", "artifact", "upload", pattern],
                capture=True,
                timeout=60,
            )
            if r.returncode != 0:
                warn(f"Artifact upload failed for '{pattern}'")
        except (subprocess.TimeoutExpired, OSError) as exc:
            warn(f"Artifact upload crashed for '{pattern}': {exc}")


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
    try:
        proc = subprocess.run(
            ["buildkite-agent", "annotate", "--style", style, "--context", context],
            input=body,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            warn("buildkite-agent annotate failed")
    except (subprocess.TimeoutExpired, OSError) as exc:
        warn(f"buildkite-agent annotate crashed: {exc}")


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
    try:
        sh(
            ["buildkite-agent", "meta-data", "set", key, value],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        warn(f"buildkite-agent meta-data set '{key}' failed")


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
          "pids_exhausted" (bool), "shm_exhausted" (bool),
          "pid_error" (str), "shm_error" (str), "pytest_ran" (bool),
          "pre_pytest_traceback" (bool),
          "pre_pytest_crash" (str).
    """
    diag = {
        "oom_killed": False,
        "exit_code": -1,
        "error": "",
        "pids_exhausted": False,
        "pid_error": "",
        "shm_exhausted": False,
        "shm_error": "",
        "pytest_ran": True,
        "pre_pytest_traceback": False,
        "pre_pytest_crash": "",
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

    # -- Scan container log --
    #
    # The log has two zones with different trust levels:
    #
    #   PRE-PYTEST: everything before the pytest session header.
    #     Contains only setup output (export, cd, pip install, etc.).
    #     No test output exists here. Tracebacks, "command not found",
    #     and crash patterns are UNAMBIGUOUS in this zone -- they
    #     indicate a genuine pre-test failure.
    #
    #   DURING-PYTEST: everything from the session header onward.
    #     Contains test output that can legitimately include
    #     tracebacks (expected failures, xfail, captured stderr),
    #     "command not found" (tests asserting missing binaries),
    #     and any other string. Pattern matching here is UNRELIABLE
    #     for crash detection.
    #
    # We split the log at the session header and only use aggressive
    # pattern matching in the pre-pytest zone.

    _PYTEST_HEADER = "============================= test session starts"

    if log_file and log_file.is_file():
        try:
            # Read up to 2MB. Pre-pytest output is typically <100KB,
            # but we read more to always find the session header.
            size = log_file.stat().st_size
            read_limit = min(size, 2 * 1024 * 1024)
            with open(log_file, errors="replace") as f:
                log_content = f.read(read_limit)
            # Also read the tail for PID-limit patterns that can
            # appear anywhere in the log (during-pytest zone too).
            with open(log_file, errors="replace") as f:
                if size > 100_000:
                    f.seek(size - 100_000)
                tail = f.read()
        except OSError as exc:
            warn(f"Could not read container log {log_file} for diagnosis: {exc}")
            log_content = ""
            tail = ""

        # Split into pre-pytest and during-pytest zones.
        header_pos = log_content.find(_PYTEST_HEADER)
        if header_pos >= 0:
            pre_pytest = log_content[:header_pos]
            diag["pytest_ran"] = True
        else:
            # No session header found. Either pytest never ran,
            # or the header is beyond our 2MB read window (rare).
            pre_pytest = log_content
            diag["pytest_ran"] = False

        # -- Pre-pytest zone: reliable crash detection --
        # These patterns are SAFE here because no test output
        # has been produced yet. A traceback in this zone IS a
        # crash. A "command not found" here IS a missing binary.
        if pre_pytest:
            # Python traceback in pre-test setup.
            if "Traceback (most recent call last):" in pre_pytest:
                diag["pre_pytest_traceback"] = True

            # Bash/shell crash patterns (unambiguous in this zone).
            pre_test_crash_patterns = [
                ": command not found",
                "ERROR: Could not install packages",
                "error: Failed to download and build",
                "bash: cd:",
                "bash: syntax error",
                "bash: line ",
            ]
            for pattern in pre_test_crash_patterns:
                if pattern in pre_pytest:
                    diag["pre_pytest_crash"] = pattern
                    break

        # -- Full log: resource-limit patterns --
        # PID limit can appear anywhere (during long test runs).
        pid_patterns = [
            "fork: Resource temporarily unavailable",
            "Cannot allocate memory",
            "OSError: [Errno 11]",  # EAGAIN from os.fork()
            "BlockingIOError: [Errno 11]",  # EAGAIN
            "RuntimeError: can't start new thread",
            "pthread_create failed: Resource temporarily unavailable",
            "thread: Resource temporarily unavailable",
        ]
        for pattern in pid_patterns:
            if pattern in tail:
                diag["pids_exhausted"] = True
                for line in reversed(tail.splitlines()):
                    if pattern in line:
                        diag["pid_error"] = line.strip()
                        break
                break

        # Shared-memory exhaustion patterns. Keep this narrower than the PID
        # checks so we only fire when the log clearly points at /dev/shm or
        # shared-memory segment creation, not on unrelated ENOSPC failures.
        shm_patterns = [
            "Error while creating shared memory segment",
            "/dev/shm/",
            "/dev/shm ",
            "shared memory segment",
        ]
        if "No space left on device" in tail and any(p in tail for p in shm_patterns):
            diag["shm_exhausted"] = True
            for line in reversed(tail.splitlines()):
                if (
                    "No space left on device" in line
                    or "/dev/shm" in line
                    or "shared memory segment" in line
                ):
                    diag["shm_error"] = line.strip()
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
            f"    - Increase the K8s pod memory limit\n"
            f"    - Reduce batch size or model size in the test\n"
            f"    - Check for memory leaks (compare pre/post VRAM snapshots above)\n"
            f"    - If /dev/shm growth is suspected, look for the separate\n"
            f"      SHARED MEMORY EXHAUSTED diagnostic below"
        )

    if diag["pids_exhausted"]:
        pid_limit_display = _format_container_pids_limit(CONTAINER_PIDS_LIMIT)
        pid_excerpt = diag["pid_error"] or "(no matching log line captured)"
        fix_lines = [
            "    - Check for process/thread leaks (for example repeated Ray",
            "      workers or daemons surviving between pytest invocations)",
        ]
        if _parse_container_pids_limit(CONTAINER_PIDS_LIMIT) == -1:
            fix_lines += [
                "    - The Docker PIDs cgroup is unlimited; inspect the",
                "      container's `ulimit -u` / nproc limit and host pressure",
            ]
        else:
            fix_lines += [
                "    - Increase VLLM_CI_DOCKER_PIDS_LIMIT "
                f"(currently {pid_limit_display})",
            ]
        error(
            f"{_DIAG_PREFIX} PID / THREAD BUDGET EXHAUSTED\n"
            f"  What happened: New process/thread creation failed inside the\n"
            f"                 container. In Docker this usually means the\n"
            f"                 container exhausted its task budget.\n"
            f"  Container:     {container_name}\n"
            f"  Exit code:     {code}\n"
            f"  PID budget:    {pid_limit_display}\n"
            f"  Log excerpt:   {pid_excerpt}\n"
            f"  How to fix:\n" + "\n".join(fix_lines)
        )

    if diag["shm_exhausted"]:
        shm_excerpt = diag["shm_error"] or "(no matching log line captured)"
        fix_lines = [
            "    - Prefer VLLM_CI_DOCKER_IPC_MODE=private for single-node jobs so",
            "      the container gets its own /dev/shm and --shm-size applies",
        ]
        if _container_uses_private_shm(CONTAINER_IPC_MODE):
            fix_lines += [
                "    - Increase VLLM_CI_DOCKER_SHM_SIZE "
                f"(currently {CONTAINER_SHM_SIZE})",
            ]
        else:
            fix_lines += [
                "    - Increase /dev/shm capacity on the Docker host / outer pod;",
                "      VLLM_CI_DOCKER_SHM_SIZE does not apply with --ipc=host",
            ]
        fix_lines.append(
            "    - Check for leaked /dev/shm/nccl-* segments from prior crashes"
        )
        error(
            f"{_DIAG_PREFIX} SHARED MEMORY EXHAUSTED\n"
            f"  What happened: The workload could not allocate a /dev/shm segment\n"
            f"                 inside the Docker IPC namespace.\n"
            f"  Container:     {container_name}\n"
            f"  Exit code:     {code}\n"
            f"  IPC mode:      {CONTAINER_IPC_MODE}\n"
            f"  Log excerpt:   {shm_excerpt}\n"
            f"  How to fix:\n" + "\n".join(fix_lines)
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


# --------------------------------------------------------------------------
# SMI tool detection: prefer amd-smi, fall back to rocm-smi
# --------------------------------------------------------------------------


def _detect_smi_tool():
    # type: () -> str
    """Detect whether ``amd-smi`` or ``rocm-smi`` is available.

    Prefers ``amd-smi`` (the modern AMD System Management Interface).
    Falls back to ``rocm-smi`` (legacy ROCm SMI) when ``amd-smi`` is
    not on ``$PATH``.

    Returns:
        ``"amd-smi"`` or ``"rocm-smi"``.
    """
    if shutil.which("amd-smi"):
        info("GPU SMI tool detected: amd-smi")
        return "amd-smi"
    if shutil.which("rocm-smi"):
        info("GPU SMI tool detected: rocm-smi (fallback)")
        return "rocm-smi"
    warn("Neither amd-smi nor rocm-smi found on $PATH")
    return "rocm-smi"  # will fail gracefully at call sites


_SMI_TOOL = _detect_smi_tool()


def _smi_cmd(args, *, timeout=ROCM_SMI_TIMEOUT_S, capture=True):
    # type: (str, ...) -> subprocess.CompletedProcess
    """Run the detected SMI tool with *args*.

    Args:
        args:    Arguments to append after the tool name.
        timeout: Maximum seconds to wait.
        capture: If True, capture stdout/stderr.

    Returns:
        ``subprocess.CompletedProcess``.

    Raises:
        subprocess.TimeoutExpired: if the command exceeds *timeout*.
    """
    return sh(
        f"{_SMI_TOOL} {args} 2>/dev/null",
        capture=capture,
        timeout=timeout,
    )


def _smi_json(args, *, timeout=ROCM_SMI_TIMEOUT_S):
    # type: (str, ...) -> object | None
    """Run the SMI tool with ``--json`` and parse the JSON output.

    Args:
        args:    Arguments to append (``--json`` is added automatically).
        timeout: Maximum seconds to wait.

    Returns:
        Parsed JSON (list or dict), or ``None`` on any failure.
    """
    try:
        r = _smi_cmd(f"{args} --json", timeout=timeout)
    except subprocess.TimeoutExpired:
        warn(f"{_SMI_TOOL} {args} --json timed out after {timeout}s")
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except (json.JSONDecodeError, ValueError):
        return None


# --------------------------------------------------------------------------
# VRAM snapshot
# --------------------------------------------------------------------------


def _snapshot_vram_amd_smi(prefix):
    # type: (str) -> dict[int, dict[str, int]]
    """VRAM snapshot via ``amd-smi metric --mem-usage --json``."""
    data = _smi_json("metric --mem-usage")
    if not isinstance(data, list):
        return {}
    result = {}  # type: dict[int, dict[str, int]]
    for entry in data:
        try:
            idx = int(entry["gpu"])
        except (KeyError, TypeError, ValueError):
            continue
        mem = entry.get("mem_usage", {})
        used_info = mem.get("used_vram", {})
        total_info = mem.get("total_vram", {})
        try:
            used = int(used_info["value"]) * 1024 * 1024
            total = int(total_info["value"]) * 1024 * 1024
        except (KeyError, TypeError, ValueError):
            continue
        if total > 0 and used > total:
            warn(
                f"GPU {idx}: VRAM used ({used}) > total "
                f"({total}) -- data may be corrupted"
            )
        result[idx] = {"used": used, "total": total}
    return result


def _snapshot_vram_rocm_smi(prefix):
    # type: (str) -> dict[int, dict[str, int]]
    """VRAM snapshot via ``rocm-smi --showmemuse --json`` (fallback)."""
    try:
        r = sh(
            "rocm-smi --showmemuse --json 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"{prefix}rocm-smi --showmemuse timed out after {ROCM_SMI_TIMEOUT_S}s")
        return {}
    if r.returncode != 0:
        try:
            r = sh(
                "rocm-smi --showmeminfo vram 2>/dev/null",
                capture=True,
                timeout=ROCM_SMI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            warn(
                f"{prefix}rocm-smi --showmeminfo timed out after {ROCM_SMI_TIMEOUT_S}s"
            )
            return {}
        if r.returncode == 0:
            info(f"{prefix}VRAM snapshot:\n{r.stdout.strip()}")
        return {}

    try:
        data = json.loads(r.stdout)
    except (json.JSONDecodeError, ValueError):
        return {}

    result = {}  # type: dict[int, dict[str, int]]
    for key, val in data.items():
        if not key.startswith("card"):
            continue
        try:
            idx = int(key.replace("card", ""))
        except ValueError:
            continue
        used_raw = val.get("VRAM Total Used Memory (B)")
        total_raw = val.get("VRAM Total Memory (B)")
        if used_raw is None or total_raw is None:
            warn(
                f"GPU {idx}: VRAM keys missing from "
                f"rocm-smi JSON -- available keys: "
                f"{list(val.keys())}"
            )
            continue
        used = int(used_raw)
        total = int(total_raw)
        if total > 0 and used > total:
            warn(
                f"GPU {idx}: VRAM used ({used}) > total "
                f"({total}) -- data may be corrupted"
            )
        result[idx] = {"used": used, "total": total}
    return result


def snapshot_gpu_vram(label=""):
    # type: (str) -> dict[int, dict[str, int]]
    """Capture per-GPU VRAM usage via amd-smi/rocm-smi and log it.

    Uses ``amd-smi metric --mem-usage --json`` when available; falls
    back to ``rocm-smi --showmemuse --json`` otherwise.

    Args:
        label: Optional prefix for log lines
            (e.g., "pre-test", "post-test").

    Returns:
        Dict mapping GPU index to ``{"used": bytes, "total": bytes}``.
        Empty dict if neither tool is available.
    """
    prefix = f"[{label}] " if label else ""
    if _SMI_TOOL == "amd-smi":
        result = _snapshot_vram_amd_smi(prefix)
    else:
        result = _snapshot_vram_rocm_smi(prefix)

    if result:
        for idx, mem in sorted(result.items()):
            used_mb = mem["used"] / (1024 * 1024)
            total_mb = mem["total"] / (1024 * 1024)
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


def _read_int_file(path):
    # type: (Path) -> int | None
    """Read an integer-like sysfs/cgroup value from disk.

    Returns None on I/O failure, parse failure, or when the kernel reports
    ``max`` for an unlimited cgroup value.
    """
    try:
        content = path.read_text().strip()
    except OSError:
        return None
    if not content or content == "max":
        return None
    try:
        return int(content)
    except ValueError:
        return None


def get_cgroup_memory_usage():
    # type: () -> tuple[int | None, int | None]
    """Return (used_bytes, limit_bytes) for the current container cgroup.

    Supports both cgroup v2 and legacy v1 layouts so the watchdog can emit
    pod-aware memory diagnostics on mixed CI node images.
    """
    v2_current = Path("/sys/fs/cgroup/memory.current")
    v2_max = Path("/sys/fs/cgroup/memory.max")
    if v2_current.is_file():
        return _read_int_file(v2_current), _read_int_file(v2_max)

    v1_current = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return _read_int_file(v1_current), _read_int_file(v1_limit)


def get_container_uptime_s():
    # type: () -> float | None
    """Return PID 1 runtime in seconds as a proxy for container/pod uptime.

    This is more meaningful in Kubernetes than host uptime because it reflects
    when the current pod started, not when the underlying node booted.
    """
    r = sh(["ps", "-o", "etimes=", "-p", "1"], capture=True, timeout=5)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


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
# ``amd-smi``/``rocm-smi`` queries the AMD kernel driver directly and
# returns PIDs in the *host* PID namespace, regardless of which container
# they came from. This is the only reliable detection method in K8s.
# ==========================================================================


def _device_pids(device):
    # type: (str) -> list[int]
    """Return PIDs holding a device file open, via ``fuser``.

    Limitation: in K8s, fuser only sees PIDs in the current PID namespace.
    Use ``gpu_pids()`` for cross-namespace visibility.

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


def gpu_pids():
    # type: () -> list[int]
    """Get PIDs using GPUs via ``amd-smi``/``rocm-smi``.

    Unlike ``fuser /dev/kfd``, this queries the AMD kernel driver directly
    and returns PIDs in the host PID namespace. In K8s, this sees processes
    from ALL containers on the node, not just the current pod.

    This is the primary zombie-detection method for K8s environments.

    Returns:
        List of integer PIDs with PID > MIN_SYSTEM_PID.
        Empty list if the SMI tool is unavailable or reports no processes.
    """
    if _SMI_TOOL == "amd-smi":
        return _gpu_pids_amd_smi()
    return _gpu_pids_rocm_smi()


def _gpu_pids_amd_smi():
    # type: () -> list[int]
    """PID enumeration via ``amd-smi process --json``."""
    data = _smi_json("process")
    if not isinstance(data, list):
        return []
    pids = []  # type: list[int]
    for entry in data:
        for proc in entry.get("process_list", []):
            try:
                pid = int(proc["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid > MIN_SYSTEM_PID:
                pids.append(pid)
    return pids


def _gpu_pids_rocm_smi():
    # type: () -> list[int]
    """PID enumeration via ``rocm-smi --showpids`` (fallback)."""
    try:
        r = sh(
            "rocm-smi --showpids 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"rocm-smi --showpids timed out after {ROCM_SMI_TIMEOUT_S}s")
        return []
    if r.returncode != 0 or not r.stdout.strip():
        return []

    pids = []  # type: list[int]
    for line in r.stdout.splitlines():
        for token in line.split():
            if token.isdigit():
                pid = int(token)
                if pid > MIN_SYSTEM_PID:
                    pids.append(pid)
                break  # only first numeric token per line
    return pids


def check_vram_clear():
    # type: () -> bool
    """Verify that no GPUs have allocated VRAM.

    Uses ``amd-smi metric --mem-usage --json`` when available; falls
    back to ``rocm-smi --showmeminfo vram`` text parsing otherwise.

    Returns:
        True if all GPUs report zero VRAM usage (or if the SMI tool
        fails, in which case we return True optimistically to avoid
        blocking CI).
    """
    if _SMI_TOOL == "amd-smi":
        return _check_vram_clear_amd_smi()
    return _check_vram_clear_rocm_smi()


def _check_vram_clear_amd_smi():
    # type: () -> bool
    """VRAM check via ``amd-smi metric --mem-usage --json``."""
    data = _smi_json("metric --mem-usage")
    if not isinstance(data, list):
        warn("amd-smi mem-usage query failed -- cannot verify VRAM")
        annotate_build(
            "### :warning: VRAM check skipped "
            "(amd-smi failed)\n\n"
            "GPU VRAM state could not be verified. Tests may "
            "run on a GPU with leaked VRAM from a previous job.",
            style="warning",
            context="vram-check-skip",
        )
        return True
    for entry in data:
        mem = entry.get("mem_usage", {})
        used_info = mem.get("used_vram", {})
        try:
            val = int(used_info["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if val > 0:
            gpu = entry.get("gpu", "?")
            warn(f"GPU {gpu}: VRAM still in use ({val} MB)")
            return False
    return True


def _check_vram_clear_rocm_smi():
    # type: () -> bool
    """VRAM check via ``rocm-smi --showmeminfo vram`` (fallback)."""
    try:
        r = sh(
            "rocm-smi --showmeminfo vram 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"rocm-smi --showmeminfo timed out after {ROCM_SMI_TIMEOUT_S}s")
        annotate_build(
            "### :warning: VRAM check skipped "
            "(rocm-smi timeout)\n\n"
            "GPU VRAM state could not be verified. Tests may "
            "run on a GPU with leaked VRAM from a previous job.",
            style="warning",
            context="vram-check-skip",
        )
        return True
    if r.returncode != 0:
        warn("rocm-smi --showmeminfo failed -- cannot verify VRAM")
        annotate_build(
            "### :warning: VRAM check skipped "
            "(rocm-smi failed)\n\n"
            "GPU VRAM state could not be verified. Tests may "
            "run on a GPU with leaked VRAM from a previous job.",
            style="warning",
            context="vram-check-skip",
        )
        return True

    for line in r.stdout.splitlines():
        lower = line.lower()
        if "used" in lower and ":" in line:
            value_part = line.rsplit(":", 1)[1].strip()
            cleaned = re.sub(r"[BKMGT]i?[Bb]?$", "", value_part.rstrip()).strip()
            try:
                val = float(cleaned)
                if val > 0:
                    warn(f"GPU VRAM still in use: {line.strip()}")
                    return False
            except (ValueError, OverflowError):
                pass
    return True


def gpu_hard_reset():
    # type: () -> bool
    """Attempt a hardware-level GPU reset via amd-smi/rocm-smi.

    This is the last resort when soft reset (writing "reset" to gpu_state)
    and zombie cleanup have both failed to free GPU memory. It resets the
    GPU at the PCIe level, which clears all VRAM and kills any remaining
    GPU contexts.

    Returns:
        True if the reset succeeded, False otherwise.
    """
    info(f"Attempting hardware GPU reset via {_SMI_TOOL}...")
    if _SMI_TOOL == "amd-smi":
        return _hard_reset_amd_smi()
    return _hard_reset_rocm_smi()


def _hard_reset_amd_smi():
    # type: () -> bool
    """GPU reset via ``amd-smi reset --gpureset --gpu all``."""
    try:
        r = _smi_cmd("reset --gpureset --gpu all")
    except subprocess.TimeoutExpired:
        warn(f"amd-smi reset timed out after {ROCM_SMI_TIMEOUT_S}s")
        return False
    if r.returncode == 0:
        info("amd-smi reset --gpureset succeeded")
        return True
    warn(f"amd-smi reset failed (rc={r.returncode}): {r.stdout.strip()}")
    return False


def _hard_reset_rocm_smi():
    # type: () -> bool
    """GPU reset via ``rocm-smi --gpureset -d <idx>`` (fallback).

    ``rocm-smi --gpureset`` requires exactly one device, so we
    enumerate GPU indices first and reset each one individually.
    """
    # Enumerate GPU indices from --showid text output.
    try:
        r = sh(
            "rocm-smi --showid 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"rocm-smi --showid timed out after {ROCM_SMI_TIMEOUT_S}s")
        return False
    if r.returncode != 0:
        warn("rocm-smi --showid failed -- cannot enumerate GPUs")
        return False

    indices = []  # type: list[int]
    for line in r.stdout.splitlines():
        m = re.search(r"GPU\[(\d+)\]", line)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        warn("rocm-smi --showid returned no GPU indices")
        return False

    all_ok = True
    for idx in indices:
        try:
            r = sh(
                f"rocm-smi --gpureset -d {idx} 2>&1",
                capture=True,
                timeout=ROCM_SMI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            warn(f"rocm-smi --gpureset -d {idx} timed out after {ROCM_SMI_TIMEOUT_S}s")
            all_ok = False
            continue
        if r.returncode != 0:
            warn(
                f"rocm-smi --gpureset -d {idx} failed "
                f"(rc={r.returncode}): {r.stdout.strip()}"
            )
            all_ok = False
        else:
            info(f"rocm-smi --gpureset -d {idx} succeeded")
    return all_ok


def validate_gpu_health():
    # type: () -> None
    """Pre-flight GPU health validation via amd-smi/rocm-smi.

    Checks three things before allowing tests to run:

    1. GPU enumeration: Confirms GPUs are visible. Fails hard if not --
       likely means the K8s device plugin did not allocate any GPUs to
       this pod.

    2. Temperature: Warns if any GPU exceeds 90C. Does not fail --
       thermal throttling degrades performance but tests may still pass.

    3. ECC errors: Warns on uncorrectable errors. Does not fail --
       hardware errors are the infra team's problem, not the test's.
    """
    section("Validating GPU health")

    if _SMI_TOOL == "amd-smi":
        _validate_health_amd_smi()
    else:
        _validate_health_rocm_smi()


def _validate_health_amd_smi():
    # type: () -> None
    """Health validation via ``amd-smi``."""
    # 1. Enumeration
    data = _smi_json("list")
    if not isinstance(data, list) or not data:
        error("amd-smi list failed -- GPUs may not be accessible")
        error("Check that the K8s device plugin allocated GPUs to this pod")
        sys.exit(1)
    info(f"amd-smi: {len(data)} GPU(s) detected")

    # 2. Temperature
    temp_data = _smi_json("metric --temperature")
    if isinstance(temp_data, list):
        for entry in temp_data:
            gpu = entry.get("gpu", "?")
            temp_info = entry.get("temperature", {})
            hotspot = temp_info.get("hotspot", {})
            try:
                val = float(hotspot["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if val > 90.0:
                warn(f"GPU {gpu}: hotspot temperature {val}C > 90C -- throttling risk")

    # 3. ECC errors
    ecc_data = _smi_json("metric --ecc")
    if isinstance(ecc_data, list):
        for entry in ecc_data:
            gpu = entry.get("gpu", "?")
            ecc = entry.get("ecc", {})
            try:
                uncorr = int(ecc["total_uncorrectable_count"])
            except (KeyError, TypeError, ValueError):
                continue
            if uncorr > 0:
                warn(f"GPU {gpu}: {uncorr} uncorrectable ECC error(s)")


def _validate_health_rocm_smi():
    # type: () -> None
    """Health validation via ``rocm-smi`` (fallback)."""
    # 1. Enumeration
    try:
        r = sh(
            "rocm-smi --showid 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        error(f"rocm-smi --showid timed out after {ROCM_SMI_TIMEOUT_S}s")
        sys.exit(1)
    if r.returncode != 0:
        error("rocm-smi --showid failed -- GPUs may not be accessible")
        error("Check that the K8s device plugin allocated GPUs to this pod")
        sys.exit(1)
    info(r.stdout.strip())

    # 2. Temperature
    try:
        r = sh(
            "rocm-smi --showtemp 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"rocm-smi --showtemp timed out after {ROCM_SMI_TIMEOUT_S}s")
        r = None
    if r is not None and r.returncode == 0:
        info(r.stdout.strip())
        for line in r.stdout.splitlines():
            lower = line.lower()
            if "temp" not in lower and "temperature" not in lower:
                continue
            for token in line.split():
                try:
                    temp = float(token)
                    if temp > 90.0:
                        warn(f"GPU temperature {temp}C > 90C -- throttling risk")
                except ValueError:
                    continue

    # 3. ECC errors
    try:
        r = sh(
            "rocm-smi --showrasinfo 2>/dev/null",
            capture=True,
            timeout=ROCM_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warn(f"rocm-smi --showrasinfo timed out after {ROCM_SMI_TIMEOUT_S}s")
        r = None
    if r is not None and r.returncode == 0 and r.stdout.strip():
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

    2. ``amd-smi``/``rocm-smi`` process query:
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

    # Method 2: amd-smi/rocm-smi (cross PID namespace -- the one that matters in K8s)
    driver_pids = gpu_pids()
    if driver_pids:
        found = True
        warn(f"{_SMI_TOOL} reports GPU processes (driver-level): {driver_pids}")
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
        if not check_vram_clear():
            warn("VRAM still allocated after cleanup -- attempting hardware reset")
            gpu_hard_reset()
            time.sleep(3)
            if not check_vram_clear():
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
    instead (see ``validate_gpu_health``).

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
#   - Kubernetes-managed model/test caches: the outer pod keeps the HF tree
#     warm across jobs, while this runner manages only build-tool caches.
#   - Ephemeral storage: set VLLM_CI_CACHE_ROOT to a fast local disk
#     (NVMe /scratch) instead of shared NFS for better I/O.
#   - Single multi-arch builds: the ccache entry still caches compiled
#     objects across jobs even though the pipeline now publishes one image
#     tag per commit instead of separate per-arch tags.
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

    validate_cache_layout()

    # Log tier configuration and storage types.
    hot_type = _detect_storage_type(
        CACHE_ROOT if CACHE_ROOT.exists() else CACHE_ROOT.parent
    )
    info(f"Hot tier (CACHE_ROOT):     {CACHE_ROOT} [{hot_type}]")
    if CACHE_BACKING_ROOT is not None:
        backing_type = _detect_storage_type(
            CACHE_BACKING_ROOT
            if CACHE_BACKING_ROOT.exists()
            else CACHE_BACKING_ROOT.parent
        )
        info(f"Warm tier (backing):       {CACHE_BACKING_ROOT} [{backing_type}]")
        info("Mode: two-tier (fast local + persistent backing)")
    else:
        info("Warm tier (backing):       not configured")
        info("Mode: single-tier (CACHE_ROOT is both hot and warm)")

    if not CACHE_ROOT.exists():
        info(f"  Creating cache root: {CACHE_ROOT}")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    info(
        f"Kubernetes-managed cache tree: {K8S_CACHE_HOST_ROOT} -> "
        f"{K8S_CACHE_CONTAINER_ROOT}"
    )

    # Try OverlayFS first (L1+L2 merged transparently by kernel). If some
    # caches fail to mount, seed only those caches from backing so partial
    # overlay failure does not leave them completely cold.
    mounted_subdirs = mount_overlay_caches()
    sync_caches_from_backing(skip_subdirs=mounted_subdirs)

    # Access-frequency seeding only matters for two-tier caches. In
    # single-tier mode there is no backing store to seed from, so the
    # "first run / no access data" message is just noise.
    if CACHE_BACKING_ROOT is not None:
        log_top_k_access_counts()

    cache_bytes = {}  # type: dict[str, int]
    for cache in CACHES:
        host_dir = _get_cache_host_dir(cache)
        host_dir.mkdir(parents=True, exist_ok=True)
        max_gb = _get_cache_max_gb(cache)
        override_gb = _get_cache_env_override_gb(cache)
        default_gb = int(cache.get("max_gb", 0))

        # Report cache size, file count, and limit.
        try:
            r = sh(
                f"du -sh '{host_dir}' 2>/dev/null | cut -f1",
                capture=True,
            )
            size = r.stdout.strip() if r.returncode == 0 else "?"
            r2 = sh(
                f"find '{host_dir}' -type f 2>/dev/null | wc -l",
                capture=True,
            )
            count = r2.stdout.strip() if r2.returncode == 0 else "?"
            r3 = sh(
                f"du -sb '{host_dir}' 2>/dev/null | cut -f1",
                capture=True,
            )
            cache_bytes[cache["env_var"]] = (
                int(r3.stdout.strip())
                if r3.returncode == 0 and r3.stdout.strip()
                else 0
            )
        except (OSError, subprocess.SubprocessError):
            size, count = "?", "?"
            cache_bytes[cache["env_var"]] = 0

        try:
            is_warm = count != "?" and int(count) > 0
            status = "warm" if is_warm else "cold"
        except ValueError:
            status = "unknown"

        if max_gb > 0:
            limit = f"{max_gb}GB"
            if (
                CACHE_BACKING_ROOT is None
                and override_gb is None
                and default_gb > 0
                and max_gb != default_gb
            ):
                limit = f"{limit} (scaled from {default_gb}GB)"
        else:
            limit = "unlimited"
        env = cache["env_var"]
        info(f"  {env:25s} {size:>8s} ({count} files) [{status}] limit={limit}")

    log_cache_budget_guardrails(cache_bytes)

    return CACHES


def build_cache_docker_args():
    # type: () -> list[str]
    """Build docker run arguments for all persistent cache mounts.

    Returns a list of ``-v`` and ``-e`` flags to pass to ``docker run``:
      - one bind mount for the Kubernetes-managed HF cache tree
      - ``-e ENV_VAR=container_path`` for the Kubernetes-managed cache envs
      - ``-v host_dir:container_path`` for each runner-managed cache
      - ``-e ENV_VAR=container_path`` for each runner-managed cache

    This is the single place where cache mounts are defined. The long-lived
    model/test caches come from the outer Buildkite pod's HF tree; the caches
    in CACHES are the runner-managed hot-tier caches under CACHE_ROOT.

    Returns:
        List of docker CLI arguments (strings).
        Empty list if ENABLE_CACHE is False.
    """
    if not ENABLE_CACHE:
        return []
    args = []  # type: list[str]
    args += ["-v", f"{K8S_CACHE_HOST_ROOT}:{K8S_CACHE_CONTAINER_ROOT}"]
    for env_var, container_path in K8S_MANAGED_CACHE_ENVS:
        args += ["-e", f"{env_var}={container_path}"]

    if CACHE_RUNTIME_FAILED:
        return args

    for cache in CACHES:
        # If overlay is active for this cache, mount the merged view
        # (which transparently includes both L1 and L2).
        # Otherwise, mount the raw L1 directory.
        overlay_merged = CACHE_OVERLAY_ROOT / cache["host_subdir"]
        if (
            not cache.get("host_dir_override")
            and cache["host_subdir"] in OVERLAY_ACTIVE_SUBDIRS
            and overlay_merged.exists()
            and overlay_merged.is_mount()
        ):
            host_dir = overlay_merged
        else:
            host_dir = _get_cache_host_dir(cache)
        args += ["-v", f"{host_dir}:{cache['container_path']}"]
        args += ["-e", f"{cache['env_var']}={cache['container_path']}"]
    return args


def log_cache_stats_diff(label):
    # type: (str) -> None
    """Log cache sizes for post-mortem comparison.

    Called before and after test execution. By diffing the two snapshots,
    you can see which caches grew (new downloads) and by how much.

    Args:
        label: Tag for the snapshot (e.g., "pre-test", "post-test").
    """
    info(f"Cache stats [{label}]:")
    for cache in CACHES:
        host_dir = _get_cache_host_dir(cache)
        if not host_dir.exists():
            continue
        try:
            r = sh(
                f"du -sh '{host_dir}' 2>/dev/null | cut -f1",
                capture=True,
            )
            size = r.stdout.strip() if r.returncode == 0 else "?"
        except (OSError, subprocess.SubprocessError):
            size = "?"
        max_gb = _get_cache_max_gb(cache)
        limit = f"/{max_gb}GB" if max_gb > 0 else ""
        info(f"  {cache['env_var']:25s} {size}{limit}")


def _get_cache_host_dir(cache):
    # type: (dict) -> Path
    """Resolve the hot-tier host directory for a cache entry."""
    override = cache.get("host_dir_override")
    return Path(override) if override else CACHE_ROOT / cache["host_subdir"]


def _get_cache_backing_dir(cache):
    # type: (dict) -> Path | None
    """Resolve the warm-tier (backing) directory for a cache entry.

    Returns None if two-tier caching is not configured, or if this
    cache uses a host_dir_override (e.g., the legacy HF path -- the
    override IS the persistent path, so there's no separate backing).
    """
    if CACHE_BACKING_ROOT is None:
        return None
    if cache.get("host_dir_override"):
        # Legacy override caches manage their own persistence.
        return None
    return CACHE_BACKING_ROOT / cache["host_subdir"]


def _get_filesystem_usage_bytes(path):
    # type: (Path) -> tuple[int, int, int]
    """Return (used, free, total) bytes for the filesystem containing path.

    Cache guardrails are applied at the shared-filesystem level because
    multiple caches may be individually below budget while the underlying
    NVMe volume is still close to eviction pressure.
    """
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(str(probe))
    return usage.used, usage.free, usage.total


def _get_filesystem_growth_budget_bytes(path):
    # type: (Path) -> int
    """Return how many more bytes may safely be written to this filesystem.

    The returned budget respects both L1 safety env vars:
      - ``VLLM_CACHE_L1_FS_MIN_HEADROOM_GB``
      - ``VLLM_CACHE_L1_FS_MAX_UTIL_PCT``

    The smaller remaining allowance wins. If neither allows more growth,
    callers should treat the filesystem as under pressure.
    """
    try:
        used, free, total = _get_filesystem_usage_bytes(path)
    except OSError:
        return 0

    headroom_gb = min(L1_FS_MIN_HEADROOM_GB, (total / (1024**3)) * 0.1)
    headroom_bytes = int(headroom_gb * 1024**3)
    allowed_by_headroom = max(0, free - headroom_bytes)
    if L1_FS_MAX_UTIL_PCT > 0:
        max_used_bytes = int(total * (L1_FS_MAX_UTIL_PCT / 100.0))
        allowed_by_util = max(0, max_used_bytes - used)
        return min(allowed_by_headroom, allowed_by_util)
    return allowed_by_headroom


def get_cache_seed_budget_bytes(cache):
    # type: (dict) -> int
    """Return how many bytes may be seeded into L1 for this cache.

    This budget is the minimum of:
      1. The cache's remaining L1 budget (``max_gb - current_size``).
      2. The shared filesystem growth budget for the cache's host path.

    That keeps startup seeding from overfilling a warm cache or blowing past
    node-level disk headroom just because L2 contains more reusable data.
    """
    host_dir = _get_cache_host_dir(cache)
    current_bytes = _get_dir_size_bytes(host_dir) if host_dir.exists() else 0
    fs_budget = _get_filesystem_growth_budget_bytes(host_dir)
    if fs_budget <= 0:
        return 0

    max_gb = _get_cache_max_gb(cache)
    if max_gb <= 0:
        return fs_budget

    max_bytes = max_gb * 1024 * 1024 * 1024
    per_cache_remaining = max(0, max_bytes - current_bytes)
    return min(per_cache_remaining, fs_budget)


def cache_under_pressure(cache):
    # type: (dict) -> bool
    """Return True when a cache or its filesystem should sync aggressively.

    Pressure is triggered by either:
      - the shared L1 filesystem having no remaining growth budget, or
      - the cache exceeding ``WATCHDOG_CACHE_PRESSURE_PCT`` of its L1 budget.

    The watchdog uses this to switch from normal to fast L1->L2 sync cadence.
    """
    host_dir = _get_cache_host_dir(cache)
    if not host_dir.exists():
        return False

    if _get_filesystem_growth_budget_bytes(host_dir) <= 0:
        return True

    max_gb = _get_cache_max_gb(cache)
    if max_gb <= 0:
        return False

    current_gb = _get_dir_size_bytes(host_dir) / (1024 * 1024 * 1024)
    return current_gb > max_gb * (WATCHDOG_CACHE_PRESSURE_PCT / 100.0)


def _normalize_cache_path(path):
    # type: (Path) -> Path
    """Return a stable absolute path for overlap checks."""
    return path.expanduser().resolve(strict=False)


def _cache_paths_overlap(a, b):
    # type: (Path, Path) -> bool
    """Return True when two cache roots overlap on disk."""
    a_norm = _normalize_cache_path(a)
    b_norm = _normalize_cache_path(b)
    return a_norm == b_norm or a_norm in b_norm.parents or b_norm in a_norm.parents


def validate_cache_layout():
    # type: () -> None
    """Fail fast on overlapping host cache roots.

    Overlaps cause the same bytes to count against multiple cache budgets,
    which makes LRU eviction misleading and can leave the node vulnerable to
    disk pressure before any single cache appears over limit.
    """
    roots = []  # type: list[tuple[str, Path]]
    for cache in CACHES:
        roots.append((cache["env_var"], _get_cache_host_dir(cache)))

    conflicts = []  # type: list[str]
    for idx, (env_a, path_a) in enumerate(roots):
        for env_b, path_b in roots[idx + 1 :]:
            if _cache_paths_overlap(path_a, path_b):
                conflicts.append(f"{env_a}={path_a} overlaps {env_b}={path_b}")

    if conflicts:
        joined = "; ".join(conflicts)
        raise RuntimeError(
            "Invalid cache layout: overlapping L1 cache roots detected. "
            "Use separate sibling paths for each cache family so budget "
            f"accounting remains accurate. Conflicts: {joined}"
        )


def log_cache_budget_guardrails(cache_bytes):
    # type: (dict[str, int]) -> None
    """Warn when grouped L1 cache budgets leave too little filesystem headroom."""
    section("Cache budget guardrails")

    groups = {}  # type: dict[int, dict[str, object]]
    for cache in CACHES:
        max_gb = _get_cache_max_gb(cache)
        if max_gb <= 0:
            continue

        host_dir = _get_cache_host_dir(cache)
        probe = host_dir if host_dir.exists() else host_dir.parent
        try:
            st = probe.stat()
            usage = shutil.disk_usage(str(probe))
        except OSError:
            continue

        group = groups.setdefault(
            st.st_dev,
            {
                "path": str(probe),
                "total_gb": usage.total / (1024**3),
                "free_gb": usage.free / (1024**3),
                "budget_gb": 0.0,
                "current_gb": 0.0,
                "members": [],
            },
        )
        group["budget_gb"] += float(max_gb)
        group["current_gb"] += cache_bytes.get(cache["env_var"], 0) / (1024**3)
        group["members"].append(cache["env_var"])

    if not groups:
        info("  Filesystem budget checks unavailable")
        return

    for group in groups.values():
        budget_gb = float(group["budget_gb"])
        current_gb = float(group["current_gb"])
        total_gb = float(group["total_gb"])
        free_gb = float(group["free_gb"])
        projected_growth_gb = max(0.0, budget_gb - current_gb)
        projected_free_gb = free_gb - projected_growth_gb
        util_pct = (budget_gb / total_gb * 100.0) if total_gb > 0 else 0.0
        members = ", ".join(group["members"])

        info(
            "  "
            f"{group['path']}: budget={budget_gb:.0f}GB "
            f"current={current_gb:.1f}GB free={free_gb:.0f}GB "
            f"projected_free_at_limit={projected_free_gb:.0f}GB "
            f"(caches: {members})"
        )

        required_headroom_gb = min(L1_FS_MIN_HEADROOM_GB, total_gb * 0.1)
        if util_pct > L1_FS_MAX_UTIL_PCT or projected_free_gb < required_headroom_gb:
            warn(
                f"  Cache budgets on {group['path']} are close to filesystem "
                f"limits ({util_pct:.0f}% of total, projected free "
                f"{projected_free_gb:.0f}GB). Reduce VLLM_CACHE_MAX_* or move "
                "cache roots to a different mount to avoid disk-pressure evictions."
            )


def _overlay_supported():
    # type: () -> bool
    """Return True if OverlayFS mounts are possible.

    Requirements:
      1. The overlay kernel module is available.
      2. We have permission to mount (CAP_SYS_ADMIN or running as root).
      3. L1 and L2 are on filesystems that support overlay (not all do).

    If any check fails, we fall back to seed-based sync.
    """
    # Check kernel module.
    r = sh("modprobe overlay 2>/dev/null || true", capture=True)
    r = sh(
        "cat /proc/filesystems 2>/dev/null | grep -q overlay",
        capture=True,
    )
    if r.returncode != 0:
        return False
    # Check mount permission with a dry-run style test.
    # We try to mount a trivial overlay; if it fails, we can't use it.
    test_dir = CACHE_ROOT.parent / ".overlay_test"
    lower = test_dir / "lower"
    upper = test_dir / "upper"
    work = test_dir / "work"
    merged = test_dir / "merged"
    try:
        for d in (lower, upper, work, merged):
            d.mkdir(parents=True, exist_ok=True)
        r = sh(
            f"mount -t overlay overlay "
            f"-o lowerdir='{lower}',upperdir='{upper}',"
            f"workdir='{work}' '{merged}' 2>/dev/null",
            capture=True,
            timeout=30,
        )
        if r.returncode == 0:
            sh(f"umount '{merged}' 2>/dev/null || true", timeout=10)
            shutil.rmtree(str(test_dir), ignore_errors=True)
            return True
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass
    shutil.rmtree(str(test_dir), ignore_errors=True)
    return False


def mount_overlay_caches():
    # type: () -> set[str]
    """Mount OverlayFS for each cache, merging L1 (local) and L2 (NFS).

    Creates a merged view at CACHE_OVERLAY_ROOT/<subdir> for each cache
    where both L1 and L2 directories exist. The container mounts point
    to the merged view instead of the raw L1 directory.

    When a file is read:
      - If it exists in L1 (upper): read from fast local storage.
      - If it exists in L2 (lower): read from NFS (transparent fallback).
    When a file is written:
      - Always goes to L1 (upper). L2 is read-only from overlay's view.

    Returns the set of cache subdirs whose overlay mount succeeded.
    Callers can use this to skip redundant seed-based L2->L1 copies for caches
    already covered by a live overlay mount.
    """
    global OVERLAY_ACTIVE_SUBDIRS
    OVERLAY_ACTIVE_SUBDIRS = set()
    if CACHE_BACKING_ROOT is None:
        return set()

    if not _overlay_supported():
        info("OverlayFS not available -- using seed-based fallback")
        return set()

    section("Mounting OverlayFS cache layers (L1 + L2)")
    CACHE_OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_OVERLAY_WORK.mkdir(parents=True, exist_ok=True)

    mounted = 0
    for cache in CACHES:
        backing = _get_cache_backing_dir(cache)
        if backing is None or not backing.exists():
            continue

        local = _get_cache_host_dir(cache)
        local.mkdir(parents=True, exist_ok=True)
        merged = CACHE_OVERLAY_ROOT / cache["host_subdir"]
        work = CACHE_OVERLAY_WORK / cache["host_subdir"]
        merged.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)

        env = cache["env_var"]
        try:
            r = sh(
                f"mount -t overlay overlay "
                f"-o lowerdir='{backing}',"
                f"upperdir='{local}',"
                f"workdir='{work}' "
                f"'{merged}' 2>&1",
                capture=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            warn(f"  {env}: overlay mount timed out after 30s (NFS may be unreachable)")
            continue
        if r.returncode == 0:
            info(f"  {env}: overlay mounted (L1={local}, L2={backing})")
            mounted += 1
            OVERLAY_ACTIVE_SUBDIRS.add(cache["host_subdir"])
        else:
            warn(f"  {env}: overlay mount failed: {r.stdout.strip()}")

    if mounted > 0:
        info(f"Mounted {mounted} overlay cache(s)")
    return set(OVERLAY_ACTIVE_SUBDIRS)


def unmount_overlay_caches():
    # type: () -> None
    """Unmount all OverlayFS mounts. Called during cleanup."""
    if not CACHE_OVERLAY_ROOT.exists():
        return
    for cache in CACHES:
        merged = CACHE_OVERLAY_ROOT / cache["host_subdir"]
        if merged.exists() and merged.is_mount():
            env = cache["env_var"]
            sh(f"umount '{merged}' 2>/dev/null || true")
            info(f"  {env}: overlay unmounted")
    OVERLAY_ACTIVE_SUBDIRS.clear()


# ---- Access frequency tracking ----
#
# Tracks how many times each cache subpath has been accessed across
# jobs. Stored as a simple "count path" text file. Used to prioritize
# seeding when OverlayFS is not available: high-frequency files are
# copied first, so the most-requested models end up in L1.
#
# The log is updated from container access (via the container log or
# by scanning atime after the test) and persisted to backing.


def _load_access_counts():
    # type: () -> dict[str, int]
    """Load the access frequency log from disk.

    Returns a dict mapping relative file paths to access counts.
    If the log doesn't exist or can't be parsed, returns empty dict.
    """
    counts = {}  # type: dict[str, int]
    # Try backing store first (survives pod death), then local.
    for log_path in [
        CACHE_BACKING_ROOT / ".access_counts" if CACHE_BACKING_ROOT else None,
        ACCESS_LOG_FILE,
    ]:
        if log_path is None or not log_path.is_file():
            continue
        try:
            for line in log_path.read_text().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    counts[parts[1]] = int(parts[0])
            if counts:
                return counts
        except OSError:
            continue
    return counts


def _save_access_counts(counts):
    # type: (dict[str, int]) -> None
    """Save the access frequency log to disk (both L1 and L2)."""
    if not counts:
        return
    # Sort by count descending for human readability.
    lines = [
        f"{count} {path}" for path, count in sorted(counts.items(), key=lambda x: -x[1])
    ]
    content = "\n".join(lines) + "\n"

    def _atomic_write(target):
        # type: (Path) -> None
        """Write via temp file + rename for crash-safe concurrent access."""
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(target.parent),
                suffix=".tmp",
            )
            try:
                os.write(fd, content.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, str(target))
        except OSError:
            if tmp is not None:
                with suppress(OSError):
                    os.unlink(tmp)

    # Write to local.
    with suppress(OSError):
        _atomic_write(ACCESS_LOG_FILE)
    # Write to backing if available.
    if CACHE_BACKING_ROOT is not None:
        with suppress(OSError):
            backing_log = CACHE_BACKING_ROOT / ".access_counts"
            _atomic_write(backing_log)


def log_top_k_access_counts(k=15):
    # type: (int) -> None
    """Log the top-K most frequently accessed cache paths.

    Printed as a table in the Buildkite log so operators can see which
    models and files are hot across jobs. Useful for:
      - Deciding which models to pre-bake into the Docker image.
      - Tuning per-cache max_gb limits.
      - Spotting unexpected access patterns (e.g., a test downloading
        a 30GB model that should be in the base image).

    Args:
        k: Number of entries to show (default 15).
    """
    counts = _load_access_counts()
    if not counts:
        info("No access frequency data yet (first run)")
        return

    section(f"Cache access frequency (top {k})")
    info(f"  {'Hits':>6s}  {'Cache tier':15s}  Path")
    info(f"  {'----':>6s}  {'----------':15s}  ----")

    # Group by cache tier (first path component).
    top = sorted(counts.items(), key=lambda x: -x[1])[:k]
    for path, count in top:
        # First component is the cache subdir (e.g., "huggingface").
        parts = path.split("/", 1)
        tier = parts[0] if parts else "?"
        relpath = parts[1] if len(parts) > 1 else path
        # Truncate long paths for readability.
        if len(relpath) > 60:
            relpath = "..." + relpath[-57:]
        info(f"  {count:>6d}  {tier:15s}  {relpath}")

    total_entries = len(counts)
    total_hits = sum(counts.values())
    info("  ------")
    info(f"  {total_entries} tracked paths, {total_hits} total hits across all jobs")


def update_access_counts_from_atime(cache, counts=None):
    # type: (dict, dict[str, int] | None) -> dict[str, int]
    """Update access counts by scanning which files were recently modified.

    After a test run, files with recent mtime were written during the test.
    We increment their access counts. This gives us a frequency signal
    for future seeding decisions.

    IMPORTANT: this tracks *download* frequency, not *read* frequency.
    We use mtime (modification time) because most filesystems mount with
    noatime/relatime, making atime unreliable. This means a model that
    is read from cache 100 times without being re-downloaded will only
    have count=1. The practical effect: the frequency table reflects
    which models are downloaded most often (cache misses), not which
    are used most often. For seeding/eviction this is still useful --
    frequently downloaded models are the ones most worth keeping warm.

    Args:
        cache:  A single CACHES entry.
        counts: Running counts dict to update. If None, loads from disk.
                Pass the return value from a previous call to accumulate
                counts across multiple caches without data loss.

    Returns:
        Updated counts dict (merged with existing counts).
    """
    if counts is None:
        counts = _load_access_counts()
    host_dir = _get_cache_host_dir(cache)
    if not host_dir.exists():
        return counts

    # Find files modified in the last 3 hours (covers the test run).
    # Uses mtime (not atime) because many filesystems mount with
    # noatime/relatime, making atime unreliable.
    r = sh(
        f"find '{host_dir}' -type f -mmin -180 -printf '%P\\n' 2>/dev/null",
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return counts

    subdir = cache["host_subdir"]
    for relpath in r.stdout.strip().splitlines():
        key = f"{subdir}/{relpath}"
        counts[key] = counts.get(key, 0) + 1

    return counts


def sync_caches_from_backing(skip_subdirs=None):
    # type: (set[str] | None) -> None
    """Seed L1 (hot) from L2 (backing) using frequency-aware selection.

    Called at pod startup (Phase 5) ONLY when OverlayFS is not available.
    When OverlayFS is mounted, seeding is unnecessary because the kernel
    handles L1/L2 read-through transparently.

    The backing store (NFS/PVC) may be much larger than L1 (e.g., 14TB
    vs 2TB), so we do NOT copy everything. Instead we select files that
    fit within each cache's max_gb limit, prioritizing by:

      1. Access frequency (from .access_counts log) -- most-accessed first.
      2. Modification time (newest first) -- tiebreaker for equal frequency.

    This means frequently-used models (like the ones most tests need)
    are seeded first, even if they're not the newest.

    Algorithm per cache:
      1. List files in backing with size and mtime.
      2. Sort by (access_count desc, mtime desc).
      3. Copy files one-by-one until we hit the cache's max_gb limit
         or run out of files.
      3. Use rsync with --files-from for efficient transfer.

    If max_gb is 0 (unlimited), we cap the seed at the ephemeral
    volume's available space minus a 10% safety margin.

    ``skip_subdirs`` lets callers exclude caches that already have a working
    overlay mount, so partial overlay success does not suppress fallback seeding
    for unrelated caches.

    Skipped if CACHE_BACKING_ROOT is not set (single-tier mode).
    """
    if CACHE_BACKING_ROOT is None:
        return
    skip_subdirs = skip_subdirs or set()

    section("Cache seed: frequency-aware L2 -> L1 copy")
    info(f"Backing root: {CACHE_BACKING_ROOT}")

    # Load access frequency data from previous jobs.
    access_counts = _load_access_counts()
    if access_counts:
        info(
            f"Loaded access counts: {len(access_counts)} entries "
            f"(top: {list(access_counts.items())[:3]})"
        )
    else:
        info("No access history yet -- seeding by mtime only")

    if not CACHE_BACKING_ROOT.exists():
        warn(
            f"Backing root {CACHE_BACKING_ROOT} does not exist "
            f"-- skipping seed (cold start)"
        )
        return

    for cache in CACHES:
        if cache["host_subdir"] in skip_subdirs:
            continue
        backing = _get_cache_backing_dir(cache)
        if backing is None:
            continue
        local = _get_cache_host_dir(cache)
        local.mkdir(parents=True, exist_ok=True)
        env = cache["env_var"]

        if not backing.exists():
            info(f"  {env}: no backing dir yet -- skip")
            continue

        max_bytes = get_cache_seed_budget_bytes(cache)
        if max_bytes <= 0:
            info(f"  {env}: no seed budget available -- skip")
            continue

        # List all files in backing with mtime and size.
        r = sh(
            f"find '{backing}' -type f -printf '%T@ %s %P\\n' 2>/dev/null",
            capture=True,
        )
        if r.returncode != 0 or not r.stdout.strip():
            info(f"  {env}: backing is empty -- skip")
            continue

        # Parse file list and sort by access frequency (desc),
        # then mtime (desc) as tiebreaker. Frequently-used models
        # are seeded first regardless of age.
        subdir = cache["host_subdir"]
        file_entries = []  # (freq, mtime, size, relpath)
        for line in r.stdout.strip().splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            mtime = float(parts[0])
            fsize = int(parts[1])
            relpath = parts[2]
            key = f"{subdir}/{relpath}"
            freq = access_counts.get(key, 0)
            file_entries.append((freq, mtime, fsize, relpath))

        # Sort: highest frequency first, then newest first.
        file_entries.sort(key=lambda e: (-e[0], -e[1]))

        # Select files that fit within the budget.
        budget = max_bytes
        selected = []  # type: list[str]
        total_size = 0
        seeded_freq = 0
        for freq, mtime, fsize, relpath in file_entries:
            if budget - fsize < 0:
                continue  # skip this file, try smaller ones
            selected.append(relpath)
            budget -= fsize
            total_size += fsize
            if freq > 0:
                seeded_freq += 1

        if not selected:
            info(f"  {env}: no files fit within budget -- skip")
            continue

        # Write file list to a temp file for rsync --files-from.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as flist:
            flist.write("\n".join(selected))
            flist_path = flist.name

        start = time.monotonic()
        total_mb = total_size / (1024 * 1024)
        info(
            f"  {env}: seeding {len(selected)} files "
            f"({total_mb:.0f}MB), "
            f"{seeded_freq} by frequency, "
            f"{len(selected) - seeded_freq} by mtime"
        )
        r = sh(
            f"rsync --archive --timeout=120 "
            f"--files-from='{flist_path}' "
            f"'{backing}/' '{local}/' 2>&1",
            capture=True,
        )
        elapsed = time.monotonic() - start
        os.unlink(flist_path)

        if r.returncode == 0:
            info(f"  {env}: seeded in {elapsed:.1f}s")
        else:
            warn(
                f"  {env}: rsync seed failed "
                f"(rc={r.returncode}, {elapsed:.1f}s) -- "
                f"proceeding with partial cache"
            )


def sync_caches_to_backing():
    # type: () -> None
    """Persist new files from the hot tier back to the backing store.

    Called during cleanup. Only copies files that are newer locally
    than in backing (--update flag). This means we only transfer the
    delta -- files the test downloaded that were not in backing before.

    This is safe for concurrent pods writing to the same backing store:
      - --update never overwrites a newer backing file with an older one.
      - rsync uses atomic renames, so partial writes don't corrupt files.

    Skipped if CACHE_BACKING_ROOT is not set (single-tier mode).
    """
    if CACHE_BACKING_ROOT is None:
        return

    info("Persisting new cache files to backing store...")
    CACHE_BACKING_ROOT.mkdir(parents=True, exist_ok=True)

    for cache in CACHES:
        backing = _get_cache_backing_dir(cache)
        if backing is None:
            continue
        local = _get_cache_host_dir(cache)
        if not local.exists():
            continue

        backing.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        # --update: skip files that are newer on the receiver (backing).
        # --ignore-existing could also work but --update is safer for
        # concurrent writers since it compares timestamps.
        r = sh(
            f"rsync --archive --update --timeout=120 '{local}/' '{backing}/' 2>&1",
            capture=True,
        )
        elapsed = time.monotonic() - start
        env = cache["env_var"]
        if r.returncode == 0:
            info(f"  {env}: persisted in {elapsed:.1f}s")
        else:
            warn(
                f"  {env}: rsync to backing failed (rc={r.returncode}, {elapsed:.1f}s)"
            )


def _detect_storage_type(path):
    # type: (Path) -> str
    """Detect what kind of storage a path sits on.

    Returns one of: "tmpfs", "nfs", "local", "unknown".
    Used for logging so operators can verify the cache tiers are
    on the expected storage backends.
    """
    r = sh(
        f"df -T '{path}' 2>/dev/null | tail -1 | awk '{{print $2}}'",
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return "unknown"
    fstype = r.stdout.strip().lower()
    if fstype in ("tmpfs", "ramfs"):
        return "tmpfs (ephemeral RAM)"
    if "nfs" in fstype:
        return f"nfs ({fstype})"
    if fstype in ("ext4", "xfs", "btrfs"):
        return f"local ({fstype})"
    if "fuse" in fstype:
        return f"fuse ({fstype})"
    return fstype


def _get_cache_env_override_gb(cache):
    # type: (dict) -> int | None
    """Return the user-provided max_gb override for a cache, if any."""
    env_key = f"VLLM_CACHE_MAX_{cache['env_var']}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            warn(
                f"Invalid value for {env_key}='{env_val}' "
                f"(expected integer GB) -- using default"
            )
    return None


def _get_cache_max_gb(cache):
    # type: (dict) -> int
    """Resolve the effective L1 budget for a cache.

    Order of precedence:
      1. ``VLLM_CACHE_MAX_<ENV_VAR>`` override.
      2. The cache's configured default ``max_gb``.
      3. In single-tier mode, dynamically scale defaults down so the sum of
         all cache budgets on the shared filesystem fits within local disk
         headroom. This keeps guardrails meaningful on ~1TB agents while
         preserving the original larger defaults for two-tier deployments.
    """
    override = _get_cache_env_override_gb(cache)
    if override is not None:
        return override

    default_gb = int(cache.get("max_gb", 0))
    if default_gb <= 0 or CACHE_BACKING_ROOT is not None:
        return default_gb

    host_dir = _get_cache_host_dir(cache)
    probe = host_dir if host_dir.exists() else host_dir.parent
    try:
        probe_stat = probe.stat()
        usage = shutil.disk_usage(str(probe))
    except OSError:
        return default_gb

    total_default_gb = 0.0
    for entry in CACHES:
        if _get_cache_env_override_gb(entry) is not None:
            continue
        entry_default_gb = int(entry.get("max_gb", 0))
        if entry_default_gb <= 0:
            continue
        entry_host = _get_cache_host_dir(entry)
        entry_probe = entry_host if entry_host.exists() else entry_host.parent
        try:
            if entry_probe.stat().st_dev != probe_stat.st_dev:
                continue
        except OSError:
            continue
        total_default_gb += float(entry_default_gb)

    if total_default_gb <= 0:
        return default_gb

    total_gb = usage.total / (1024**3)
    usable_gb = max(1.0, total_gb - min(L1_FS_MIN_HEADROOM_GB, total_gb * 0.1))
    if total_default_gb <= usable_gb:
        return default_gb

    scaled_gb = max(1, int(default_gb * (usable_gb / total_default_gb)))
    return min(default_gb, scaled_gb)


def _get_dir_size_bytes(path):
    # type: (Path) -> int
    """Return total size of all files under path, in bytes."""
    r = sh(
        f"du -sb '{path}' 2>/dev/null | cut -f1",
        capture=True,
    )
    if r.returncode == 0 and r.stdout.strip().isdigit():
        return int(r.stdout.strip())
    return 0


def evict_cache_lru(cache):
    # type: (dict) -> None
    """Evict oldest-accessed files from a single cache until under max_gb.

    Uses file access time (atime) as the LRU signal. Files that haven't
    been read in the longest time are evicted first. This works because:

    - HuggingFace: models not used recently are least valuable.
    - pip: old wheels for previous dependency versions.
    - ccache: compiled objects for old code revisions.
    - Test data: datasets from tests that no longer run.

    Eviction is file-level, not directory-level. Empty directories are
    cleaned up after eviction.

    Args:
        cache: A single entry from the CACHES registry.
    """
    host_dir = _get_cache_host_dir(cache)
    max_gb = _get_cache_max_gb(cache)
    env = cache["env_var"]

    if max_gb <= 0:
        return  # unlimited

    if not host_dir.exists():
        return

    current_bytes = _get_dir_size_bytes(host_dir)
    max_bytes = max_gb * 1024 * 1024 * 1024

    if current_bytes <= max_bytes:
        return

    current_gb = current_bytes / (1024 * 1024 * 1024)
    info(
        f"  {env}: {current_gb:.1f}GB > {max_gb}GB limit "
        f"-- evicting oldest-accessed files"
    )

    # List all files sorted by modification time (oldest first).
    # Uses mtime (%T@) instead of atime (%A@) because many filesystems
    # mount with noatime/relatime, making atime unreliable.
    r = sh(
        f"find '{host_dir}' -type f -printf '%T@ %s %p\\n' 2>/dev/null | sort -n",
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        warn(f"  {env}: could not list files for eviction")
        return

    evicted_count = 0
    evicted_bytes = 0
    for line in r.stdout.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            file_size = int(parts[1])
        except ValueError:
            continue
        file_path = parts[2]

        # Safety: resolve symlinks and .. to prevent path traversal.
        try:
            resolved = str(Path(file_path).resolve())
        except OSError:
            continue
        if not resolved.startswith(str(host_dir.resolve())):
            continue

        try:
            os.unlink(file_path)
            evicted_count += 1
            evicted_bytes += file_size
            current_bytes -= file_size
        except OSError:
            continue

        if current_bytes <= max_bytes:
            break

    # Clean up empty directories left behind.
    sh(f"find '{host_dir}' -type d -empty -delete 2>/dev/null || true")

    evicted_mb = evicted_bytes / (1024 * 1024)
    remaining_gb = current_bytes / (1024 * 1024 * 1024)
    info(
        f"  {env}: evicted {evicted_count} files "
        f"({evicted_mb:.0f}MB), now {remaining_gb:.1f}GB"
    )


def evict_all_caches():
    # type: () -> None
    """Run LRU eviction on all caches that exceed their max_gb limit.

    Called during Docker housekeeping (Phase 5) so caches are trimmed
    before the test runs, not after. This prevents a situation where
    a test downloads a large model, fills the cache past the limit,
    and the next job on the same node starts with a full disk.
    """
    section("Cache eviction (per-cache LRU)")
    any_evicted = False
    for cache in CACHES:
        max_gb = _get_cache_max_gb(cache)
        if max_gb <= 0:
            continue
        host_dir = _get_cache_host_dir(cache)
        if not host_dir.exists():
            continue
        current_bytes = _get_dir_size_bytes(host_dir)
        max_bytes = max_gb * 1024 * 1024 * 1024
        if current_bytes > max_bytes:
            any_evicted = True
            evict_cache_lru(cache)
        else:
            current_gb = current_bytes / (1024 * 1024 * 1024)
            info(f"  {cache['env_var']:25s} {current_gb:.1f}GB / {max_gb}GB -- OK")
    if not any_evicted:
        info("All L1 caches within limits")


def evict_l2_cache(cache):
    # type: (dict) -> None
    """Evict stale and oversized data from L2 (backing/NFS) for one cache.

    Two eviction criteria (both enforced):

    1. Time-based (l2_max_days): any file not accessed in more than
       l2_max_days is deleted, regardless of L2 size. This prevents
       abandoned models, old wheel versions, and stale compiled kernels
       from accumulating indefinitely on shared NFS.

    2. Size-based (l2_max_gb): if L2 still exceeds l2_max_gb after
       time-based eviction, we evict the oldest-accessed files until
       under the limit (same LRU-by-atime as L1).

    Only runs when CACHE_BACKING_ROOT is configured (two-tier mode).

    Args:
        cache: A single entry from the CACHES registry.
    """
    backing = _get_cache_backing_dir(cache)
    if backing is None or not backing.exists():
        return

    env = cache["env_var"]
    max_days = int(cache.get("l2_max_days", L2_DEFAULT_MAX_DAYS))
    l2_max_gb = int(cache.get("l2_max_gb", 0))

    if max_days <= 0 and l2_max_gb <= 0:
        return  # both disabled

    # Check write access before attempting any deletions.
    if not os.access(str(backing), os.W_OK):
        warn(
            f"  {env} L2: backing at {backing} is read-only "
            f"-- cannot evict. Mount the PVC as ReadWriteMany "
            f"(RWX) or run L2 eviction from a pod with write "
            f"access."
        )
        return

    # -- Time-based eviction --
    if max_days > 0:
        # Find files not modified in more than max_days.
        # Uses mtime (not atime) because many filesystems mount with
        # noatime/relatime, making atime unreliable.
        r = sh(
            f"find '{backing}' -type f -mtime +{max_days} "
            f"-printf '%s %P\\n' 2>/dev/null",
            capture=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            stale_files = r.stdout.strip().splitlines()
            stale_bytes = 0
            stale_count = 0
            for line in stale_files:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    fsize = int(parts[0])
                except ValueError:
                    continue
                fpath = backing / parts[1]
                try:
                    resolved = str(fpath.resolve())
                except OSError:
                    continue
                if not resolved.startswith(str(backing.resolve())):
                    continue  # safety
                with suppress(OSError):
                    os.unlink(str(fpath))
                    stale_count += 1
                    stale_bytes += fsize

            if stale_count > 0:
                stale_mb = stale_bytes / (1024 * 1024)
                info(
                    f"  {env} L2: evicted {stale_count} files "
                    f"({stale_mb:.0f}MB) older than {max_days}d"
                )

    # -- Size-based eviction --
    if l2_max_gb > 0:
        current_bytes = _get_dir_size_bytes(backing)
        max_bytes = l2_max_gb * 1024 * 1024 * 1024
        if current_bytes > max_bytes:
            current_gb = current_bytes / (1024 * 1024 * 1024)
            info(
                f"  {env} L2: {current_gb:.1f}GB > "
                f"{l2_max_gb}GB limit -- evicting oldest"
            )
            # LRU by mtime, oldest first (atime is unreliable on
            # noatime/relatime filesystems).
            r = sh(
                f"find '{backing}' -type f "
                f"-printf '%T@ %s %p\\n' "
                f"2>/dev/null | sort -n",
                capture=True,
            )
            if r.returncode == 0 and r.stdout.strip():
                evicted_count = 0
                evicted_bytes = 0
                for line in r.stdout.strip().splitlines():
                    parts = line.split(None, 2)
                    if len(parts) < 3:
                        continue
                    try:
                        fsize = int(parts[1])
                    except ValueError:
                        continue
                    fpath = parts[2]
                    try:
                        resolved = str(Path(fpath).resolve())
                    except OSError:
                        continue
                    if not resolved.startswith(str(backing.resolve())):
                        continue
                    with suppress(OSError):
                        os.unlink(fpath)
                        evicted_count += 1
                        evicted_bytes += fsize
                        current_bytes -= fsize
                    if current_bytes <= max_bytes:
                        break
                if evicted_count > 0:
                    mb = evicted_bytes / (1024 * 1024)
                    gb = current_bytes / (1024 * 1024 * 1024)
                    info(
                        f"  {env} L2: evicted "
                        f"{evicted_count} files "
                        f"({mb:.0f}MB), now {gb:.1f}GB"
                    )

    # Clean up empty directories.
    sh(f"find '{backing}' -type d -empty -delete 2>/dev/null || true")


def evict_all_l2_caches():
    # type: () -> None
    """Run time-based and size-based eviction on all L2 (backing) caches.

    Only runs when CACHE_BACKING_ROOT is configured. Runs during
    Phase 5 (housekeeping) alongside L1 eviction and Docker cleanup.

    Only runs on nightly builds (NIGHTLY=1) to avoid adding latency
    to every PR job. L2 accumulation is slow (days/weeks), so daily
    cleanup is sufficient.
    """
    if CACHE_BACKING_ROOT is None:
        return

    section("L2 cache eviction (time + size)")
    any_evicted = False
    for cache in CACHES:
        backing = _get_cache_backing_dir(cache)
        if backing is None or not backing.exists():
            continue
        max_days = int(cache.get("l2_max_days", L2_DEFAULT_MAX_DAYS))
        l2_max_gb = int(cache.get("l2_max_gb", 0))
        if max_days <= 0 and l2_max_gb <= 0:
            continue
        current_bytes = _get_dir_size_bytes(backing)
        current_gb = current_bytes / (1024 * 1024 * 1024)
        limit_str = f"{l2_max_gb}GB" if l2_max_gb > 0 else "unlimited"
        info(
            f"  {cache['env_var']:25s} L2: {current_gb:.1f}GB "
            f"(limit={limit_str}, max_age={max_days}d)"
        )
        evict_l2_cache(cache)
        any_evicted = True

    if not any_evicted:
        info("No L2 caches to evict")


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
      1. DNS resolution -- can we resolve the Docker registry and
         huggingface.co? Catches broken DNS in K8s (kube-dns/coredns
         down, pod DNS policy misconfigured). Only tests hosts that
         this job will actually contact (configured via VLLM_CI_REGISTRY).
         Without this, a DNS failure surfaces minutes later as a Docker
         ``dial tcp: lookup ...: no such host`` after Docker exhausts
         its internal retry loop.
      2. Docker registry reachability -- can we reach the image registry?
         Catches network partition, proxy issues, registry outages.
     2b. Network throughput -- is the network fast enough? Measures
         download speed against the registry. Warns below 1 MB/s
         (slow pulls), errors below 100 KB/s (likely timeout).
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

    IMPORTANT: Every check is wrapped in try/except so a single failure
    (timeout, missing binary, permission error) never crashes the script
    or skips subsequent checks. The function itself is also wrapped at
    the call site in main() inside ``with timed(...):``.
    """
    section("Infrastructure health checks")

    registry = os.environ.get("VLLM_CI_REGISTRY", "docker.io")

    # -- 1. DNS resolution --
    # Resolve the hosts that THIS job will actually contact: the Docker
    # registry (configurable) and huggingface.co (model downloads).
    # Only test relevant hosts -- hardcoding public endpoints like
    # ghcr.io is wrong when CI uses a private registry or mirror.
    # Without this pre-check, a DNS failure surfaces minutes later as
    # "dial tcp: lookup ...: no such host" after Docker exhausts its
    # internal retry loop. Catching it here fails in <5s with context.
    try:
        dns_hosts = []  # type: list[str]
        # Extract hostname from registry (strip port if present).
        registry_host = registry.split(":")[0].split("/")[0]
        dns_hosts.append(registry_host)
        # HuggingFace is always needed for model downloads.
        dns_hosts.append("huggingface.co")
        # Deduplicate while preserving order.
        seen = set()  # type: set[str]
        dns_hosts = [h for h in dns_hosts if not (h in seen or seen.add(h))]
        dns_ok = True
        for host in dns_hosts:
            try:
                r = sh(
                    f"getent hosts {host} 2>/dev/null",
                    capture=True,
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                dns_ok = False
                warn(
                    f"{_DIAG_PREFIX} DNS resolution timed out "
                    f"for '{host}' (>5s). DNS may be broken."
                )
                continue
            if r.returncode != 0 or not r.stdout.strip():
                dns_ok = False
                warn(
                    f"{_DIAG_PREFIX} DNS resolution failed "
                    f"for '{host}'. Docker pull and model "
                    f"downloads may fail. Check pod DNS policy "
                    f"and kube-dns/coredns health."
                )
        if dns_ok:
            info(f"DNS resolution: OK ({', '.join(dns_hosts)})")
    except Exception as exc:
        warn(f"DNS check failed unexpectedly: {exc}")

    # -- 2. Docker registry reachability --
    # Try to reach the registry API. We don't need to authenticate --
    # a TCP connection or HTTP response is enough to confirm the network
    # path is open.
    try:
        r = sh(
            f"curl -sf --connect-timeout 10 --max-time 15 "
            f"-o /dev/null -w '%{{http_code}}' "
            f"https://{registry}/v2/ 2>/dev/null",
            capture=True,
        )
        if r.returncode != 0:
            warn(
                f"{_DIAG_PREFIX} Cannot reach Docker registry "
                f"'{registry}'. docker pull will likely fail. "
                f"Check network connectivity, proxy settings, "
                f"and firewall rules."
            )
        else:
            info(f"Docker registry ({registry}): reachable")
    except Exception as exc:
        warn(f"Registry reachability check failed: {exc}")

    # -- 2b. Network throughput --
    # Download 1MB from a real endpoint and measure speed. We try
    # multiple URLs across different providers so a single repo or
    # CDN being down doesn't blind us. Each URL is a file that our
    # test suite actually uses (models downloaded during tests).
    # Last resort: ping our own GitHub repo (always available if
    # the network is functional at all).
    try:
        _throughput_probes = [
            # HF models used by the test suite (1MB range request each).
            (
                "HF/TitanML-tiny-mixtral",
                "https://huggingface.co/TitanML/tiny-mixtral"
                "/resolve/main/model.safetensors",
            ),
            (
                "HF/Qwen2.5-0.5B",
                "https://huggingface.co/Qwen/Qwen2.5-0.5B"
                "/resolve/main/model.safetensors",
            ),
            (
                "HF/whisper-tiny",
                "https://huggingface.co/openai/whisper-tiny"
                "/resolve/main/model.safetensors",
            ),
            # GitHub: our own repo (raw file, always available).
            (
                "GitHub/vllm-project",
                "https://raw.githubusercontent.com/vllm-project/vllm/main/README.md",
            ),
        ]
        speed_measured = False
        for probe_name, probe_url in _throughput_probes:
            try:
                r = sh(
                    f"curl -sf --connect-timeout 5 "
                    f"--max-time 15 -r 0-1048575 "
                    f"-o /dev/null "
                    f"-w '%{{speed_download}}' "
                    f"'{probe_url}' 2>/dev/null",
                    capture=True,
                )
            except subprocess.TimeoutExpired:
                continue
            if r.returncode != 0 or not r.stdout.strip():
                continue
            try:
                speed_bps = float(r.stdout.strip())
            except (ValueError, OverflowError):
                continue
            # Skip bogus measurements (empty response, <1KB transferred).
            if speed_bps < 1:
                continue
            speed_mbps = speed_bps / (1024 * 1024)
            speed_measured = True
            if speed_bps < 1024 * 100:  # < 100 KB/s
                warn(
                    f"{_DIAG_PREFIX} Network throughput: "
                    f"{speed_mbps:.2f} MB/s via {probe_name} "
                    f"(very slow, <100KB/s). "
                    f"Docker pull and model downloads will be "
                    f"severely affected. Check network bandwidth, "
                    f"proxy throttling, and NIC health."
                )
            elif speed_bps < 1024 * 1024:  # < 1 MB/s
                warn(
                    f"Network throughput: {speed_mbps:.2f} MB/s "
                    f"via {probe_name} "
                    f"(slow, may cause pull timeouts)"
                )
            else:
                info(f"Network throughput: {speed_mbps:.1f} MB/s via {probe_name} (OK)")
            break  # first successful probe is enough
        if not speed_measured:
            warn(
                f"{_DIAG_PREFIX} Network throughput: could not "
                f"measure (all {len(_throughput_probes)} probes "
                f"failed). Network may be unreachable."
            )
    except Exception as exc:
        warn(f"Network throughput check failed: {exc}")

    # -- 3. Available memory --
    try:
        used_bytes, limit_bytes = get_cgroup_memory_usage()
        if used_bytes is not None and limit_bytes is not None and limit_bytes > 0:
            used_gb = used_bytes / (1024**3)
            limit_gb = limit_bytes / (1024**3)
            pct = int(100 * used_bytes / limit_bytes)
            info(f"Pod memory usage: {used_gb:.1f} GB / {limit_gb:.1f} GB ({pct}%)")
            if pct >= POD_MEMORY_WARN_PCT:
                warn(
                    f"{_DIAG_PREFIX} Pod memory usage is {pct}% of limit. "
                    "Tests may OOM or be evicted."
                )
        else:
            r = sh("cat /proc/meminfo 2>/dev/null", capture=True)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / (1024 * 1024)
                        info(f"Available host memory: {avail_gb:.1f} GB")
                        if avail_gb < 8:
                            warn(
                                f"{_DIAG_PREFIX} Low available host memory: "
                                f"{avail_gb:.1f} GB. Tests may OOM or the kubelet "
                                "may evict this pod. Check node memory pressure."
                            )
                        break
    except Exception as exc:
        warn(f"Memory check failed: {exc}")

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
    except Exception as exc:
        warn(f"Disk I/O check failed: {exc}")

    # -- 5. Pod uptime (K8s only) --
    try:
        if is_k8s():
            uptime_s = get_container_uptime_s()
            if uptime_s is not None:
                if uptime_s < 120:
                    warn(
                        f"{_DIAG_PREFIX} Pod/container uptime is only "
                        f"{uptime_s:.0f}s. This pod was recently started."
                    )
                else:
                    info(f"Pod/container uptime: {uptime_s:.0f}s")
    except Exception as exc:
        warn(f"Pod uptime check failed: {exc}")


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


def _load_commit_tar(image, cache_dir):
    # type: (str, str) -> bool
    """Tier 0: Load a per-commit image tar from local NVMe.

    A previous test job on this node may have saved the image after pulling
    it. Loading from NVMe avoids any network traffic.

    Returns True if the image is now available in Docker.
    """
    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        return False

    tar_path = os.path.join(cache_dir, f"commit-{commit}.tar")
    if not os.path.isfile(tar_path):
        return False

    info(f"Tier 0: Loading per-commit tar from NVMe ({tar_path})")
    r = sh(["docker", "load", "-i", tar_path], capture=True)
    if r.returncode != 0:
        warn(f"Failed to load commit tar, removing: {tar_path}")
        with suppress(OSError):
            os.remove(tar_path)
        return False

    # Verify the image is actually present after load
    r2 = sh(["docker", "image", "inspect", image], capture=True)
    if r2.returncode == 0:
        info("Tier 0: Image loaded from NVMe cache")
        return True

    warn("Commit tar loaded but image not found (tag mismatch?)")
    return False


def _try_download_wheel_artifact(wheel_tmp, artifact_dir):
    # type: (str, str) -> list[str]
    """Try to download wheel artifacts from a specific artifact directory.

    Returns a list of plain ``.whl`` file paths if found, empty list otherwise.
    """
    try:
        r = sh(
            [
                "buildkite-agent",
                "artifact",
                "download",
                f"{artifact_dir}/*",
                wheel_tmp + "/",
            ],
            capture=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0:
        return []

    wheel_paths = _glob.glob(os.path.join(wheel_tmp, artifact_dir, "*.whl"))
    if wheel_paths:
        return wheel_paths

    zst_paths = _glob.glob(os.path.join(wheel_tmp, artifact_dir, "*.zst"))
    if zst_paths:
        warn(
            f"Tier 1: Found compressed wheel artifacts in {artifact_dir} "
            "(expected plain .whl uploads)."
        )
    return []


def _assemble_from_wheel(image):
    # type: (str) -> bool
    """Tier 1: Assemble a test image locally from ci_base + wheel artifact.

    The build step uploads a plain ROCm wheel artifact at:

      artifacts/vllm-wheel-rocm/*.whl

    For a short migration window, we also accept the legacy path:

      artifacts/vllm-wheel-multi-arch/*.whl

    If ci_base is already loaded (by the hooks from NVMe tar) and the
    wheel is available, we build a lightweight image locally -- zero
    Docker Hub traffic.

    Returns True if the image was assembled successfully.
    """
    # Check buildkite-agent first (cheapest check).
    if not shutil.which("buildkite-agent"):
        info("Tier 1: buildkite-agent not found, skipping")
        return False

    # Check if wheel artifact exists BEFORE pulling ci_base.
    # Downloading the artifact is cheap (small file, local API call).
    # Pulling ci_base is expensive on a cold agent. Don't waste time
    # pulling ci_base if there's no wheel to install into it.
    info("Tier 1: Checking for wheel artifact...")
    wheel_tmp = tempfile.mkdtemp(prefix="vllm-wheel-")
    try:
        artifact_dirs = [
            "artifacts/vllm-wheel-rocm",
            "artifacts/vllm-wheel-multi-arch",
        ]
        wheel_artifacts = []
        for artifact_dir in artifact_dirs:
            info(f"Tier 1: Trying {artifact_dir}/ ...")
            wheel_artifacts = _try_download_wheel_artifact(wheel_tmp, artifact_dir)
            if wheel_artifacts:
                info(f"Tier 1: Found wheel artifact via {artifact_dir}/")
                break
        if not wheel_artifacts:
            info("Tier 1: No wheel artifact available from any path")
            return False

        for artifact_path in wheel_artifacts:
            shutil.copy2(
                artifact_path,
                os.path.join(wheel_tmp, os.path.basename(artifact_path)),
            )

        # Find the workspace (Buildkite checkout)
        workspace = os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH", "/workspace/build")
        whl_files = _glob.glob(os.path.join(wheel_tmp, "*.whl"))
        if not whl_files or not os.path.isdir(workspace):
            info("Tier 1: Wheel or workspace not found, skipping")
            return False

        # Wheel confirmed available. NOW pull ci_base (not before,
        # to avoid wasting time if the wheel doesn't exist).
        r = sh(
            ["docker", "image", "inspect", CI_BASE_IMAGE],
            capture=True,
        )
        if r.returncode != 0:
            info("Tier 1: Pulling ci_base...")
            r = sh(
                ["docker", "pull", CI_BASE_IMAGE],
                capture=True,
                timeout=300,
            )
            if r.returncode != 0:
                info("Tier 1: Could not pull ci_base, skipping local assembly")
                return False

        # Stage a temporary Docker build context instead of mutating the
        # checkout in-place. Some Buildkite agent pods mount the checkout
        # read-only or with a different owner uid, which makes workspace/dist
        # writes fail even though the checkout is readable.
        build_context = os.path.join(wheel_tmp, "build-context")
        shutil.copytree(
            workspace,
            build_context,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                ".mypy_cache",
                ".pytest_cache",
                "build",
                "dist",
                "wheel-export",
            ),
        )

        # The repo's .dockerignore excludes dist/, which would hide the
        # downloaded wheel we place into this temporary context. Override it
        # with a context-specific ignore file that keeps dist/*.whl visible
        # while still dropping the main noisy/generated directories.
        dockerignore = os.path.join(build_context, ".dockerignore")
        with open(dockerignore, "w") as f:
            f.write(
                ".git\n"
                "__pycache__/\n"
                ".mypy_cache/\n"
                ".pytest_cache/\n"
                ".venv/\n"
                "build/\n"
                "wheel-export/\n"
                "*.py[cod]\n"
            )

        # Copy wheels into the temporary build context.
        dist_dir = os.path.join(build_context, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        for whl in whl_files:
            shutil.copy2(whl, dist_dir)

        # Write a minimal Dockerfile for local assembly.
        # Use uv (already in ci_base) for fast installs consistent with
        # Dockerfile.rocm.
        dockerfile = os.path.join(build_context, "Dockerfile.ci-assemble")
        with open(dockerfile, "w") as f:
            f.write(
                f"FROM {CI_BASE_IMAGE}\n"
                "COPY dist/*.whl /opt/vllm-wheels/\n"
                "RUN uv pip install --system --no-deps /opt/vllm-wheels/*.whl\n"
                "WORKDIR /vllm-workspace\n"
                "COPY . /vllm-workspace\n"
                "RUN uv pip install --system -e tests/vllm_test_utils\n"
                "RUN mkdir -p src && mv vllm src/vllm 2>/dev/null || true\n"
            )

        info("Building local image: ci_base + wheel + workspace...")
        with timed("local image assembly"):
            r = sh(
                ["docker", "build", "-t", image, "-f", dockerfile, build_context],
                capture=True,
            )

        if r.returncode != 0:
            build_output = "\n".join(
                line
                for line in [(r.stdout or "").strip(), (r.stderr or "").strip()]
                if line
            )
            if build_output:
                build_output = "\n".join(build_output.splitlines()[-80:])
                warn(
                    "Tier 1: Local image build failed; docker build output tail:\n"
                    f"{build_output}"
                )
            warn("Tier 1: Local image build failed, falling back to pull")
            return False

        # Verify
        r2 = sh(["docker", "image", "inspect", image], capture=True)
        if r2.returncode == 0:
            info("Tier 1: Local assembly succeeded -- zero Docker Hub traffic")
            return True

        warn("Tier 1: Build succeeded but image not found")
        return False

    finally:
        shutil.rmtree(wheel_tmp, ignore_errors=True)


def _save_commit_tar(image, cache_dir):
    # type: (str, str) -> None
    """Save a pulled/assembled image to NVMe for same-node reuse.

    Other test jobs for the same commit on this node will load it via
    Tier 0 instead of pulling or assembling again.
    """
    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        return

    tar_path = os.path.join(cache_dir, f"commit-{commit}.tar")
    if os.path.isfile(tar_path):
        return  # already saved by another job

    tmp_path = tar_path + ".tmp"
    info(f"Saving image to NVMe for same-node reuse: {tar_path}")
    r = sh(["docker", "save", image, "-o", tmp_path], capture=True)
    if r.returncode == 0:
        try:
            os.rename(tmp_path, tar_path)
            info("Image saved to NVMe cache")
        except OSError:
            with suppress(OSError):
                os.remove(tmp_path)
    else:
        with suppress(OSError):
            os.remove(tmp_path)


def _get_disk_info(path):
    # type: (str) -> tuple[int | None, int | None, int | None]
    """Return (usage_pct, used_gb, total_gb) for the partition containing ``path``.

    Uses shutil.disk_usage (stdlib) for reliable, parse-free disk stats.
    Falls back to df if shutil fails (e.g., permission denied).
    """
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total // (1024 * 1024 * 1024)
        used_gb = usage.used // (1024 * 1024 * 1024)
        pct = int(100 * usage.used / usage.total) if usage.total > 0 else 0
        return pct, used_gb, total_gb
    except (OSError, ValueError):
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
    # DOCKER_IMAGE_NAME may override the default commit-tagged image, so keep
    # both names protected if both are present.
    current_commit = os.environ.get("BUILDKITE_COMMIT", "")
    protected_names = set()  # type: set[str]
    if current_commit:
        protected_names.add(f"rocm/vllm-ci:{current_commit}")
    docker_image_env = os.environ.get("DOCKER_IMAGE_NAME", "").strip()
    if docker_image_env:
        protected_names.add(docker_image_env)

    # Filter out protected and base images.
    candidates = []  # type: list[tuple[str, str, str, str]]
    for img_id, name, created, size in raw_images:
        if name in protected_names:
            info(f"  Protected (current job): {name}")
            continue
        if name.startswith("rocm/") and ":latest" in name:
            info(f"  Protected (base image): {name}")
            continue
        # Protect the ci_base image (shared Tier-1 layer from PR #36949).
        # This image is rebuilt weekly and shared by all ROCm jobs.
        if "ci_base" in name:
            info(f"  Protected (ci_base): {name}")
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


# ==========================================================================
# Background health watchdog
#
# A daemon thread that samples system health during the test run and
# writes to a log file in results_dir. Uploaded as a Buildkite artifact
# so post-mortem debugging has a timeline of resource usage.
#
# Why: when a pod is evicted, disk fills from a model download, or
# host memory spikes, we only find out after the container dies. The
# watchdog captures a timeline so we can see WHAT was happening in
# the seconds/minutes before the failure.
#
# The watchdog is:
#   - A daemon thread (killed automatically on process exit).
#   - Writes to a file, not stdout (avoids interleaving with test logs).
#   - Tolerant of all errors (never crashes the main script).
#   - Stoppable via an Event so cleanup can flush it before uploading.
# ==========================================================================


class _HealthWatchdog:
    """Background thread that periodically samples system health.

    Writes timestamped snapshots to a log file that is uploaded
    as a Buildkite artifact alongside container.log and results.xml.

    Monitors:
      - Host memory (MemAvailable from /proc/meminfo)
      - Disk usage (Docker root and cache root partitions)
      - GPU VRAM (via amd-smi/rocm-smi, best-effort)
      - Container status (running, OOMKilled, exited)
      - Container PID usage (via ``docker stats --no-stream``)

    Active duties (when two-tier cache is enabled):
      - Incremental model sync: rsyncs new files from L1 (NVMe) to L2
        (NFS) every cycle. This means models downloaded mid-test are
        available to other nodes before the test finishes.
      - Disk pressure relief: when L1 cache usage exceeds a threshold,
        accelerates sync to ensure data reaches L2 before eviction.

    All sampling is wrapped in try/except so a single failed read
    (e.g., /proc temporarily unavailable during cgroup migration)
    never kills the watchdog.
    """

    def __init__(self, log_path, container_name):
        # type: (Path, str | None) -> None
        self._log_path = log_path
        self._container = container_name
        self._stop = threading.Event()
        self._thread = None  # type: threading.Thread | None
        self._sync_in_progress = False
        self._last_sync_time = 0.0  # monotonic

    def start(self):
        # type: () -> None
        """Start the watchdog daemon thread."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="health-watchdog"
        )
        self._thread.start()
        info(
            f"Health watchdog started "
            f"(interval={WATCHDOG_INTERVAL_S}s, "
            f"log={self._log_path})"
        )

    def stop(self):
        # type: () -> None
        """Signal the watchdog to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        # type: () -> None
        """Main loop: sample, write, sleep, repeat."""
        with open(self._log_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Health watchdog log -- "
                f"container {self._container or 'multi-node'}\n"
                f"# Sampling every {WATCHDOG_INTERVAL_S}s\n"
                f"# Format: timestamp | metric | value\n\n"
            )
            f.flush()
            while not self._stop.is_set():
                with best_effort("watchdog sample"):
                    self._sample(f)
                self._stop.wait(timeout=WATCHDOG_INTERVAL_S)

    def _sample(self, f):
        # type: (...) -> None
        """Take one snapshot of all metrics."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # -- Host memory --
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    f.write(f"{ts} | mem_avail_gb | {gb:.1f}\n")
                    break
        except (OSError, ValueError, IndexError):
            pass

        # -- Pod/container memory --
        try:
            used_bytes, limit_bytes = get_cgroup_memory_usage()
            if used_bytes is not None:
                used_gb = used_bytes / (1024**3)
                if limit_bytes is not None and limit_bytes > 0:
                    limit_gb = limit_bytes / (1024**3)
                    pct = int(100 * used_bytes / limit_bytes)
                    f.write(
                        f"{ts} | pod_mem_gb | {used_gb:.1f}/{limit_gb:.1f} ({pct}%)\n"
                    )
                else:
                    f.write(f"{ts} | pod_mem_gb | {used_gb:.1f}/unbounded\n")
        except (OSError, ValueError, IndexError):
            pass

        # -- Disk usage (Docker root) --
        try:
            r = sh(
                "docker info -f '{{.DockerRootDir}}'",
                capture=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                pct, used_gb, total_gb = _get_disk_info(r.stdout.strip())
                if pct is not None:
                    f.write(f"{ts} | disk_docker_gb | {used_gb}/{total_gb} ({pct}%)\n")
        except (OSError, subprocess.SubprocessError):
            pass

        # -- Disk usage (cache root) --
        try:
            pct, used_gb, total_gb = _get_disk_info(str(CACHE_ROOT))
            if pct is not None:
                f.write(f"{ts} | disk_cache_gb | {used_gb}/{total_gb} ({pct}%)\n")
        except (OSError, subprocess.SubprocessError):
            pass

        # -- GPU VRAM (best-effort, short timeout) --
        try:
            data = _smi_json("metric --mem-usage", timeout=10)
            if isinstance(data, list):
                for entry in data:
                    gpu = entry.get("gpu", "?")
                    mem = entry.get("mem_usage", {})
                    used = mem.get("used_vram", {})
                    total = mem.get("total_vram", {})
                    u = used.get("value", "?")
                    t = total.get("value", "?")
                    f.write(f"{ts} | gpu{gpu}_vram_mb | {u}/{t}\n")
            elif _SMI_TOOL == "rocm-smi":
                # Fallback: just log whether VRAM is non-zero.
                r = _smi_cmd("--showmemuse --json", timeout=10)
                if r.returncode == 0:
                    f.write(f"{ts} | gpu_vram_raw | {r.stdout.strip()[:200]}\n")
        except (
            OSError,
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
        ):
            pass

        # -- Container status --
        if self._container:
            try:
                r = sh(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Status}} {{.State.OOMKilled}} {{.State.ExitCode}}",
                        self._container,
                    ],
                    capture=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    f.write(f"{ts} | container | {r.stdout.strip()}\n")
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                r = sh(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.PIDs}}",
                        self._container,
                    ],
                    capture=True,
                    timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    f.write(f"{ts} | container_pids | {r.stdout.strip()}\n")
            except (OSError, subprocess.SubprocessError):
                pass

        # -- Incremental cache sync (L1 -> L2) --
        if CACHE_BACKING_ROOT is not None and not self._sync_in_progress:
            with best_effort("watchdog cache sync"):
                self._maybe_sync_caches(f, ts)

        f.flush()

    def _maybe_sync_caches(self, f, ts):
        # type: (...) -> None
        """Incrementally rsync new files from L1 (NVMe) to L2 (NFS).

        Runs at most every WATCHDOG_SYNC_INTERVAL_S seconds. Under L1 disk
        pressure, runs every WATCHDOG_SYNC_PRESSURE_INTERVAL_S seconds.

        Before rsyncing, checks that L2 has room for the delta. If L2 would
        overflow its l2_max_gb budget, the rsync is skipped for that cache
        and L1 LRU eviction is triggered instead (oldest files removed from
        NVMe to free space for ongoing downloads).

        The rsync uses --update so it only copies files newer on L1 than L2.
        Safe for concurrent pods writing to the same NFS backing store.
        """
        now = time.monotonic()

        # Check if any L1 cache is under disk pressure
        under_pressure = False
        for cache in CACHES:
            try:
                if cache_under_pressure(cache):
                    under_pressure = True
                    break
            except OSError:
                pass

        interval = (
            WATCHDOG_SYNC_PRESSURE_INTERVAL_S
            if under_pressure
            else WATCHDOG_SYNC_INTERVAL_S
        )
        if now - self._last_sync_time < interval:
            return

        self._sync_in_progress = True
        try:
            synced = 0
            skipped_full = 0
            evicted = 0

            for cache in CACHES:
                backing = _get_cache_backing_dir(cache)
                if backing is None:
                    continue
                local = _get_cache_host_dir(cache)
                if not local.exists():
                    continue

                backing.mkdir(parents=True, exist_ok=True)
                env = cache["env_var"]
                l2_max_gb = int(cache.get("l2_max_gb", 0))

                # Check L2 has room before rsyncing. If L2 is at or over
                # budget, skip the sync and evict from L1 instead.
                if l2_max_gb > 0:
                    l2_bytes = _get_dir_size_bytes(backing)
                    l2_gb = l2_bytes / (1024 * 1024 * 1024)
                    if l2_gb >= l2_max_gb:
                        f.write(
                            f"{ts} | cache_sync | {env} L2 full "
                            f"({l2_gb:.1f}/{l2_max_gb}GB) -- "
                            f"skipping sync, evicting L1 LRU\n"
                        )
                        skipped_full += 1
                        # Evict oldest files from L1 to free space for
                        # ongoing downloads. Uses the existing LRU eviction
                        # which respects the cache's max_gb budget.
                        try:
                            evict_cache_lru(cache)
                            evicted += 1
                        except Exception:
                            pass
                        continue

                # --update: only copy files newer on local than backing.
                # --timeout=30: don't hang if NFS is slow.
                # --whole-file: skip delta-transfer (NVMe read is faster
                #   than computing checksums over NFS).
                r = subprocess.run(
                    [
                        "rsync",
                        "--archive",
                        "--update",
                        "--whole-file",
                        "--timeout=30",
                        str(local) + "/",
                        str(backing) + "/",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                if r.returncode == 0:
                    synced += 1
                else:
                    f.write(
                        f"{ts} | cache_sync | {env} rsync failed (rc={r.returncode})\n"
                    )

            parts = []
            if synced > 0:
                parts.append(f"synced {synced}")
            if skipped_full > 0:
                parts.append(f"L2-full {skipped_full}")
            if evicted > 0:
                parts.append(f"L1-evicted {evicted}")
            if parts:
                pressure_tag = " [PRESSURE]" if under_pressure else ""
                f.write(f"{ts} | cache_sync | {', '.join(parts)}{pressure_tag}\n")
            self._last_sync_time = now
        finally:
            self._sync_in_progress = False


def build_single_node_docker_cmd(
    *,
    image,  # type: str
    name,  # type: str
    commands,  # type: str
    render_gid,  # type: str
    results_dir,  # type: Path
    render_devices,  # type: str
    rdma,  # type: bool
):
    # type: (...) -> list[str]
    """Build the ``docker run`` command for a single-node AMD test job."""
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
        safe_tokens = []  # type: list[str]
        for token in tokens:
            if (
                token == "--device"
                or token.startswith("--device=")
                or token.startswith("/dev/")
            ):
                safe_tokens.append(token)
            else:
                warn(
                    f"Dropping unexpected render_devices token: '{token}' "
                    f"(expected --device or /dev/ path)"
                )
        docker_cmd.extend(safe_tokens)

    # RDMA passthrough for ibverbs-based tests (e.g., test_moriio_connector).
    if rdma:
        docker_cmd += ["--device", "/dev/infiniband", "--cap-add=IPC_LOCK"]

    docker_cmd += [
        "--network=host",
        "--group-add",
        render_gid,
        f"--pids-limit={CONTAINER_PIDS_LIMIT}",
    ]

    # Docker documents ``--shm-size`` as the size of the container's /dev/shm
    # tmpfs, while ``--ipc=host`` switches the container to the Docker host's
    # IPC namespace. Only private/shareable IPC modes should therefore use the
    # container-scoped ``--shm-size`` setting.
    if _container_uses_private_shm(CONTAINER_IPC_MODE):
        docker_cmd.append(f"--shm-size={CONTAINER_SHM_SIZE}")
    docker_cmd.append(f"--ipc={CONTAINER_IPC_MODE}")

    docker_cmd += build_test_engine_docker_env_args()

    docker_cmd += [
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
        f"GIT_ROOT={CONTAINER_REPO_ROOT}",
        "-e",
        "PYTHONPATH=..",
        # NCCL tuning for ROCm multi-GPU tests.
        # NCCL_DEBUG=WARN: log NCCL warnings (INFO is too noisy for CI).
        "-e",
        "NCCL_DEBUG=WARN",
        # Tell pytest to write JUnit XML to the bind-mounted results dir.
        # This is the KEY to the exit-code fix: the XML is written BEFORE
        # Python's atexit handlers run, and it persists on the host.
        # Append to any existing PYTEST_ADDOPTS to avoid silently dropping
        # upstream settings (e.g., --timeout from the pipeline).
        "-e",
        "PYTEST_ADDOPTS={}--junitxml={}/results.xml".format(
            _v + " " if (_v := os.environ.get("PYTEST_ADDOPTS", "").strip()) else "",
            RESULTS_MOUNT,
        ),
    ]

    # Persistent cache mounts -- all caches defined in the CACHES registry.
    docker_cmd += build_cache_docker_args()

    # Unset PYTORCH_ROCM_ARCH inside the container so PyTorch
    # auto-detects the GPU arch at runtime. The image is pre-compiled;
    # a stale value from the host/build env can cause PyTorch to skip
    # the right kernels or attempt recompilation.
    # See: https://github.com/vllm-project/vllm/pull/38272
    container_commands = f"unset PYTORCH_ROCM_ARCH && {commands}"

    docker_cmd += [
        "--name",
        name,
        image,
        "/bin/bash",
        "-euo",
        "pipefail",
        "-c",
        container_commands,
    ]

    return docker_cmd


def log_container_ipc_runtime(container_name):
    # type: (str) -> None
    """Log the effective Docker IPC mode and visible ``/dev/shm`` size."""
    r = sh(
        [
            "docker",
            "inspect",
            "--format",
            "IpcMode={{.HostConfig.IpcMode}} ShmSize={{.HostConfig.ShmSize}}",
            container_name,
        ],
        capture=True,
        timeout=10,
    )
    if r.returncode == 0 and r.stdout.strip():
        info(f"Container IPC config: {r.stdout.strip()}")

    r = sh(
        ["docker", "exec", container_name, "df", "-h", "/dev/shm"],
        capture=True,
        timeout=10,
    )
    if r.returncode == 0 and r.stdout.strip():
        info("Container /dev/shm:")
        for line in r.stdout.strip().splitlines():
            info(f"  {line}")


def log_container_pid_runtime(container_name):
    # type: (str) -> None
    """Log the effective Docker PID budget and current task usage."""
    r = sh(
        [
            "docker",
            "inspect",
            "--format",
            "PidsLimit={{.HostConfig.PidsLimit}}",
            container_name,
        ],
        capture=True,
        timeout=10,
    )
    if r.returncode == 0 and r.stdout.strip():
        info(f"Container PID config: {r.stdout.strip()}")

    r = sh(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "PIDs={{.PIDs}}",
            container_name,
        ],
        capture=True,
        timeout=10,
    )
    if r.returncode == 0 and r.stdout.strip():
        info(f"Container PID usage: {r.stdout.strip()}")


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
    docker_cmd = build_single_node_docker_cmd(
        image=image,
        name=name,
        commands=commands,
        render_gid=render_gid,
        results_dir=results_dir,
        render_devices=render_devices,
        rdma=rdma,
    )

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

    try:
        r = sh(docker_cmd, check=True, capture=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        error(
            f"{_DIAG_PREFIX} DOCKER RUN FAILED\n"
            f"  What happened: 'docker run --detach' exited "
            f"with code {exc.returncode}\n"
            f"  Error:  {stderr[:500]}\n"
            f"  Common causes:\n"
            f"    - Image not found (check docker pull step above)\n"
            f"    - Device not available (/dev/kfd, /dev/dri/*)\n"
            f"    - Disk full (check Docker root partition)\n"
            f"    - Docker daemon error (check 'docker info')"
        )
        return 1
    container_id = r.stdout.strip()
    info(f"Container started: {container_id[:12]} (full ID: {container_id})")
    with best_effort("container IPC runtime logging"):
        log_container_ipc_runtime(name)
    with best_effort("container PID runtime logging"):
        log_container_pid_runtime(name)

    # -- Step 2: Stream logs to stdout (Buildkite) AND a file (artifact) --
    log_file = results_dir / "container.log"
    info(f"Container log will be saved to: {log_file}")
    log_proc = None  # type: subprocess.Popen | None
    try:
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
    except OSError as exc:
        warn(f"Failed to start container log streaming: {exc}")
        if log_proc is not None:
            with suppress(OSError):
                log_proc.kill()
            with suppress(OSError):
                log_proc.wait(timeout=5)
        sh(["docker", "rm", "-f", name], capture=True)
        return 1
    # Close our copy of the pipe so log_proc gets SIGPIPE if tee dies.
    log_proc.stdout.close()

    # -- Step 2b: Start background health watchdog --
    watchdog_log = results_dir / "health_watchdog.log"
    watchdog = _HealthWatchdog(watchdog_log, name)
    watchdog.start()

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
        sh(["docker", "kill", name], capture=True)
        exit_code = 124  # matches GNU timeout convention

    # -- Step 4: Flush log streaming --
    info("Flushing container log stream...")
    try:
        tee_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        warn("Log stream flush timed out after 30s -- killing tee process")
        tee_proc.kill()
        tee_proc.wait()
    # Reap the docker-logs process to avoid a zombie.
    with suppress(subprocess.TimeoutExpired):
        log_proc.wait(timeout=5)
    if log_proc.poll() is None:
        log_proc.kill()
        log_proc.wait()

    info(f"Container exit code (docker wait): {exit_code}")

    # -- Step 4b: Stop health watchdog --
    watchdog.stop()
    if watchdog_log.is_file():
        info(
            f"Health watchdog log: {watchdog_log} ({watchdog_log.stat().st_size} bytes)"
        )

    # -- Step 5: Post-test snapshots (compare with step 0 for leaks/growth) --
    snapshot_gpu_vram("post-test")
    log_cache_stats_diff("post-test")

    # -- Step 6: Diagnose exit (OOM, signals, PID limit) BEFORE docker rm --
    if ENABLE_DIAGNOSTICS:
        diag = diagnose_container_exit(name, log_file=log_file)
    else:
        diag = {
            "oom_killed": False,
            "exit_code": exit_code,
            "error": "",
            "pids_exhausted": False,
            "pid_error": "",
            "shm_exhausted": False,
            "shm_error": "",
            "pytest_ran": True,
            "pre_pytest_traceback": False,
            "pre_pytest_crash": "",
        }

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
            "### :no_entry: PID / Thread Budget Exhausted "
            f"({_format_container_pids_limit(CONTAINER_PIDS_LIMIT)})\n\n"
            "The container could not fork new processes/threads.\n"
            "See `[run-amd-test.py diagnostics]` in the build log for details.",
            style="error",
            context="pids-exhausted",
        )
        if exit_code == 0:
            exit_code = 1

    if diag["shm_exhausted"]:
        annotate_build(
            "### :file_cabinet: Shared Memory Exhausted\n\n"
            "The workload could not allocate a `/dev/shm` segment.\n"
            "See `[run-amd-test.py diagnostics]` in the build log for the "
            "effective IPC mode and fix guidance.",
            style="error",
            context="shm-exhausted",
        )
        if exit_code == 0:
            exit_code = 1

    # -- Step 6b: Pre-pytest log hints (ADVISORY ONLY) --
    #
    # Tracebacks and crash patterns in the pre-pytest log zone
    # (before "test session starts") MIGHT indicate a setup crash,
    # but they can also be informational: pip/uv print tracebacks
    # during normal dependency resolution, build scripts log handled
    # exceptions, etc. We CANNOT reliably distinguish a real crash
    # from a logged-and-handled exception in the pre-test output.
    #
    # These are NEVER used to override the exit code. The exit code
    # is governed by:
    #   - set -euo pipefail (bash failures)
    #   - JUnit XML override (pytest atexit masking)
    #   - XML-missing fail-safe (pytest never ran + exit 0)
    #
    # These hints only produce warnings so operators can investigate.
    if diag.get("pre_pytest_traceback") and exit_code == 0:
        warn(
            f"{_DIAG_PREFIX} Python traceback found in "
            f"pre-test setup output (before pytest started) "
            f"but the container exited 0. This may be an "
            f"informational exception from pip/uv, or it may "
            f"indicate a masked crash. Check the container "
            f"log to verify."
        )
    if diag.get("pre_pytest_crash") and exit_code == 0:
        warn(
            f"{_DIAG_PREFIX} Pre-test setup output contains "
            f"'{diag['pre_pytest_crash']}' but the container "
            f"exited 0. This may be normal (e.g., a fallback "
            f"path was taken) or it may indicate a masked "
            f"failure. Check the container log."
        )

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
    #
    # PYTEST_ADDOPTS is injected as a container env var for EVERY job,
    # so any pytest invoked inside the container (whether directly in
    # the command string or inside a bash wrapper script) will produce
    # JUnit XML at the bind-mounted results path.
    #
    # Not all CI steps invoke pytest at all. Scheduled integration
    # tests (server+eval, accuracy benchmarks) and standalone examples
    # run bash/Python scripts that never import pytest. For these,
    # no XML is produced and the container's exit code (protected by
    # set -euo pipefail) is the only result source.
    #
    # Strategy:
    #   1. Always check if XML exists (it costs nothing).
    #   2. If XML exists + has failures + exit 0 -- override (atexit fix).
    #   3. If XML exists + 0 failures + exit 0 -- pass.
    #   4. If XML missing + exit 0:
    #      a. If pytest actually ran (session header in log) -- fail-safe
    #         (pytest ran but XML didn't reach host).
    #      b. If pytest never ran -- trust exit code (non-pytest test).
    #   5. If exit non-zero -- propagate regardless.
    xml_path = results_dir / "results.xml"
    if not ENABLE_JUNIT_OVERRIDE:
        info("JUnit XML override DISABLED (VLLM_ROCM_CI_JUNIT_OVERRIDE=0)")
    elif exit_code == 0:
        # If the XML isn't on the host (bind-mount failure, overlay issue,
        # permissions), try to rescue it from the still-running container
        # via docker cp BEFORE giving up.
        if not xml_path.is_file():
            warn(
                f"JUnit XML not found at bind-mount path {xml_path} "
                f"-- attempting docker cp fallback"
            )
            r_cp = sh(
                f"docker cp {name}:{RESULTS_MOUNT}/results.xml {xml_path} 2>/dev/null",
                capture=True,
            )
            if r_cp.returncode == 0 and xml_path.is_file():
                info(
                    f"Recovered JUnit XML via docker cp "
                    f"({xml_path.stat().st_size} bytes)"
                )
            else:
                warn(f"docker cp fallback also failed (rc={r_cp.returncode})")

        failures = parse_junit_failures(xml_path)
        info(
            f"JUnit XML validation: path={xml_path}, "
            f"exists={xml_path.is_file()}, failures={failures}"
        )
        if failures is not None and failures > 0:
            error(
                f"{_DIAG_PREFIX} EXIT CODE OVERRIDE "
                f"(atexit hook detected)\n"
                f"  What happened: The container exited 0 but "
                f"JUnit XML reports\n"
                f"                 {failures} failure(s)/error(s)."
                f" This means a\n"
                f"                 library's atexit hook called "
                f"os._exit(0) and\n"
                f"                 overwrote pytest's real exit "
                f"code.\n"
                f"  Container:     {name}\n"
                f"  JUnit XML:     {xml_path}\n"
                f"  Action:        Overriding exit code from 0 "
                f"to 1."
            )
            exit_code = 1
        elif failures is None:
            # XML missing or unparsable. The next step depends on
            # whether pytest actually ran inside the container.
            pytest_ran = diag.get("pytest_ran", False)
            if pytest_ran:
                # Pytest ran (session header found in log) but the
                # XML didn't make it to the host. This is a bind-mount
                # or docker-cp failure. We MUST fail because we can't
                # verify whether tests passed.
                error(
                    f"{_DIAG_PREFIX} JUNIT XML MISSING "
                    f"(pytest ran but XML not found)\n"
                    f"  What happened: Container exited 0 and "
                    f"the pytest session\n"
                    f"                 header was found in the "
                    f"log, but the JUnit\n"
                    f"                 XML could not be read. "
                    f"The bind mount or\n"
                    f"                 docker cp failed.\n"
                    f"  Expected at:   {xml_path}\n"
                    f"  Container:     {name}\n"
                    f"  Bind mount:    "
                    f"-v {results_dir}:{RESULTS_MOUNT}\n"
                    f"  Action:        Overriding exit code "
                    f"from 0 to 1 (fail-safe)."
                )
                annotate_build(
                    "### :warning: JUnit XML missing\n\n"
                    "Pytest ran but the XML result file was "
                    "not found on the host. Cannot verify "
                    "test results.\n"
                    "**Failing as a safety measure.** Check "
                    "the bind mount.\n\n"
                    f"Expected: `{xml_path}`\n"
                    f"Mount: "
                    f"`-v {results_dir}:{RESULTS_MOUNT}`",
                    style="error",
                    context="junit-missing",
                )
                exit_code = 1
            else:
                # Pytest never ran (no session header in log).
                # This is a non-pytest test (bash script, standalone
                # Python eval, accuracy benchmark). The exit code
                # from set -euo pipefail is the only result source.
                # Trusting it.
                info(
                    "No JUnit XML and no pytest session "
                    "detected -- non-pytest test. "
                    "Trusting container exit code (0)."
                )

    # -- Step 7b: Non-zero exit with no JUnit XML --
    # When exit is non-zero and no XML exists, provide diagnostics
    # about whether pytest ran or not. For non-pytest tests, this
    # just confirms the non-zero exit code is from the test itself.
    if exit_code != 0 and not xml_path.is_file():
        pytest_ran = diag.get("pytest_ran", True)
        if not pytest_ran:
            error(
                f"{_DIAG_PREFIX} PYTEST NEVER RAN\n"
                f"  What happened: Container exited "
                f"{exit_code} and no JUnit XML\n"
                f"                 was produced. The "
                f"container log shows no pytest\n"
                f"                 output markers. A "
                f"pre-test command likely failed.\n"
                f"  Container:     {name}\n"
                f"  How to fix:\n"
                f"    - Check the container log for the "
                f"first error (pip install,\n"
                f"      cd, missing binary, import "
                f"failure)\n"
                f"    - The test commands were:\n"
                f"      {commands[:200]}"
            )
            annotate_build(
                f"### :x: Pytest never ran (exit code "
                f"{exit_code})\n\n"
                f"No JUnit XML was produced. A pre-test "
                f"setup command likely failed.\n"
                f"Check the build log for the root cause.",
                style="error",
                context="pytest-never-ran",
            )
        else:
            warn(
                f"Container exited {exit_code} with no "
                f"JUnit XML -- pytest may have crashed "
                f"before writing XML (segfault, import "
                f"error, fixture failure)"
            )

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
    if (
        word[0] == "-"
        and not word.startswith("--")
        and len(word) >= 2
        and word[1:].isalpha()
    ):
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

    # Insert --ignore flags after a directory token (first occurrence only,
    # to avoid double-injection if the pattern appears multiple times).
    for pattern, files in _INLINE_IGNORES.items():
        if pattern in command:
            flags = " ".join(f"--ignore={f}" for f in files)
            command = command.replace(pattern, f"{pattern}{flags} ", 1)

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


def _inject_junit_into_multi_node_cmd(cmd, node_idx, pair_idx, results_host_dir):
    # type: (str, int, int, Path) -> str
    """Wrap a multi-node test command to produce JUnit XML.

    The original multi-node bash script (run-multi-node-test.sh) runs
    commands via ``docker exec``, which has the same torch atexit exit-code
    problem as ``docker run``. We inject PYTEST_ADDOPTS to produce JUnit
    XML inside the container, then copy it out after the test completes.

    When nightly Test Engine collection is enabled, this wrapper also exports
    the Buildkite analytics environment variables into the shell that launches
    pytest inside each node container. That keeps the implementation AMD-local
    to this runner without modifying the shared multi-node shell helper.

    Each (node, pair) combination gets a unique XML filename to avoid
    collisions when multiple command pairs are executed:
      node0, pair0 -> results_node0_pair0.xml
      node1, pair0 -> results_node1_pair0.xml
      node0, pair1 -> results_node0_pair1.xml

    Args:
        cmd:              Original pytest command string for this node.
        node_idx:         Node index (0 = head, 1+ = workers).
        pair_idx:         Command pair index (0-based).
        results_host_dir: Host directory for results (bind-mounted or copied).

    Returns:
        Modified command string with JUnit XML output enabled.
    """
    # The XML path is inside the container. We'll copy it out after the
    # test using ``docker cp``.
    xml_path = f"/tmp/results_node{node_idx}_pair{pair_idx}.xml"
    git_root_exports = build_git_root_shell_exports()
    test_engine_exports = build_test_engine_shell_exports()
    # Inject PYTEST_ADDOPTS before the command. If the command already
    # sets PYTEST_ADDOPTS, this prepends (pytest merges them).
    export_prefix = f"{git_root_exports}{test_engine_exports}"
    return (
        f'{export_prefix}export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} '
        f'--junitxml={xml_path}" && {cmd}'
    )


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
      - Inject Buildkite Test Engine env exports into each per-node command
        when ``NIGHTLY=1`` and ``BUILDKITE_ANALYTICS_TOKEN`` is present.
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
        error(
            f"node0 has {len(node0_cmds)} commands, "
            f"node1 has {len(node1_cmds)} -- counts must match. "
            f"Extra commands would be silently dropped."
        )
        return 1

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
    test_engine = get_test_engine_config()
    info(f"Buildkite Test Engine: {test_engine['reason']}")

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
    # Each (node, pair) gets a unique XML filename so that multi-pair jobs
    # don't overwrite earlier pairs' results (which would silently lose failures).
    modified_node0 = []  # type: list[str]
    modified_node1 = []  # type: list[str]
    num_pairs = len(node0_cmds)
    for i, (cmd0, cmd1) in enumerate(zip(node0_cmds, node1_cmds)):
        mod0 = _inject_junit_into_multi_node_cmd(cmd0, 0, i, results_dir)
        mod1 = _inject_junit_into_multi_node_cmd(cmd1, 1, i, results_dir)
        modified_node0.append(mod0)
        modified_node1.append(mod1)
        info(f"  Pair [{i}] node0 (with JUnit): {mod0[:120]}...")
        info(f"  Pair [{i}] node1 (with JUnit): {mod1[:120]}...")

    # Build the composite command that run-multi-node-test.sh will execute.
    composite = f"(command {_SMI_TOOL} || true)"
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
    watchdog_log = results_dir / "health_watchdog.log"
    watchdog = _HealthWatchdog(watchdog_log, None)
    watchdog.start()
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
            "Set `VLLM_TEST_TIMEOUT` env var to override (default: 10200s).",
            style="error",
            context="timeout",
        )
    finally:
        watchdog.stop()
    exec_elapsed = time.monotonic() - exec_start

    info(f"Multi-node composite exit code: {exit_code} (ran for {exec_elapsed:.1f}s)")

    # Copy JUnit XMLs from node containers to the host results directory.
    # The containers may still exist if run-multi-node-test.sh's trap hasn't
    # fired yet, or if it used --rm and they're already gone.
    # Each (node, pair) combination has a unique XML file to ensure that
    # multi-pair jobs don't lose earlier pairs' failure data.
    info("Collecting JUnit XML from node containers...")
    total_failures = 0
    for pair_idx in range(num_pairs):
        for node_idx in range(num_nodes):
            container_xml = f"/tmp/results_node{node_idx}_pair{pair_idx}.xml"
            host_xml = results_dir / f"results_node{node_idx}_pair{pair_idx}.xml"
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
                        f"  Node {node_idx} pair {pair_idx}: "
                        f"JUnit XML ({xml_size}B) {failures} failure(s)"
                    )
                    total_failures += failures
                else:
                    warn(
                        f"  Node {node_idx} pair {pair_idx}: "
                        f"JUnit XML ({xml_size}B) could not be parsed"
                    )
            else:
                stderr_msg = (
                    r.stderr or ""
                ).strip() or "container may have been removed"
                warn(
                    f"  Could not copy JUnit XML from node{node_idx} "
                    f"pair {pair_idx}: {stderr_msg}"
                )

    # JUnit XML validation: override exit code if any node reported failures.
    if exit_code == 0 and total_failures > 0:
        warn(
            f"Multi-node composite exited 0 but JUnit XML reports "
            f"{total_failures} total failure(s) -- overriding exit code to 1."
        )
        exit_code = 1

    # Buildkite annotation for multi-node failures.
    if exit_code != 0:
        # Try to build annotation from whichever node/pair XML exists.
        for pair_idx in range(num_pairs):
            for node_idx in range(num_nodes):
                host_xml = results_dir / f"results_node{node_idx}_pair{pair_idx}.xml"
                annotation = build_failure_annotation(host_xml)
                if annotation:
                    annotate_build(
                        f"**Node {node_idx} pair {pair_idx}:**\n\n{annotation}",
                        style="error",
                        context=f"test-failures-node{node_idx}-pair{pair_idx}",
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
    upload_artifacts(f"{results_dir}/*.xml", f"{results_dir}/*.log")

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
      3. Kill GPU zombie processes via amd-smi/rocm-smi (cross PID namespace).
      4. Reset GPU state file (write "reset", locked).
      5. Remove stale containers from this commit.
      6. Keep the Docker image for local cache reuse.
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
        self._running = False  # type: bool
        self._signal_triggered = False  # type: bool

    def run(self):
        # type: () -> None
        """Execute all cleanup steps. Idempotent (safe to call multiple times)."""
        if self._done or self._running:
            return
        self._running = True

        try:
            section("Cleanup: tearing down test environment")

            # When triggered by a signal (SIGTERM/SIGINT), Buildkite will send
            # SIGKILL shortly after (~10s). Skip slow operations (rsync to NFS,
            # access count updates) that can't complete in time and would just
            # get killed mid-write, potentially corrupting the backing store.
            if self._signal_triggered:
                info("Signal-triggered cleanup: skipping cache sync (time-limited)")
            else:
                # 0a. Update access frequency counts from this job's file atimes.
                if CACHE_BACKING_ROOT is not None:
                    with best_effort("cleanup access count update"):
                        info("  [0/8] Updating cache access counts...")
                        counts = _load_access_counts()
                        for cache in CACHES:
                            counts = update_access_counts_from_atime(cache, counts)
                        _save_access_counts(counts)
                        log_top_k_access_counts()
                else:
                    info("  [0/8] Access count update skipped (single-tier cache)")

            # 0b. Unmount overlay caches (must happen before rsync to backing,
            # because the upper layer files are only visible after unmount
            # when overlay is active).
            with best_effort("cleanup overlay unmount"):
                unmount_overlay_caches()

            if not self._signal_triggered:
                # 0c. Persist caches to backing store before tearing anything
                # down. Do this first so cache data survives even if later
                # cleanup steps fail or the pod is killed mid-cleanup.
                with best_effort("cleanup cache sync"):
                    sync_caches_to_backing()

            # 1. Remove test container(s).
            if self.container:
                with best_effort("cleanup container removal"):
                    info(f"  [1/8] Removing test container: {self.container}")
                    sh(["docker", "rm", "-f", self.container], capture=True)
            else:
                info("  [1/8] No container to remove (not started)")
            if self.is_multi_node:
                with best_effort("cleanup multi-node containers"):
                    info("  [1/8] Cleaning up multi-node containers and network")
                    _cleanup_multi_node()

            # 2. Kill GPU zombies (fuser -- same PID namespace).
            with best_effort("cleanup GPU zombies (fuser)"):
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

            # 3. Kill GPU zombies (amd-smi/rocm-smi -- cross PID namespace).
            with best_effort("cleanup GPU zombies (driver)"):
                info(
                    "  [3/8] Checking for GPU zombies "
                    "(amd-smi/rocm-smi, cross-namespace)"
                )
                smi_pids = gpu_pids()
                if smi_pids:
                    info(f"         Found {len(smi_pids)} process(es) via {_SMI_TOOL}")
                for pid in smi_pids:
                    _kill_pid(pid)

            # 4. Reset GPU state (locked, non-blocking).
            with best_effort("cleanup GPU state reset"):
                if _gpu_state_file_available() and os.access(
                    str(GPU_STATE_FILE), os.W_OK
                ):
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
            with best_effort("cleanup stale containers"):
                if self.commit:
                    info(
                        f"  [5/8] Removing stale containers for commit "
                        f"{self.commit[:12]}"
                    )
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

            # 6. Docker image: keep for same-commit retries.
            # Removing the image after every run defeats the Docker layer
            # cache and forces a full pull on retries of the same commit.
            # The LRU/LFU eviction in cleanup_docker_disk() handles disk
            # pressure -- no need to eagerly remove here.
            if self.image:
                info(f"  [6/8] Keeping Docker image: {self.image} (for retry cache)")
            else:
                info("  [6/8] No image tracked")

            # 7. Remove results directory (artifacts already uploaded).
            with best_effort("cleanup results directory"):
                if self.results_dir and self.results_dir.exists():
                    info(f"  [7/8] Removing results directory: {self.results_dir}")
                    shutil.rmtree(self.results_dir, ignore_errors=True)
                else:
                    info("  [7/8] No results directory to remove")

            # 8. Record completion in Buildkite metadata.
            with best_effort("cleanup metadata"):
                info("  [8/8] Recording cleanup_completed in Buildkite metadata")
                set_buildkite_meta("cleanup_completed", "true")

            info("Cleanup complete")
        finally:
            self._running = False
            self._done = True


_cleanup = _Cleanup()


def _on_signal(signum, _frame):
    # type: (int, object) -> None
    """Signal handler for SIGTERM and SIGINT.

    Runs cleanup and exits with 128+signum (POSIX convention).
    Skips slow operations (cache sync) since Buildkite SIGKILL follows soon.
    """
    _cleanup._signal_triggered = True
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
    global CACHE_RUNTIME_FAILED
    # -- Register cleanup for any exit path --
    atexit.register(_cleanup.run)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGQUIT, _on_signal)

    # -- Phase 1: Environment + config --
    section("Environment")
    normalize_cache_root()
    validate_container_pids_config()
    validate_container_ipc_config()
    with best_effort("K8s context logging"):
        log_k8s_context()
    with best_effort("config logging"):
        log_effective_config()

    # -- Phase 1b: Hard resets (destructive, one-shot) --
    with best_effort("hard resets"):
        execute_hard_resets()

    # -- Phase 2: Infrastructure health --
    # Docker health is always checked (fatal if Docker is down).
    # Infra checks are diagnostic only -- they must NEVER crash the script.
    check_docker_health()
    if ENABLE_INFRA_CHECKS:
        try:
            with timed("Infrastructure health checks"):
                check_infra_health()
        except Exception as exc:
            warn(
                f"Infrastructure health checks crashed "
                f"unexpectedly: {exc}. Continuing anyway."
            )
    else:
        info("Infrastructure checks DISABLED (VLLM_ROCM_CI_INFRA_CHECKS=0)")

    # -- Phase 3: GPU pre-flight --
    # Zombie cleanup and VRAM checks are best-effort. If they crash
    # (e.g., rocm-smi hangs, fuser missing), we still attempt the test.
    # GPU health validation (Phase 4) IS fatal: no GPUs = no point.
    if ENABLE_GPU_PREFLIGHT:
        try:
            with timed("GPU pre-flight"):
                kill_gpu_zombies()
                wait_for_clean_gpus()
        except Exception as exc:
            warn(f"GPU pre-flight crashed: {exc}. Continuing.")

        section("ROCm info (host)")
        # Log the host (K8s node / agent) ROCm version -- this is the
        # driver-level version, not what's inside the Docker image.
        # Useful for debugging driver/image mismatches.
        try:
            r = sh(
                "amd-smi version 2>/dev/null",
                capture=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                info(f"Host ROCm (amd-smi): {r.stdout.strip()}")
            else:
                raise FileNotFoundError
        except Exception:
            # amd-smi not available -- fall back to rocm-smi or
            # /opt/rocm/.info/version file.
            try:
                r = sh(
                    "rocm-smi --version 2>/dev/null",
                    capture=True,
                    timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    info(f"Host ROCm (rocm-smi): {r.stdout.strip()}")
                else:
                    raise FileNotFoundError
            except Exception:
                # Last resort: read the version file directly.
                for vpath in [
                    "/opt/rocm/.info/version",
                    "/opt/rocm/include/rocm-core/rocm_version.h",
                ]:
                    try:
                        ver = Path(vpath).read_text().strip()
                        info(f"Host ROCm ({vpath}): {ver}")
                        break
                    except OSError:
                        continue
                else:
                    warn("Could not determine host ROCm version")

        with best_effort("rocminfo"):
            sh("rocminfo", timeout=ROCM_SMI_TIMEOUT_S)

        # -- Phase 4: GPU health --
        validate_gpu_health()
    else:
        info("GPU pre-flight DISABLED (VLLM_ROCM_CI_GPU_PREFLIGHT=0)")

    # -- Phase 5: Docker + cache housekeeping --
    # Housekeeping is best-effort: a crash in stale container cleanup
    # or cache eviction should not prevent the test from running.
    with timed("Docker and cache housekeeping"):
        with best_effort("stale container cleanup"):
            cleanup_stale_containers()
        if ENABLE_DOCKER_EVICTION:
            with best_effort("Docker disk eviction"):
                cleanup_docker_disk()
        else:
            info("Docker eviction DISABLED (VLLM_ROCM_CI_DOCKER_EVICTION=0)")
        if ENABLE_CACHE_EVICTION:
            with best_effort("L1 cache eviction"):
                evict_all_caches()
            with best_effort("L2 cache eviction"):
                evict_all_l2_caches()
        else:
            info("Cache eviction DISABLED (VLLM_ROCM_CI_CACHE_EVICTION=0)")

    # -- Phase 6: GPU reset --
    if ENABLE_GPU_PREFLIGHT:
        try:
            with timed("GPU reset"):
                reset_gpus()
        except Exception as exc:
            warn(f"GPU reset crashed: {exc}. Continuing.")
    else:
        info("GPU reset DISABLED (VLLM_ROCM_CI_GPU_PREFLIGHT=0)")

    # -- Phase 7: Image pull --
    section("Pulling container")
    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        error("BUILDKITE_COMMIT is not set")
        sys.exit(1)

    image = os.environ.get("DOCKER_IMAGE_NAME", "").strip()
    if image:
        info(f"Image tag source: DOCKER_IMAGE_NAME ({image})")
        _validate_image_arch(image)
    else:
        image = f"{DEFAULT_IMAGE_REPO}:{commit}"
        info(f"Image tag source: default commit image ({image})")
    container = f"rocm_{commit}_{os.urandom(5).hex()}"
    info(f"Image: {image}")
    info(f"Container name: {container}")

    _cleanup.image = image
    _cleanup.container = container
    _cleanup.commit = commit

    # Tiered image acquisition: try local sources first, fall back to pull.
    # VLLM_LOCAL_IMAGE_CACHE is set by the Buildkite agent hooks on K8s
    # nodes with NVMe-backed cache (maintained by base-tar-updater DaemonSet).
    image_ready = False

    # Check if image is already in Docker (e.g., loaded by hooks or a
    # previous run on the same DinD instance).
    r = sh(["docker", "image", "inspect", image], capture=True)
    if r.returncode == 0:
        info("Image already present in Docker")
        image_ready = True

    # Tier 0: per-commit tar on local NVMe
    if not image_ready and LOCAL_IMAGE_CACHE:
        with timed("Tier 0: NVMe commit tar"):
            image_ready = _load_commit_tar(image, LOCAL_IMAGE_CACHE)

    # Tier 1: local assembly from ci_base + wheel artifact.
    # Does NOT require LOCAL_IMAGE_CACHE -- only needs ci_base in Docker
    # and a wheel artifact in Buildkite. Works on any agent.
    if not image_ready:
        with best_effort("Tier 1: local assembly"):
            image_ready = _assemble_from_wheel(image)

    # Tier 2/3: docker pull
    if not image_ready:
        try:
            with timed(f"docker pull {image}"):
                r = sh(["docker", "pull", image], capture=True, timeout=600)
                if r.returncode == 0:
                    info(f"Tier 2: Pulled image {image}")
                    image_ready = True
        except subprocess.TimeoutExpired:
            warn(f"Tier 2: docker pull {image} timed out after 600s")

    # Final fallback: retry with full retry logic (may exit on failure)
    if not image_ready:
        with timed(f"docker pull {image} (final attempt)"):
            docker_pull_with_retry(image)

    # Save to NVMe for same-node reuse by subsequent jobs
    if LOCAL_IMAGE_CACHE:
        _save_commit_tar(image, LOCAL_IMAGE_CACHE)

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

    # Set up all persistent caches (HF, ModelScope, test data, pip, etc.).
    # Cache setup failure should not block the test -- caches are a
    # performance optimization, not a correctness requirement.
    if ENABLE_CACHE:
        CACHE_RUNTIME_FAILED = False
        try:
            setup_caches()
        except Exception as exc:
            CACHE_RUNTIME_FAILED = True
            warn(f"Cache setup crashed: {exc}. Tests will run without cache.")
    else:
        info("Persistent caches DISABLED (VLLM_ROCM_CI_CACHE_ENABLED=0)")

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
