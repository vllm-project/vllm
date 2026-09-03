# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cgroup-aware memory detection for the CPU backend.

`get_memory_node_info` feeds `CPUWorker.determine_available_memory`'s
fraction-of-available-memory KV-cache auto-sizing. Without the cgroup clamp
tested here, that path sizes off host/NUMA-node RAM instead of the
container's memory.max and over-allocates on a memory-limited K8s pod or CI
container, OOM-killing the process (or a sibling, e.g. dockerd in a DinD
sidecar) instead of vLLM itself raising a clean error.
"""

import sys
from io import StringIO
from types import SimpleNamespace

import pytest

from vllm.utils import cpu_resource_utils as cru
from vllm.utils.mem_constants import GiB_bytes

_V2_LIMIT_PATH = "/sys/fs/cgroup/memory.max"
_V2_USAGE_PATH = "/sys/fs/cgroup/memory.current"
_V1_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_V1_USAGE_PATH = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_NUMA_MEMINFO_PATH = "/sys/devices/system/node/node0/meminfo"


@pytest.fixture(autouse=True)
def _clear_cgroup_cache():
    cru.get_cgroup_memory_limit.cache_clear()
    yield
    cru.get_cgroup_memory_limit.cache_clear()


def _stub_files(monkeypatch, files: dict):
    """Stub ``open()`` for a fixed set of paths; ``None`` -> OSError (missing)."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path in files:
            content = files[path]
            if content is None:
                raise OSError(f"no such file: {path}")
            return StringIO(content)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def _stub_no_cgroup_files(monkeypatch):
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: None})


def _fake_vm(total, available):
    return SimpleNamespace(total=total, available=available)


# --------------------------------------------------------------------------
# get_cgroup_memory_limit
# --------------------------------------------------------------------------


def test_cgroup_v2_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {_V2_LIMIT_PATH: f"{20 * GiB_bytes}\n", _V2_USAGE_PATH: f"{5 * GiB_bytes}\n"},
    )
    assert cru.get_cgroup_memory_limit() == (20 * GiB_bytes, 5 * GiB_bytes)


def test_cgroup_v2_unlimited_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: "max\n",
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{1 * GiB_bytes}\n",
        },
    )
    assert cru.get_cgroup_memory_limit() == (8 * GiB_bytes, 1 * GiB_bytes)


def test_cgroup_v1_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: None,
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{2 * GiB_bytes}\n",
        },
    )
    assert cru.get_cgroup_memory_limit() == (8 * GiB_bytes, 2 * GiB_bytes)


def test_cgroup_v1_unlimited_sentinel_is_ignored(monkeypatch):
    """cgroup v1 reports a huge sentinel (close to PAGE_COUNTER_MAX) for an
    unconstrained cgroup; that must read as "no limit", not a literal
    multi-exabyte cap."""
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch, {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: f"{(1 << 63) - 1}\n"}
    )
    assert cru.get_cgroup_memory_limit() == (None, None)


def test_cgroup_no_limit_files_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_no_cgroup_files(monkeypatch)
    assert cru.get_cgroup_memory_limit() == (None, None)


def test_cgroup_skipped_on_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    # Even if a v2 file happened to exist, non-Linux must short-circuit.
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: f"{1 * GiB_bytes}\n"})
    assert cru.get_cgroup_memory_limit() == (None, None)


# --------------------------------------------------------------------------
# get_memory_node_info
# --------------------------------------------------------------------------


def test_numa_meminfo_clamped_to_cgroup_limit(monkeypatch):
    """A big multi-socket box running one pod with a tight cgroup limit
    must report the pod's limit, not the host's full NUMA-node RAM."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru.os.path, "exists", lambda path: path == _NUMA_MEMINFO_PATH)
    host_total_kb = 775 * GiB_bytes // 1024
    host_free_kb = host_total_kb - 10 * GiB_bytes // 1024
    meminfo = (
        f"Node 0 MemTotal:       {host_total_kb} kB\n"
        f"Node 0 MemFree:        {host_free_kb} kB\n"
        "Node 0 Active(file):   0 kB\n"
        "Node 0 Inactive(file): 0 kB\n"
        "Node 0 SReclaimable:   0 kB\n"
    )
    _stub_files(
        monkeypatch,
        {
            _NUMA_MEMINFO_PATH: meminfo,
            _V2_LIMIT_PATH: f"{20 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{5 * GiB_bytes}\n",
        },
    )
    info = cru.get_memory_node_info(0)
    assert info.total_memory == 20 * GiB_bytes
    assert info.available_memory == 15 * GiB_bytes


def test_non_numa_fallback_clamped_to_cgroup_limit(monkeypatch):
    """Non-NUMA hosts (no per-node meminfo) fall back to psutil, but that
    still reports host-wide RAM -- it must be clamped to the cgroup limit
    exactly like the NUMA path, or a constrained non-NUMA pod would size
    its KV cache off the host instead of its own memory.max."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        cru.psutil, "virtual_memory", lambda: _fake_vm(775 * GiB_bytes, 700 * GiB_bytes)
    )
    _stub_files(
        monkeypatch,
        {_V2_LIMIT_PATH: f"{20 * GiB_bytes}\n", _V2_USAGE_PATH: f"{5 * GiB_bytes}\n"},
    )
    info = cru.get_memory_node_info(0)
    assert info.total_memory == 20 * GiB_bytes
    assert info.available_memory == 15 * GiB_bytes


def test_non_numa_fallback_uses_host_values_without_cgroup_limit(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        cru.psutil, "virtual_memory", lambda: _fake_vm(775 * GiB_bytes, 700 * GiB_bytes)
    )
    _stub_no_cgroup_files(monkeypatch)
    info = cru.get_memory_node_info(0)
    assert info.total_memory == 775 * GiB_bytes
    assert info.available_memory == 700 * GiB_bytes


def test_cgroup_limit_at_or_above_host_total_is_not_a_clamp(monkeypatch):
    """An unconstrained/oversized cgroup limit (>= host RAM) must not
    "clamp" upward -- the host figure stays authoritative."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        cru.psutil, "virtual_memory", lambda: _fake_vm(20 * GiB_bytes, 15 * GiB_bytes)
    )
    _stub_files(
        monkeypatch, {_V2_LIMIT_PATH: f"{775 * GiB_bytes}\n", _V2_USAGE_PATH: "0\n"}
    )
    info = cru.get_memory_node_info(0)
    assert info.total_memory == 20 * GiB_bytes
    assert info.available_memory == 15 * GiB_bytes
