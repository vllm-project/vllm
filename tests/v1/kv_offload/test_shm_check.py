# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/dev/shm capacity for the CPU KV offload buffers of data-parallel engines
that share one host."""

import logging
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.device_communicators import shm_broadcast
from vllm.utils import cpu_resource_utils
from vllm.v1.engine import utils as engine_utils
from vllm.v1.kv_offload import shm_check
from vllm.v1.kv_offload.shm_check import check_local_engines_offload_shm

GiB = 1 << 30


def _fake_shm(monkeypatch, free: int) -> None:
    usage = shm_broadcast.shutil.disk_usage(shm_broadcast.SHM_PATH)._replace(
        total=free, free=free
    )
    monkeypatch.setattr(shm_broadcast.shutil, "disk_usage", lambda _: usage)


def _fake_platform(monkeypatch, kind: str) -> None:
    from vllm.v1.kv_offload.cpu import spec as cpu_spec_module

    platform = SimpleNamespace(
        is_cuda_alike=lambda: kind in ("cuda", "rocm"),
        is_rocm=lambda: kind == "rocm",
        is_xpu=lambda: kind == "xpu",
    )
    monkeypatch.setattr(cpu_spec_module, "current_platform", platform)


def _offload_vllm_config(spec_name: str = "CPUOffloadingSpec", connector=None):
    return VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector or "OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": spec_name,
                "cpu_bytes_to_use": 32 * GiB,
            },
        )
    )


def _stub_cgroup(monkeypatch) -> MagicMock:
    """Stub the cgroup preflight, whose reading of the host's own cgroup files
    is not what these tests are about, and return the stub the launcher's
    warning path calls."""
    cgroup = MagicMock()
    monkeypatch.setattr(cpu_resource_utils, "check_cgroup_memory_available", cgroup)
    monkeypatch.setattr(shm_broadcast, "check_cgroup_memory_available", MagicMock())
    return cgroup


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == shm_check.__name__ and r.levelno == logging.WARNING
    ]


def _warning(caplog) -> str:
    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    return warnings[0]


class _Spawned(Exception):
    """Raised in place of starting the engine processes."""


def test_engine_launch_warns_then_starts_the_engines(monkeypatch, caplog_vllm):
    """The warning must be emitted where the local engines are started, before
    any of them runs: inside an engine, a dense model's data-parallel rank sees
    only itself. It must not stop the launch, because the size compared here is
    an upper bound (see check_local_engines_offload_shm)."""
    _fake_shm(monkeypatch, 252 * GiB)
    _fake_platform(monkeypatch, "cuda")
    _stub_cgroup(monkeypatch)

    def spawned():
        raise _Spawned

    monkeypatch.setattr(engine_utils, "get_mp_context", spawned)
    with pytest.raises(_Spawned):
        engine_utils.CoreEngineProcManager(
            local_engine_count=8,
            start_index=0,
            local_start_index=0,
            vllm_config=_offload_vllm_config(),
            local_client=True,
            handshake_address="ipc:///unused",
            executor_class=MagicMock(),
            log_stats=False,
        )
    message = _warning(caplog_vllm)
    assert "offload buffers of 8 data-parallel engines" in message
    assert "still start" in message


def test_budget_that_rounds_down_to_fit_is_not_refused(monkeypatch, caplog_vllm):
    """CPUOffloadingSpec allocates num_chunks = cpu_bytes_to_use //
    aligned_kv_bytes_per_chunk, so a region is at most its budget: eight 32 GiB
    budgets with 320 MiB chunks allocate 255 GiB, which fits here. The launcher
    cannot know the chunk size, so it may warn, but it must not refuse."""
    _fake_shm(monkeypatch, 255 * GiB + GiB // 2)
    _fake_platform(monkeypatch, "cuda")
    _stub_cgroup(monkeypatch)

    check_local_engines_offload_shm(_offload_vllm_config(), 8)

    # 8 x 32 GiB budgeted against 255.5 GiB free: a warning, not a refusal.
    message = _warning(caplog_vllm)
    assert "262144 MiB required, 261632 MiB free" in message


@pytest.mark.parametrize(
    "spec_name,platform,connector,engines,warns",
    [
        # 8 x 32 GiB overfills 252 GiB of free /dev/shm; 7 x 32 GiB fits.
        ("CPUOffloadingSpec", "cuda", None, 8, True),
        ("CPUOffloadingSpec", "cuda", None, 7, False),
        # One engine is left to its own free-space check.
        ("CPUOffloadingSpec", "cuda", None, 1, False),
        # The CPU spec keeps a private tensor per rank on ROCm, not /dev/shm.
        ("CPUOffloadingSpec", "rocm", None, 8, False),
        # XPU maps the shared region too.
        ("CPUOffloadingSpec", "xpu", None, 8, True),
        # Tiering uses the shared region on every platform.
        ("TieringOffloadingSpec", "rocm", None, 8, True),
        ("CPUOffloadingSpec", "cuda", "SimpleCPUOffloadConnector", 8, False),
    ],
)
def test_local_engines_shm_check_scope(
    monkeypatch, caplog_vllm, spec_name, platform, connector, engines, warns
):
    _fake_shm(monkeypatch, 252 * GiB)
    _fake_platform(monkeypatch, platform)
    cgroup = _stub_cgroup(monkeypatch)

    check_local_engines_offload_shm(_offload_vllm_config(spec_name, connector), engines)

    if warns:
        assert "Insufficient space in /dev/shm" in _warning(caplog_vllm)
        # check_shm_free_space raises before its own cgroup preflight, so the
        # warning path runs it for the same total.
        cgroup.assert_called_once_with(engines * 32 * GiB, ANY)
    else:
        assert _warnings(caplog_vllm) == []
        cgroup.assert_not_called()
