# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/dev/shm capacity for the CPU KV offload buffers of data-parallel engines
that share one host."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.device_communicators import shm_broadcast
from vllm.v1.engine import utils as engine_utils
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


def test_engine_launch_refuses_offload_that_overfills_shm(monkeypatch):
    """The check must run where the local engines are started: inside each
    engine, a dense model's data-parallel rank sees only itself."""
    _fake_shm(monkeypatch, 252 * GiB)
    _fake_platform(monkeypatch, "cuda")

    def no_spawn():
        raise AssertionError("engines started before the /dev/shm check")

    monkeypatch.setattr(engine_utils, "get_mp_context", no_spawn)
    with pytest.raises(
        RuntimeError, match="offload buffers of 8 data-parallel engines"
    ):
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


@pytest.mark.parametrize(
    "spec_name,platform,connector,engines,refused",
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
    monkeypatch, spec_name, platform, connector, engines, refused
):
    _fake_shm(monkeypatch, 252 * GiB)
    _fake_platform(monkeypatch, platform)
    monkeypatch.setattr(shm_broadcast, "check_cgroup_memory_available", MagicMock())
    vllm_config = _offload_vllm_config(spec_name, connector)
    check = check_local_engines_offload_shm
    if refused:
        with pytest.raises(RuntimeError, match="Insufficient space in /dev/shm"):
            check(vllm_config, engines)
    else:
        check(vllm_config, engines)
