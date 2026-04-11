# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_importing_rocm_platform_does_not_force_spawn(monkeypatch):
    import vllm.platforms.rocm as rocm

    fallback_calls = 0

    def fake_get_device_properties(device):
        nonlocal fallback_calls
        fallback_calls += 1
        return types.SimpleNamespace(gcnArchName="gfx950")

    fake_amdsmi = types.ModuleType("amdsmi")
    fake_amdsmi.AmdSmiException = RuntimeError
    fake_amdsmi.amdsmi_get_gpu_asic_info = lambda handle: {}
    fake_amdsmi.amdsmi_get_gpu_device_uuid = lambda handle: "fake-uuid"
    fake_amdsmi.amdsmi_get_processor_handles = lambda: []
    fake_amdsmi.amdsmi_init = lambda: None
    fake_amdsmi.amdsmi_shut_down = lambda: None
    fake_amdsmi.amdsmi_topo_get_link_type = (
        lambda handle, peer: {"hops": 1, "type": 2}
    )

    with monkeypatch.context() as m:
        m.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        m.setattr("sys.argv", ["pytest"])
        m.setitem(sys.modules, "amdsmi", fake_amdsmi)
        m.setattr(torch.cuda, "_is_compiled", lambda: True)
        m.setattr(torch.cuda, "is_initialized", lambda: False)
        m.setattr(torch.cuda, "get_device_properties", fake_get_device_properties)
        m.setattr("vllm.utils.system_utils.is_in_ray_actor", lambda: False)
        m.setattr("vllm.utils.system_utils.in_wsl", lambda: False)
        m.setattr("vllm.utils.system_utils.xpu_is_initialized", lambda: False)

        rocm = importlib.reload(rocm)
        assert fallback_calls == 0

        _maybe_force_spawn()
        assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
        assert fallback_calls == 0

    importlib.reload(rocm)
